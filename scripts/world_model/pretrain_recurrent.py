"""Budgeted recurrent sequence training with fixed held-out episodes."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from demo.world_model import WorldModelConfig, ScaledTanh, load_world_model, save_world_model
from demo.world_model_recurrent import RecurrentWorldModel
from demo.world_model_sequence import ExpertSequenceDataset, episode_split


def freeze_action_prior(model):
    """Outcome adaptation must not degrade the existing deployment action prior."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("reward_head.", "value_head.", "event_head.")))


def freeze_actor_only(model):
    """Fit the action head without moving its representation or outcome models."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("actor."))


def configure_action_limit(model, limit):
    # Keep parameter shapes/checkpoints compatible. Both bounded output heads
    # must match config so save/load preserves their forward behavior.
    model.config = replace(model.config, action_limit=float(limit))
    for head in (model.actor, model.inverse_head):
        for layer in head.modules():
            if isinstance(layer, ScaledTanh):
                layer.limit = float(limit)


def action_change_weights(batch):
    # Training targets only; never provided to observe()/deployment.
    change = (batch["action"]-batch["proprio"][...,16:24]).norm(dim=-1)
    batch["actor_weight"] = 1.0 + 4.0*(change > 0.02).float()
    return batch


class EarlyStopping:
    """Significant improvement resets patience; every true best is still saved."""
    def __init__(self, patience=3, min_delta=0.001, min_steps=300):
        self.patience, self.min_delta, self.min_steps = patience, min_delta, min_steps
        self.reference = float("inf")
        self.bad_checks = 0

    def update(self, value, step):
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite validation loss")
        if value < self.reference - self.min_delta:
            self.reference = value
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        return bool(self.patience and step >= self.min_steps
                    and self.bad_checks >= self.patience)


def event_windows(dataset, actor_only=False):
    """Find supervised windows containing real positives, without mixing episodes."""
    lookup = {(key, start): i for i, (key, start, aligned) in enumerate(dataset.windows)}
    indices = set()
    with h5py.File(dataset.path, "r") as f:
        for key in dataset.keys:
            g = f[key]
            if "training_labels" not in g:
                continue
            labels = g["training_labels"]
            positive = (labels["grasped_event"][...].any(-1)
                        | labels["in_basket_event"][...].any(-1)
                        | labels["basket_success"][...])
            if actor_only:
                positive &= bool(g.attrs.get("actor_supervision", True))
                if "actor_valid" in labels:
                    positive &= labels["actor_valid"][...]
            for t in np.flatnonzero(positive):
                start = max(0, min(int(t)-dataset.sequence_length+1,
                                   len(g["actions"])-dataset.sequence_length))
                if (key, start) in lookup and t >= start + min(dataset.burn_in, dataset.sequence_length-1):
                    indices.add(lookup[key, start])
    return sorted(indices)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--aligned_dataset", default="")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--init", default="")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--sequence_length", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--val_every", type=int, default=100)
    p.add_argument("--patience", type=int, default=3, help="0 disables early stopping")
    p.add_argument("--min_delta", type=float, default=0.001)
    p.add_argument("--min_steps", type=int, default=300)
    p.add_argument("--freeze_prior", action="store_true", help="Adapt outcome heads only; preserve actor/representation")
    p.add_argument("--actor_only", action="store_true", help="Freeze representation/outcomes; refine action head only")
    p.add_argument("--actor_action_limit", type=float, default=None, help="Optional candidate output bound for actor-only adaptation")
    args = p.parse_args()
    if args.iterations < 1 or args.sequence_length < 2:
        p.error("positive iteration budget and sequence_length >= 2 required")
    if args.val_every < 1 or args.patience < 0 or args.min_delta < 0 or args.min_steps < 0:
        p.error("Invalid early-stopping settings")
    if args.freeze_prior and (not args.init or not args.aligned_dataset):
        p.error("--freeze_prior requires --init and --aligned_dataset")
    if args.actor_only and (args.freeze_prior or not args.init):
        p.error("--actor_only requires --init and cannot combine with --freeze_prior")
    if args.actor_action_limit is not None and (not args.actor_only or not np.isfinite(args.actor_action_limit) or args.actor_action_limit <= 0):
        p.error("--actor_action_limit requires --actor_only and a finite positive bound")
    if Path(args.checkpoint).exists():
        raise FileExistsError("Choose a new checkpoint path; existing weights are preserved")
    torch.manual_seed(args.seed)
    datasets, validations, splits = [], [], {}
    outcome_supervised = False
    for path in filter(None, (args.dataset, args.aligned_dataset)):
        train_keys, val_keys = episode_split(path, args.seed)
        datasets.append(ExpertSequenceDataset(path, sequence_length=args.sequence_length, keys=train_keys))
        val = ExpertSequenceDataset(path, sequence_length=args.sequence_length, keys=val_keys)
        # Fixed, spread-out windows from whole held-out episodes, not changing
        # validation batches. This makes checkpoint comparisons meaningful.
        indices = torch.linspace(0, len(val)-1, min(8, len(val))).long().tolist()
        indices = sorted(set(indices) | set(event_windows(val, args.actor_only)))
        validations.extend([val[index] for index in indices])
        val.close()
        splits[path] = {"train": train_keys, "validation": val_keys}
        with h5py.File(path) as f:
            outcome_supervised |= any("policy_input" in f[k] for k in train_keys)

    model = RecurrentWorldModel(WorldModelConfig()).to(args.device)
    if args.init:
        old, payload = load_world_model(args.init, args.device)
        if args.actor_only and payload.get("format_version") != 2:
            raise ValueError("Actor-only adaptation requires an existing recurrent model")
        if payload.get("format_version") == 2:
            if model.config != old.config:
                model = RecurrentWorldModel(old.config).to(args.device)
            model.load_state_dict(old.state_dict())
        else:
            # Reuse demonstrated representation/action prior, but discard the
            # legacy reward/value heads trained with fabricated terminal labels.
            shared = {k: v for k, v in old.state_dict().items()
                      if not k.startswith(("reward_head.", "value_head."))}
            incompatible = model.load_state_dict(shared, strict=False)
            model.update_target(1.0)
            print(f"initialized existing encoder/dynamics/actor; new tensors={len(incompatible.missing_keys)}")
    if args.actor_only:
        if args.actor_action_limit is not None:
            configure_action_limit(model, args.actor_action_limit)
        freeze_actor_only(model)
        outcome_supervised = bool(payload.get("metadata", {}).get("outcome_supervised", False))
        print("frozen representation/dynamics/outcomes; adapting actor only", flush=True)
    if args.freeze_prior:
        freeze_action_prior(model)
        print("frozen representation/dynamics/actor; adapting reward/value/event heads only", flush=True)
    optimizer = torch.optim.AdamW(
        [v for v in model.parameters() if v.requires_grad], lr=3e-4
    )
    source = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    # Equal source sampling avoids small labeled episodes being swamped by
    # the 68k-step success-only legacy data.
    source_weights = []
    for ds in datasets:
        positives = event_windows(ds, args.actor_only)
        weights = torch.full((len(ds),), (0.5 if positives else 1.0)/len(ds))
        if positives:
            weights[positives] += 0.5/len(positives)
        # With a frozen prior, legacy data have no outcome labels and cannot
        # supervise any trainable head. Do not spend half the budget on them.
        source_weight = (float(ds.path == args.aligned_dataset) if args.freeze_prior
                         else 1.0/len(datasets))
        source_weights.append(weights*source_weight)
        print(f"source={ds.path} event_windows={len(positives)}", flush=True)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.cat(source_weights), args.iterations * args.batch_size, replacement=True
    )
    loader = DataLoader(source, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    iterator = iter(loader)
    validation = {
        k: torch.stack([item[k] for item in validations]).to(args.device)
        for k in validations[0]
    }
    if args.actor_only:
        validation = action_change_weights(validation)
    selection_metric = "actor" if args.actor_only else "total"
    best = float("inf")
    stopper = EarlyStopping(args.patience, args.min_delta, args.min_steps)
    metadata = {
        "seed": args.seed, "episode_splits": splits,
        "outcome_supervised": outcome_supervised,
        "sequence_length": args.sequence_length,
        "init": args.init, "closed_loop_verified": False,
        "early_stopping": {"val_every": args.val_every, "patience": args.patience,
                           "min_delta": args.min_delta, "min_steps": args.min_steps},
        "event_balanced_sampling": True, "prior_frozen": args.freeze_prior or args.actor_only,
        "actor_only": args.actor_only, "selection_metric": selection_metric,
    }
    print(json.dumps(metadata, ensure_ascii=False))
    for step in range(1, args.iterations+1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {k: v.to(args.device) for k, v in batch.items()}
        if args.actor_only:
            batch = action_change_weights(batch)
        losses = model.compute_loss(batch)
        objective = losses[selection_metric]
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("nonfinite sequence loss")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        model.update_target()
        if step == 1 or step % args.val_every == 0 or step == args.iterations:
            with torch.no_grad():
                heldout = model.compute_loss(validation)
            values = {k: float(v) for k, v in heldout.items()}
            print(f"step={step} train={float(objective):.5f} objective={selection_metric} validation={values}", flush=True)
            stop = stopper.update(values[selection_metric], step)
            if values[selection_metric] < best:
                best = values[selection_metric]
                save_world_model(args.checkpoint, model, optimizer=optimizer,
                    step=step, metadata={**metadata, "validation": values})
            if stop:
                print(f"early_stop step={step} bad_checks={stopper.bad_checks} best={best:.6f}", flush=True)
                break
    for ds in datasets:
        ds.close()
    print(f"saved heldout-selected checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
