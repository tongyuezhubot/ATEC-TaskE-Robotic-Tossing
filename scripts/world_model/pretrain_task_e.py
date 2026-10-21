"""Pretrain the Task E world model from successful scripted demonstrations."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.world_model import (  # noqa: E402
    WorldModel,
    WorldModelConfig,
    load_world_model,
    save_world_model,
)
from demo.world_model_dataset import ExpertTransitionDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Collector trajectory.hdf5")
    parser.add_argument("--total_iters", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "atec_robot_model" / "baseline" / "world_model" / "policy.pt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_iters < 1 or args.batch_size < 1:
        raise ValueError("total_iters and batch_size must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    dataset = ExpertTransitionDataset(args.dataset, image_size=args.image_size, only_success=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(dataset) >= args.batch_size,
        num_workers=0,
    )
    if args.resume:
        model, payload = load_world_model(args.resume, device, eval_mode=False)
        if model.config.image_size != args.image_size:
            raise ValueError("Resume checkpoint image_size differs from --image_size.")
        print(f"[INFO] Resumed checkpoint step={payload.get('step', 0)} from {args.resume}")
    else:
        model = WorldModel(
            WorldModelConfig(image_size=args.image_size, latent_dim=args.latent_dim)
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5)
    iterator = iter(loader)
    last_losses: dict[str, float] = {}
    print(f"[INFO] Expert transitions={len(dataset)}, batches/epoch={len(loader)}, device={device}")
    try:
        for iteration in range(1, args.total_iters + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            model.train()
            losses = model.compute_loss(batch)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite expert loss at iteration {iteration}: {losses}")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            last_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}

            if iteration == 1 or iteration % args.log_every == 0:
                print(
                    f"[PRETRAIN] iter={iteration}/{args.total_iters} "
                    f"loss={last_losses['total']:.5f} actor={last_losses['actor']:.5f} "
                    f"dynamics={last_losses['dynamics']:.5f} reward={last_losses['reward']:.5f}"
                )
            if args.save_every > 0 and iteration % args.save_every == 0:
                save_world_model(
                    args.checkpoint,
                    model,
                    optimizer=optimizer,
                    step=iteration,
                    metadata={
                        "task": "ATEC-TaskE-Piper",
                        "source": "successful_scripted_demonstrations",
                        "dataset": str(Path(args.dataset).resolve()),
                        "last_losses": last_losses,
                    },
                )
                print(f"[INFO] Saved expert-pretrained checkpoint: {args.checkpoint}")
        if args.save_every <= 0 or args.total_iters % args.save_every != 0:
            save_world_model(
                args.checkpoint,
                model,
                optimizer=optimizer,
                step=args.total_iters,
                metadata={
                    "task": "ATEC-TaskE-Piper",
                    "source": "successful_scripted_demonstrations",
                    "dataset": str(Path(args.dataset).resolve()),
                    "last_losses": last_losses,
                },
            )
        print("[PASS] Expert world-model pretraining completed with finite losses.")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
