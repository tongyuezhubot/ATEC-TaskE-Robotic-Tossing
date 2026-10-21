"""Single-env online update: <=5000 steps, smooth perturbations, expert mixing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--resume", required=True)
p.add_argument("--expert_dataset", required=True)
p.add_argument("--output_dir", required=True)
p.add_argument("--total_steps", type=int, default=1000)
p.add_argument("--batch_size", type=int, default=4)
p.add_argument("--sequence_length", type=int, default=12)
p.add_argument("--train_every", type=int, default=10)
p.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if not 1 <= args.total_steps <= 5000:
    p.error("First online budget must be 1..5000 steps")
args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app

import h5py
import numpy as np
import torch
import gymnasium as gym
import atec_rl_lab.tasks
from isaaclab_tasks.utils import parse_env_cfg
from demo.world_model import load_world_model, save_world_model
from demo.world_model_sequence import ExpertSequenceDataset, episode_split
from demo.world_model_sequence_replay import SequenceReplay
from demo.world_model_prior_planner import PriorCEMPlanner
from scripts.world_model.aligned_transitions import aligned_step, save_aligned_group
from demo.world_model_dataset import DEFAULT_JOINT_POS
from demo.solution_world_model import DEMO_HOME_ACTION, home_is_stable


def main():
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    model, payload = load_world_model(args.resume, args.device, eval_mode=False)
    if payload.get("format_version") != 2:
        raise ValueError("Use a recurrent checkpoint")
    outcome_ready = payload.get("metadata", {}).get("outcome_supervised", False)
    if payload.get("metadata", {}).get("prior_frozen", False):
        from scripts.world_model.pretrain_recurrent import freeze_action_prior
        freeze_action_prior(model)
        print("keeping checkpoint action prior frozen during online head updates", flush=True)
    train_keys, _ = episode_split(args.expert_dataset, args.seed)
    # Keep precisely the original held-out episodes out of online updates.
    splits = payload.get("metadata", {}).get("episode_splits", {})
    if args.expert_dataset in splits:
        train_keys = splits[args.expert_dataset]["train"]
    expert = ExpertSequenceDataset(args.expert_dataset,
        sequence_length=args.sequence_length, keys=train_keys)
    loader = torch.utils.data.DataLoader(expert, batch_size=args.batch_size, shuffle=True)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=1e-4)
    replay = SequenceReplay(min(5000, args.total_steps), model.config)
    planner = PriorCEMPlanner(model)
    config = parse_env_cfg("ATEC-TaskE-Piper", device=args.device, num_envs=1)
    config.seed = args.seed
    env = gym.make("ATEC-TaskE-Piper", cfg=config)
    obs, _ = env.reset(seed=args.seed)
    records, episode, steps, updates = [], 0, 0, 0
    belief = None
    noise = torch.zeros(1, 8, device=args.device)
    events_total = np.zeros(7, np.int64)
    startup_step, home_done, home_steps = 0, False, 0
    belief_warmup_step = 0
    home_target = torch.tensor([[-0.000033, 0.924525, -1.514983, 0.000011,
        1.219900, -0.000033, 0.035, -0.035]], device=args.device)
    home_command = torch.tensor([DEMO_HOME_ACTION], device=args.device)
    action_rate_limit = torch.tensor([[0.05]*6+[0.02]*2], device=args.device)

    def flush(file):
        nonlocal records, episode
        if not records:
            return
        g = file.create_group(f"traj_{episode}")
        save_aligned_group(g, records)
        proprio = np.stack([r["policy_input"]["proprio"] for r in records])
        g.create_dataset("obs", data=proprio[:, :8] + DEFAULT_JOINT_POS)
        g.create_dataset("qvel", data=proprio[:, 8:16])
        g.create_dataset("actions", data=np.stack([r["action"] for r in records]))
        g.create_group("images")["rgb"] = g["policy_input/rgb"]
        g.attrs["actor_supervision"] = False
        g.attrs["success"] = bool(records[-1]["training_labels"]["basket_success"])
        g.attrs["budget_boundary"] = not (records[-1]["terminated"] or records[-1]["truncated"])
        episode += 1
        records = []
        file.flush()

    try:
        with h5py.File(output / "trajectory.hdf5", "x") as file:
            for step in range(1, args.total_steps+1):
                if not app.is_running():
                    break
                with torch.no_grad():
                    proprio = obs["proprio"].to(args.device)
                    image = obs["image"]["video_rgb"].to(args.device)
                    policy_ready = False
                    if startup_step < 25:
                        startup_step += 1
                        center = torch.zeros(1, 8, device=args.device)
                    elif not home_done:
                        home_steps += 1
                        default = torch.as_tensor(DEFAULT_JOINT_POS, device=args.device)
                        center = home_command
                        home_done = home_is_stable(proprio, default, home_target)
                        if not home_done and home_steps >= 300:
                            raise RuntimeError("Home initialization did not converge within 300 steps")
                    elif belief_warmup_step < 4:
                        belief = model.observe(proprio, image, belief)
                        belief_warmup_step += 1
                        center = home_command
                    else:
                        policy_ready = True
                        belief = model.observe(proprio, image, belief)
                        planner.previous_action = proprio[:, 16:24]
                        center = planner.plan(belief) if outcome_ready else model.actor(belief)
                    noise = (0.9*noise + 0.1*torch.randn_like(noise)*0.08
                        if policy_ready else torch.zeros_like(noise))
                    action = (center + noise.clamp(-0.12, 0.12)).clamp(
                        -model.config.action_limit, model.config.action_limit)
                    # One step is 0.02 simulated seconds. Limit joint target
                    # changes as well as perturbation amplitude.
                    action = proprio[:, 16:24] + (action-proprio[:, 16:24]).clamp(
                        -action_rate_limit, action_rate_limit)
                result, record = aligned_step(env, action, model.config.image_size)
                next_obs, reward, terminated, truncated, info = result
                done = terminated | truncated
                labels = record["training_labels"]
                step_events = np.concatenate((labels["grasped_event"],
                    labels["in_basket_event"], [labels["basket_success"]])).astype(np.float32)
                next_p = torch.tensor(record["next_policy_input"]["proprio"])[None]
                next_i = torch.tensor(record["next_policy_input"]["rgb"])[None]
                replay.add(obs["proprio"], obs["image"]["video_rgb"], action, reward,
                    next_p, next_i, done, terminated=terminated,
                    next_obs_valid=torch.tensor([record["next_obs_valid"]]),
                    events=step_events)
                records.append(record)
                labels = record["training_labels"]
                events_total += np.concatenate((labels["grasped_event"], labels["in_basket_event"],
                    [labels["basket_success"]])).astype(np.int64)
                obs = next_obs
                steps = step
                if bool(done.any()):
                    flush(file)
                    belief = None
                    startup_step, home_done, home_steps = 0, False, 0
                    belief_warmup_step = 0
                    noise.zero_()
                    planner.reset()
                if len(replay) >= args.sequence_length and step % args.train_every == 0:
                    try:
                        online = replay.sample_sequences(args.batch_size,args.sequence_length,args.device)
                    except ValueError:
                        continue
                    try:
                        demonstrations = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        demonstrations = next(iterator)
                    demonstrations = {k:v.to(args.device) for k,v in demonstrations.items()}
                    losses = model.compute_loss(online)
                    prior_losses = model.compute_loss(demonstrations)
                    loss = 0.5*losses["total"] + 0.5*prior_losses["total"]
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite online loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    optimizer.step()
                    model.update_target()
                    updates += 1
                if step % 100 == 0:
                    print(f"step={step} updates={updates} events={events_total.tolist()}", flush=True)
            flush(file)
        save_world_model(output / "policy.pt", model, step=steps,
            optimizer=optimizer, metadata={
                **payload.get("metadata", {}), "online_steps": steps,
                "online_updates": updates, "seed": args.seed,
                "event_counts": events_total.tolist(),
                # True only after balanced offline labeled supervision. Sparse
                # zero-reward online updates alone do not establish planner quality.
                "outcome_supervised": bool(outcome_ready),
                "closed_loop_verified": False,
            })
    finally:
        expert.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
