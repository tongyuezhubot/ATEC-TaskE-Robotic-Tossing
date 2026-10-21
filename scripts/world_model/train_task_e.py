"""Collect Task E transitions and train the small latent world model online.

The 1000-step command documented in the project is a plumbing smoke test.  A
checkpoint from random sparse-reward data is not expected to solve Task E.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="Train the ATEC Task E latent world model.")
parser.add_argument("--task", default="ATEC-TaskE-Piper")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--total_steps", type=int, default=1000)
parser.add_argument("--random_steps", type=int, default=1000)
parser.add_argument("--buffer_size", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_starts", type=int, default=64)
parser.add_argument("--train_every", type=int, default=1)
parser.add_argument("--updates_per_step", type=int, default=1)
parser.add_argument("--save_every", type=int, default=500)
parser.add_argument("--log_every", type=int, default=25)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--image_size", type=int, default=96)
parser.add_argument("--latent_dim", type=int, default=128)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument(
    "--checkpoint",
    default=str(REPO_ROOT / "atec_robot_model" / "baseline" / "world_model" / "policy.pt"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Task E always contains cameras. This flag selects Isaac Lab's rendering
# experience even when the command is headless.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from demo.world_model import (  # noqa: E402
    CEMConfig,
    CEMPlanner,
    ReplayBuffer,
    WorldModel,
    WorldModelConfig,
    save_world_model,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _validate_args() -> None:
    positive = {
        "total_steps": args_cli.total_steps,
        "buffer_size": args_cli.buffer_size,
        "batch_size": args_cli.batch_size,
        "train_every": args_cli.train_every,
        "updates_per_step": args_cli.updates_per_step,
        "image_size": args_cli.image_size,
        "latent_dim": args_cli.latent_dim,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}")
    if args_cli.buffer_size < args_cli.batch_size:
        raise ValueError("buffer_size must be at least batch_size.")


def _image(obs: dict[str, object]) -> torch.Tensor:
    try:
        return obs["image"]["video_rgb"]
    except (KeyError, TypeError) as exc:
        raise KeyError("Expected Task E observation image.video_rgb.") from exc


def _save(model, optimizer, step: int, last_losses: dict[str, float]) -> None:
    save_world_model(
        args_cli.checkpoint,
        model,
        optimizer=optimizer,
        step=step,
        metadata={
            "task": args_cli.task,
            "seed": args_cli.seed,
            "smoke_only": args_cli.total_steps <= 1000,
            "last_losses": last_losses,
        },
    )
    size_mb = Path(args_cli.checkpoint).stat().st_size / (1024 * 1024)
    print(f"[INFO] Saved world model at step {step}: {args_cli.checkpoint} ({size_mb:.1f} MiB)")


def main() -> None:
    _validate_args()
    random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    device = torch.device(args_cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but PyTorch cannot access CUDA.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset(seed=args_cli.seed)
    proprio_dim = int(obs["proprio"].shape[-1])
    action_dim = int(env.action_space.shape[-1])
    if proprio_dim != 24 or action_dim != 8:
        raise RuntimeError(
            f"Unexpected Task E contract: proprio={proprio_dim}, action={action_dim}; expected 24 and 8."
        )

    config = WorldModelConfig(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        image_size=args_cli.image_size,
        latent_dim=args_cli.latent_dim,
    )
    model = WorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args_cli.learning_rate, weight_decay=1.0e-5)
    replay = ReplayBuffer(args_cli.buffer_size, config)
    planner = CEMPlanner(
        model,
        CEMConfig(horizon=5, population=128, elites=16, iterations=3, discount=config.gamma),
    )
    learning_starts = max(args_cli.learning_starts, args_cli.batch_size)
    last_losses: dict[str, float] = {}
    updates = 0

    print(
        "[INFO] World-model smoke configuration: "
        f"device={device}, proprio={proprio_dim}, action={action_dim}, "
        f"image={config.image_size}, buffer={args_cli.buffer_size}"
    )
    try:
        for step in range(1, args_cli.total_steps + 1):
            if step <= args_cli.random_steps or len(replay) < learning_starts:
                action = torch.empty(
                    (args_cli.num_envs, action_dim), device=device, dtype=torch.float32
                ).uniform_(-config.action_limit, config.action_limit)
            else:
                model.eval()
                with torch.inference_mode():
                    latent = model.encode(obs["proprio"].to(device), _image(obs).to(device))
                    action = planner.plan(latent)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            replay.add(
                obs["proprio"],
                _image(obs),
                action,
                reward,
                next_obs["proprio"],
                _image(next_obs),
                done,
            )
            obs = next_obs

            if len(replay) >= learning_starts and step % args_cli.train_every == 0:
                model.train()
                for _ in range(args_cli.updates_per_step):
                    batch = replay.sample(args_cli.batch_size, device)
                    losses = model.compute_loss(batch)
                    total_loss = losses["total"]
                    if not torch.isfinite(total_loss):
                        raise FloatingPointError(f"Non-finite loss at step {step}: {losses}")
                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
                    updates += 1
                    last_losses = {name: float(value.detach().cpu()) for name, value in losses.items()}

            if step == 1 or step % args_cli.log_every == 0:
                memory = "cpu"
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
                    reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
                    memory = f"cuda_alloc={allocated:.0f}MiB,cuda_reserved={reserved:.0f}MiB"
                loss_text = f"{last_losses.get('total', float('nan')):.5f}"
                print(
                    f"[WM] step={step}/{args_cli.total_steps} replay={len(replay)} "
                    f"updates={updates} loss={loss_text} reward={float(reward.mean()):.4f} {memory}"
                )

            if args_cli.save_every > 0 and step % args_cli.save_every == 0:
                _save(model, optimizer, step, last_losses)

        if args_cli.save_every <= 0 or args_cli.total_steps % args_cli.save_every != 0:
            _save(model, optimizer, args_cli.total_steps, last_losses)
        print("[PASS] World-model collection, training, finite-loss, CUDA, and checkpoint smoke test completed.")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
