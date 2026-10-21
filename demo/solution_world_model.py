"""ATEC Task E inference adapter for a trained latent world model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .world_model import CEMConfig, CEMPlanner, load_world_model


DEMO_HOME_ACTION = (
    -0.00006664105, -0.5509314537, -0.0300006866, 0.00002227853,
    0.4400000572, -0.00006618880, 0.0, 0.0,
)


def home_is_stable(proprio, default_joint_pos, home_joint_pos):
    error = home_joint_pos-(proprio[:, :8]+default_joint_pos)
    position_ready = (torch.all(error[:, :6].abs() <= 0.01)
                      & torch.all(error[:, 6:].abs() <= 0.001))
    return bool(position_ready & torch.all(proprio[:, 8:16].abs() <= 0.05))


class AlgSolution:
    """Keep the official ``predicts`` interface while planning in latent space."""

    _QPOS_SLICE = slice(0, 8)
    _QVEL_SLICE = slice(8, 16)

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        default_path = (
            Path(__file__).resolve().parent.parent
            / "atec_robot_model"
            / "baseline"
            / "world_model"
            / "policy.pt"
        )
        checkpoint_path = Path(os.environ.get("ATEC_TASK_E_WORLD_MODEL", default_path)).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Task E world-model checkpoint not found: {checkpoint_path}. "
                "Run scripts/world_model/train_task_e.py first."
            )
        self.model, self.checkpoint = load_world_model(checkpoint_path, self.device)
        self.is_recurrent = self.checkpoint.get("format_version") == 2
        self.belief = None
        self.outcome_supervised = bool(
            self.checkpoint.get("metadata", {}).get("outcome_supervised", False)
        )
        self.control_mode = os.environ.get("ATEC_WM_CONTROL", "cem").strip().lower()
        if self.control_mode not in {"actor", "cem"}:
            raise ValueError(
                f"Unsupported ATEC_WM_CONTROL={self.control_mode!r}; expected 'actor' or 'cem'."
            )
        from .world_model_prior_planner import PriorCEMPlanner
        planner_class = PriorCEMPlanner if self.is_recurrent else CEMPlanner
        self.planner = planner_class(
            self.model,
            CEMConfig(
                horizon=int(os.environ.get("ATEC_WM_HORIZON", "6")),
                population=int(os.environ.get("ATEC_WM_POPULATION", "64")),
                elites=int(os.environ.get("ATEC_WM_ELITES", "8")),
                iterations=int(os.environ.get("ATEC_WM_ITERATIONS", "2")),
                discount=self.model.config.gamma,
            ),
        )
        self.default_joint_pos = torch.tensor(
            [[0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035]],
            dtype=torch.float32,
            device=self.device,
        )
        self.teleop_home_joint_pos = torch.tensor(
            [[-0.000033, 0.924525, -1.514983, 0.000011, 1.219900, -0.000033, 0.035, -0.035]],
            dtype=torch.float32,
            device=self.device,
        )
        # Median initial command of the existing 20 demonstrations. The fifth
        # joint's physical stop means re-encoding measured home qpos gives a
        # different previous-action input; preserve the actual expert command.
        self.teleop_home_action = torch.tensor(
            [DEMO_HOME_ACTION],
            dtype=torch.float32, device=self.device,
        )
        self.belief_burn_in = 4
        self.belief_warmup_step = 0
        self.home_steps = 0
        self.home_timeout = 300
        self.max_action_delta = torch.tensor(
            [[0.05]*6 + [0.02]*2], dtype=torch.float32, device=self.device
        )
        self.startup_zero_steps = 25
        self.startup_step = 0
        self.home_done = False
        self.last_action: torch.Tensor | None = None
        self.control_phase = "startup"

    def _policy_action(self, latent: torch.Tensor) -> torch.Tensor:
        if getattr(self, "is_recurrent", False) and self.control_mode == "cem":
            if not self.outcome_supervised:
                raise RuntimeError("CEM requires real outcome supervision; use ATEC_WM_CONTROL=actor for prior diagnosis")
        """Select either behavior-cloned actor control or model-based planning."""
        if self.control_mode == "actor":
            return self.model.actor(latent)
        return self.planner.plan(latent)

    def get_action_spec(self) -> dict[str, dict[str, Any]]:
        return {}

    def _home_action(self, proprio: torch.Tensor) -> tuple[torch.Tensor, bool]:
        joint_pos = proprio[:, self._QPOS_SLICE] + self.default_joint_pos
        reached = home_is_stable(proprio, self.default_joint_pos, self.teleop_home_joint_pos)
        action = self.teleop_home_action.expand_as(joint_pos)
        return action, reached

    def predicts(self, obs, current_score):
        del current_score
        if not isinstance(obs, dict) or "proprio" not in obs or "image" not in obs:
            raise ValueError("World model expects obs['proprio'] and obs['image'].")
        if "video_rgb" not in obs["image"]:
            raise ValueError("World model expects obs['image']['video_rgb'].")

        proprio = obs["proprio"].to(self.device).float()
        if proprio.shape[-1] != self.model.config.proprio_dim:
            raise ValueError(
                f"Checkpoint expects {self.model.config.proprio_dim} proprio values, "
                f"received {proprio.shape[-1]}."
            )
        if self.startup_step < self.startup_zero_steps:
            self.control_phase = "startup"
            self.startup_step += 1
            action = torch.zeros((proprio.shape[0], self.model.config.action_dim), device=self.device)
            return {"action": action.cpu().tolist(), "giveup": False}

        if not self.home_done:
            self.control_phase = "home"
            self.home_steps += 1
            action, self.home_done = self._home_action(proprio)
            if not self.home_done and self.home_steps >= self.home_timeout:
                self.control_phase = "home_timeout"
                return {"action": action.cpu().tolist(), "giveup": True}
            return {"action": action.cpu().tolist(), "giveup": False}

        image = obs["image"]["video_rgb"].to(self.device)
        if self.is_recurrent and self.belief_warmup_step < self.belief_burn_in:
            self.control_phase = "belief_warmup"
            with torch.inference_mode():
                self.belief = self.model.observe(proprio, image, self.belief)
            self.belief_warmup_step += 1
            action = self.teleop_home_action.expand(proprio.shape[0], -1)
            return {"action": action.cpu().tolist(), "giveup": False}

        self.control_phase = self.control_mode
        with torch.inference_mode():
            if self.is_recurrent:
                self.belief = self.model.observe(proprio, image, self.belief)
                self.planner.previous_action = proprio[:, 16:24]
                latent = self.belief
            else:
                latent = self.model.encode(proprio, image)
            action = self._policy_action(latent)
            if self.last_action is not None:
                action = 0.8 * action + 0.2 * self.last_action
            previous = proprio[:, 16:24]
            action = previous + (action-previous).clamp(
                -self.max_action_delta, self.max_action_delta
            )
            action = action.clamp(-self.model.config.action_limit, self.model.config.action_limit)
            self.last_action = action
        return {"action": action.cpu().tolist(), "giveup": False}

    def get_control_diagnostics(self, proprio):
        """Debug-only legal-input state; never supplies extra inputs to predicts."""
        q = proprio.to(self.device).float()[:, :8] + self.default_joint_pos
        error = self.teleop_home_joint_pos-q
        return {"phase": self.control_phase, "home_done": self.home_done,
                "home_error": error[0].detach().cpu().tolist(),
                "home_max_error": float(error.abs().max()),
                "home_steps": self.home_steps,
                "belief_warmup_steps": self.belief_warmup_step,
                "belief_norm": float(self.belief.norm()) if self.belief is not None else None}

    def reset(self):
        self.belief = None
        self.last_action = None
        self.startup_step = 0
        self.home_done = False
        self.control_phase = "startup"
        self.home_steps = 0
        self.belief_warmup_step = 0
        if hasattr(self.planner, "reset"):
            self.planner.reset()
