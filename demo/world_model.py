"""Small latent world model and CEM planner for ATEC Task E.

The module has no Isaac Sim dependency so its tensor contracts, replay buffer,
checkpoint format, and planner can be tested on CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class WorldModelConfig:
    proprio_dim: int = 24
    action_dim: int = 8
    action_limit: float = 3.0
    image_size: int = 96
    image_feature_dim: int = 128
    latent_dim: int = 128
    hidden_dim: int = 256
    gamma: float = 0.99

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "WorldModelConfig":
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 8
    population: int = 256
    elites: int = 32
    iterations: int = 4
    min_std: float = 0.05
    momentum: float = 0.1
    discount: float = 0.99

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.population < 2 or self.iterations < 1:
            raise ValueError("CEM horizon, population, and iterations must be positive.")
        if not 1 <= self.elites <= self.population:
            raise ValueError("CEM elites must be in [1, population].")


def _image_to_bchw(image: torch.Tensor) -> torch.Tensor:
    """Convert a single image or batch from HWC/CHW to BCHW."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected a 3-D or 4-D RGB tensor, got {tuple(image.shape)}")
    if image.shape[1] in (3, 4):
        image = image[:, :3]
    elif image.shape[-1] in (3, 4):
        image = image[..., :3].permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Could not identify RGB channels in shape {tuple(image.shape)}")
    return image.contiguous()


def resize_rgb_uint8(image: torch.Tensor, image_size: int) -> torch.Tensor:
    """Return a CPU uint8 BCHW image batch at the replay resolution."""
    image = _image_to_bchw(image).detach()
    if image.dtype == torch.uint8:
        image_float = image.float()
    else:
        image_float = image.float()
        # Isaac returns uint8 when normalize=False; accepting [0, 1] floats
        # makes the core module convenient for tests and other cameras.
        if image_float.numel() and float(image_float.detach().max().cpu()) <= 1.0:
            image_float = image_float * 255.0
    if image_float.shape[-2:] != (image_size, image_size):
        image_float = F.interpolate(
            image_float,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return image_float.round().clamp_(0, 255).to(dtype=torch.uint8, device="cpu")


class ImageEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = _image_to_bchw(image)
        image = image.float()
        if image.dtype == torch.uint8:  # pragma: no cover - kept for clarity
            image = image / 255.0
        else:
            # Replay images are uint8. Float tensors supplied directly by callers
            # are treated as [0, 255] if their range is larger than one.
            scale = 255.0 if image.detach().amax() > 1.0 else 1.0
            image = image / scale
        return self.network(image)


class ScaledTanh(nn.Module):
    def __init__(self, limit: float):
        super().__init__()
        self.limit = float(limit)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(value) * self.limit


class WorldModel(nn.Module):
    """RGB + proprio encoder with deterministic latent dynamics and heads."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        cfg = config
        self.image_encoder = ImageEncoder(cfg.image_feature_dim)
        self.observation_encoder = nn.Sequential(
            nn.Linear(cfg.image_feature_dim + cfg.proprio_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
            ScaledTanh(cfg.action_limit),
        )
        self.proprio_decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.proprio_dim),
        )
        self.inverse_head = nn.Sequential(
            nn.Linear(2 * cfg.latent_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
            ScaledTanh(cfg.action_limit),
        )

    def encode(self, proprio: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or proprio.shape[-1] != self.config.proprio_dim:
            raise ValueError(
                f"Expected proprio shape (B, {self.config.proprio_dim}), got {tuple(proprio.shape)}"
            )
        image = _image_to_bchw(image)
        if image.shape[-2:] != (self.config.image_size, self.config.image_size):
            image = F.interpolate(
                image.float(),
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        image_features = self.image_encoder(image)
        if image_features.shape[0] != proprio.shape[0]:
            raise ValueError("Image and proprio batch sizes differ.")
        return self.observation_encoder(torch.cat((image_features, proprio.float()), dim=-1))

    def next_latent(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.dynamics(torch.cat((latent, action), dim=-1))
        return F.layer_norm(latent + delta, (self.config.latent_dim,))

    def predict_reward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.reward_head(torch.cat((latent, action), dim=-1)).squeeze(-1)

    def predict_value(self, latent: torch.Tensor) -> torch.Tensor:
        return self.value_head(latent).squeeze(-1)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        latent = self.encode(batch["proprio"], batch["image"])
        with torch.no_grad():
            next_target = self.encode(batch["next_proprio"], batch["next_image"])

        action = batch["action"].float()
        reward = batch["reward"].float().view(-1)
        done = batch["done"].float().view(-1)
        predicted_next = self.next_latent(latent, action)
        predicted_reward = self.predict_reward(latent, action)
        predicted_value = self.predict_value(latent)
        with torch.no_grad():
            value_target = reward + self.config.gamma * (1.0 - done) * self.predict_value(next_target)

        losses = {
            "dynamics": F.smooth_l1_loss(predicted_next, next_target),
            "reward": ((predicted_reward - reward).square() *
                batch.get("reward_valid", torch.ones_like(reward)).float()
            ).sum() / batch.get("reward_valid", torch.ones_like(reward)).sum().clamp_min(1),
            "value": ((predicted_value - value_target).square() *
                batch.get("reward_valid", torch.ones_like(reward)).float()
            ).sum() / batch.get("reward_valid", torch.ones_like(reward)).sum().clamp_min(1),
            "proprio": F.mse_loss(self.proprio_decoder(latent), batch["proprio"].float()),
            "inverse": F.mse_loss(
                self.inverse_head(torch.cat((latent, next_target), dim=-1)), action
            ),
            "actor": F.mse_loss(self.actor(latent), action),
        }
        losses["total"] = (
            losses["dynamics"]
            + losses["reward"]
            + 0.5 * losses["value"]
            + 0.1 * losses["proprio"]
            + 0.1 * losses["inverse"]
            + 0.5 * losses["actor"]
        )
        return losses


class ReplayBuffer:
    """Fixed-size CPU replay storing downsampled uint8 RGB frames."""

    def __init__(self, capacity: int, config: WorldModelConfig):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive.")
        self.capacity = int(capacity)
        self.config = config
        size = config.image_size
        self.image = torch.empty((capacity, 3, size, size), dtype=torch.uint8)
        self.next_image = torch.empty_like(self.image)
        self.proprio = torch.empty((capacity, config.proprio_dim), dtype=torch.float32)
        self.next_proprio = torch.empty_like(self.proprio)
        self.action = torch.empty((capacity, config.action_dim), dtype=torch.float32)
        self.reward = torch.empty((capacity,), dtype=torch.float32)
        self.done = torch.empty((capacity,), dtype=torch.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        proprio: torch.Tensor,
        image: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_proprio: torch.Tensor,
        next_image: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        proprio = proprio.detach().float().view(-1, self.config.proprio_dim).cpu()
        next_proprio = next_proprio.detach().float().view(-1, self.config.proprio_dim).cpu()
        action = action.detach().float().view(-1, self.config.action_dim).cpu()
        reward = reward.detach().float().view(-1).cpu()
        done = done.detach().float().view(-1).cpu()
        image = resize_rgb_uint8(image, self.config.image_size)
        next_image = resize_rgb_uint8(next_image, self.config.image_size)
        batch_size = proprio.shape[0]
        if not all(tensor.shape[0] == batch_size for tensor in (next_proprio, action, reward, done, image, next_image)):
            raise ValueError("Replay transition tensors have different batch sizes.")
        if batch_size > self.capacity:
            start = batch_size - self.capacity
            proprio, next_proprio, action = proprio[start:], next_proprio[start:], action[start:]
            reward, done, image, next_image = reward[start:], done[start:], image[start:], next_image[start:]
            batch_size = self.capacity

        indices = (torch.arange(batch_size) + self.position) % self.capacity
        self.proprio[indices] = proprio
        self.image[indices] = image
        self.action[indices] = action
        self.reward[indices] = reward
        self.next_proprio[indices] = next_proprio
        self.next_image[indices] = next_image
        self.done[indices] = done
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def sample(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        if batch_size < 1 or batch_size > self.size:
            raise ValueError(f"Cannot sample batch {batch_size} from replay size {self.size}.")
        indices = torch.randint(0, self.size, (batch_size,))
        return {
            "proprio": self.proprio[indices].to(device),
            "image": self.image[indices].to(device),
            "action": self.action[indices].to(device),
            "reward": self.reward[indices].to(device),
            "next_proprio": self.next_proprio[indices].to(device),
            "next_image": self.next_image[indices].to(device),
            "done": self.done[indices].to(device),
        }


class CEMPlanner:
    def __init__(self, model: WorldModel, config: CEMConfig | None = None):
        self.model = model
        self.config = config or CEMConfig(discount=model.config.gamma)

    @torch.no_grad()
    def plan(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        return torch.stack([self._plan_one(item) for item in latent], dim=0)

    def _plan_one(self, latent: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        device = latent.device
        action_dim = self.model.config.action_dim
        mean = torch.zeros((cfg.horizon, action_dim), device=device)
        action_limit = self.model.config.action_limit
        mean[0] = self.model.actor(latent.unsqueeze(0))[0]
        std = torch.full_like(mean, action_limit)

        for _ in range(cfg.iterations):
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                (cfg.population, cfg.horizon, action_dim), device=device
            )
            actions.clamp_(-action_limit, action_limit)
            rollout_latent = latent.unsqueeze(0).expand(cfg.population, -1)
            returns = torch.zeros(cfg.population, device=device)
            discount = 1.0
            for horizon_step in range(cfg.horizon):
                current_action = actions[:, horizon_step]
                returns.add_(discount * self.model.predict_reward(rollout_latent, current_action))
                rollout_latent = self.model.next_latent(rollout_latent, current_action)
                discount *= cfg.discount
            returns.add_(discount * self.model.predict_value(rollout_latent))
            elite_indices = returns.topk(cfg.elites, sorted=False).indices
            elite_actions = actions[elite_indices]
            elite_mean = elite_actions.mean(dim=0)
            elite_std = elite_actions.std(dim=0, unbiased=False).clamp_min(cfg.min_std)
            mean = cfg.momentum * mean + (1.0 - cfg.momentum) * elite_mean
            std = cfg.momentum * std + (1.0 - cfg.momentum) * elite_std
        return mean[0].clamp(-action_limit, action_limit)


def save_world_model(
    path: str | Path,
    model: WorldModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": getattr(model, "architecture_version", 1),
        "config": asdict(model.config),
        "model": model.state_dict(),
        "step": int(step),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_world_model(
    path: str | Path,
    device: torch.device | str = "cpu",
    *,
    eval_mode: bool = True,
) -> tuple[WorldModel, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or "config" not in payload or "model" not in payload:
        raise ValueError(f"Invalid world-model checkpoint: {path}")
    config = WorldModelConfig.from_dict(payload["config"])
    if payload.get("format_version", 1) == 2:
        from .world_model_recurrent import RecurrentWorldModel
        model = RecurrentWorldModel(config).to(device)
    else:
        model = WorldModel(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    if eval_mode:
        model.eval()
    return model, payload
