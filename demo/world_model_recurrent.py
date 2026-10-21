"""Recurrent extension of the existing Task E encoder, dynamics and actor."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from .world_model import WorldModel, _image_to_bchw


def masked_mean(value, mask):
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


class RecurrentWorldModel(WorldModel):
    architecture_version = 2

    def __init__(self, config):
        super().__init__(config)
        cfg = config
        self.belief_cell = nn.GRUCell(cfg.latent_dim + cfg.action_dim, cfg.latent_dim)
        self.visual_predictor = nn.Linear(cfg.latent_dim, cfg.image_feature_dim)
        # A tiny spatial reconstruction anchor complements EMA and variance:
        # a constant visual encoder cannot reconstruct different observations.
        self.visual_anchor = nn.Linear(cfg.image_feature_dim, 3 * 8 * 8)
        self.event_head = nn.Linear(cfg.latent_dim + cfg.action_dim, 7)
        self.target_image_encoder = deepcopy(self.image_encoder).requires_grad_(False)
        self.target_observation_encoder = deepcopy(self.observation_encoder).requires_grad_(False)
        self.target_belief_cell = deepcopy(self.belief_cell).requires_grad_(False)
        self.target_value_head = deepcopy(self.value_head).requires_grad_(False)

    def _features(self, proprio, image, target=False):
        image = _image_to_bchw(image).to(proprio.device)
        if image.shape[-2:] != (self.config.image_size, self.config.image_size):
            image = F.interpolate(image.float(),
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear", align_corners=False, antialias=True)
            image = image.round().clamp(0, 255).to(torch.uint8)
        encoder = self.target_image_encoder if target else self.image_encoder
        fuse = self.target_observation_encoder if target else self.observation_encoder
        visual = encoder(image)
        observation = fuse(torch.cat((visual, proprio.float()), dim=-1))
        return observation, visual, image

    def observe(self, proprio, image, belief=None, is_first=None, target=False):
        if proprio.ndim != 2 or proprio.shape[-1] != 24:
            raise ValueError("Deployment requires official 24-D proprioception")
        if belief is None:
            belief = proprio.new_zeros((proprio.shape[0], self.config.latent_dim))
        if is_first is not None:
            belief = belief * (~is_first.bool()).float().unsqueeze(-1)
        observation, _, _ = self._features(proprio, image, target)
        cell = self.target_belief_cell if target else self.belief_cell
        # This is the actual previous environment action, including home and
        # expert warm-up actions; training and deployment use the same source.
        return cell(torch.cat((observation, proprio[:, 16:24]), dim=-1), belief)

    def encode(self, proprio, image):
        return self.observe(proprio, image)

    def compute_loss(self, batch):
        if batch["action"].ndim == 2:
            batch = {key: value.unsqueeze(1) for key, value in batch.items()}
        b, length, _ = batch["action"].shape
        belief = batch["proprio"].new_zeros((b, self.config.latent_dim))
        target_belief = torch.zeros_like(belief)
        zero = belief.sum() * 0
        losses = {key: zero for key in (
            "dynamics", "reward", "value", "proprio", "inverse", "actor",
            "visual", "anchor", "variance", "events",
        )}
        for t in range(length):
            first = batch.get("is_first")
            reset = first[:, t] if first is not None else (
                torch.ones(b, dtype=torch.bool, device=belief.device) if t == 0
                else batch["done"][:, t-1].bool()
            )
            proprio, image = batch["proprio"][:, t], batch["image"][:, t]
            action = batch["action"][:, t]
            observation, visual, rgb = self._features(proprio, image)
            belief = belief * (~reset.bool()).float().unsqueeze(-1)
            belief = self.belief_cell(
                torch.cat((observation, proprio[:, 16:24]), dim=-1), belief
            )
            predicted = self.next_latent(belief, action)
            with torch.no_grad():
                target_belief = self.observe(
                    proprio, image, target_belief, reset, target=True
                )
                target_next = self.observe(
                    batch["next_proprio"][:, t], batch["next_image"][:, t],
                    target_belief, target=True,
                )
                _, target_visual, _ = self._features(
                    batch["next_proprio"][:, t], batch["next_image"][:, t], target=True
                )
                terminal = batch.get("terminated", batch["done"])[:, t].bool()
                reward = batch["reward"][:, t]
                value_target = reward + self.config.gamma * (~terminal).float() * (
                    self.target_value_head(target_next).squeeze(-1)
                )

            next_valid = batch.get("next_obs_valid", torch.ones_like(batch["done"]))[:, t]
            reward_valid = batch.get("reward_valid", torch.ones_like(batch["done"]))[:, t]
            # A truncated transition with no terminal observation cannot bootstrap.
            value_valid = reward_valid.bool() & (terminal | next_valid.bool())
            expert = batch.get("expert", torch.ones_like(batch["done"]))[:, t]
            if "actor_weight" in batch:
                expert = expert.float() * batch["actor_weight"][:, t]
            losses["dynamics"] = losses["dynamics"] + masked_mean(
                (predicted - target_next).square().mean(-1), next_valid
            )
            losses["proprio"] = losses["proprio"] + masked_mean(
                F.smooth_l1_loss(self.proprio_decoder(predicted),
                    batch["next_proprio"][:, t], reduction="none").mean(-1), next_valid
            )
            losses["visual"] = losses["visual"] + masked_mean(
                1 - F.cosine_similarity(self.visual_predictor(predicted), target_visual, dim=-1),
                next_valid,
            )
            thumbnail = F.adaptive_avg_pool2d(rgb.float() / 255, (8, 8)).flatten(1)
            losses["anchor"] = losses["anchor"] + F.mse_loss(
                self.visual_anchor(visual), thumbnail
            )
            losses["variance"] = losses["variance"] + F.relu(
                0.1 - visual.std(dim=0, unbiased=False)
            ).mean()
            losses["reward"] = losses["reward"] + masked_mean(
                (self.predict_reward(belief, action) - reward).square(), reward_valid
            )
            losses["value"] = losses["value"] + masked_mean(
                (self.predict_value(belief) - value_target).square(), value_valid
            )
            losses["actor"] = losses["actor"] + masked_mean(
                F.smooth_l1_loss(self.actor(belief), action, reduction="none").mean(-1), expert
            )
            losses["inverse"] = losses["inverse"] + masked_mean(
                F.smooth_l1_loss(self.inverse_head(torch.cat((belief, target_next), -1)),
                    action, reduction="none").mean(-1), next_valid
            )
            if "events" in batch and bool(batch["events_valid"][:, t].any()):
                logits = self.event_head(torch.cat((belief, action), dim=-1))
                event_loss = F.binary_cross_entropy_with_logits(
                    logits, batch["events"][:, t].float(), reduction="none",
                    pos_weight=logits.new_full((7,), 5.0),
                ).mean(-1)
                losses["events"] = losses["events"] + masked_mean(
                    event_loss, batch["events_valid"][:, t]
                )

        losses = {key: value / length for key, value in losses.items()}
        losses["total"] = (
            losses["dynamics"] + losses["visual"] + losses["reward"]
            + 0.5 * losses["value"] + 0.2 * losses["proprio"]
            + 0.1 * losses["inverse"] + 0.5 * losses["actor"]
            + 0.2 * losses["anchor"] + 0.1 * losses["variance"] + losses["events"]
        )
        return losses

    @torch.no_grad()
    def update_target(self, tau=0.01):
        for online, target in (
            (self.image_encoder, self.target_image_encoder),
            (self.observation_encoder, self.target_observation_encoder),
            (self.belief_cell, self.target_belief_cell),
            (self.value_head, self.target_value_head),
        ):
            for source, dest in zip(online.parameters(), target.parameters()):
                dest.lerp_(source, tau)
