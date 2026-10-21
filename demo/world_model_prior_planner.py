"""Conservative short-horizon planning around the demonstrated actor."""
from __future__ import annotations

import torch
from .world_model import CEMPlanner, CEMConfig


class PriorCEMPlanner(CEMPlanner):
    def __init__(self, model, config=None):
        super().__init__(model, config or CEMConfig(
            horizon=6, population=64, elites=8, iterations=2,
            discount=model.config.gamma,
        ))
        self.previous_action = None

    def reset(self):
        self.previous_action = None

    def _plan_one(self, latent):
        cfg = self.config
        limit = self.model.config.action_limit
        mean = []
        state = latent.unsqueeze(0)
        for _ in range(cfg.horizon):
            action = self.model.actor(state)
            mean.append(action[0])
            state = self.model.next_latent(state, action)
        mean = torch.stack(mean)
        std = torch.full_like(mean, 0.15 * limit)
        for _ in range(cfg.iterations):
            actions = (mean[None] + std[None] * torch.randn(
                cfg.population, cfg.horizon, self.model.config.action_dim,
                device=latent.device,
            )).clamp(-limit, limit)
            # Enforce bounded changes through each rollout, including the first
            # command relative to the action actually applied to the environment.
            if self.previous_action is not None:
                actions[:, 0] = self.previous_action + (
                    actions[:, 0] - self.previous_action
                ).clamp(-0.25, 0.25)
            for t in range(1, cfg.horizon):
                actions[:, t] = actions[:, t-1] + (
                    actions[:, t] - actions[:, t-1]
                ).clamp(-0.25, 0.25)
            state = latent[None].expand(cfg.population, -1)
            returns = torch.zeros(cfg.population, device=latent.device)
            discount = 1
            for t in range(cfg.horizon):
                action = actions[:, t]
                penalty = 0.05 * ((action-self.model.actor(state))/limit).square().mean(-1)
                returns += discount * (self.model.predict_reward(state, action) - penalty)
                state = self.model.next_latent(state, action)
                discount *= cfg.discount
            returns += discount * self.model.predict_value(state)
            elite = actions[returns.topk(cfg.elites).indices]
            mean = cfg.momentum * mean + (1-cfg.momentum) * elite.mean(0)
            std = elite.std(0, unbiased=False).clamp_min(cfg.min_std)
        return mean[0].clamp(-limit, limit)
