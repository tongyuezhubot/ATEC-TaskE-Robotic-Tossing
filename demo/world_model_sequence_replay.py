"""Episode-aware sequence sampling layered on the existing image replay."""
from __future__ import annotations

import torch
from .world_model import ReplayBuffer


class SequenceReplay(ReplayBuffer):
    def __init__(self, capacity, config):
        super().__init__(capacity, config)
        self.serial = torch.full((capacity,), -1, dtype=torch.long)
        self.episode = torch.full((capacity,), -1, dtype=torch.long)
        self.expert = torch.zeros(capacity, dtype=torch.bool)
        self.counter = 0
        self.episode_counter = 0
        self.terminated = torch.zeros(capacity, dtype=torch.bool)
        self.valid = torch.zeros(capacity, dtype=torch.bool)
        self.events = torch.zeros(capacity, 7)
        self.events_valid = torch.zeros(capacity, dtype=torch.bool)

    def add(self, proprio, image, action, reward, next_proprio, next_image, done,
            *, terminated=None, next_obs_valid=None, events=None):
        if proprio.shape[0] != 1:
            raise ValueError("Online sequence replay currently requires one env")
        index = self.position
        super().add(proprio, image, action, reward, next_proprio, next_image, done)
        self.serial[index] = self.counter
        self.episode[index] = self.episode_counter
        self.terminated[index] = bool(done[0]) if terminated is None else bool(terminated[0])
        self.valid[index] = not bool(done[0]) if next_obs_valid is None else bool(next_obs_valid[0])
        self.events_valid[index] = events is not None
        if events is not None:
            self.events[index] = torch.as_tensor(events).detach().cpu().reshape(7)
        self.counter += 1
        self.episode_counter += int(bool(done[0]))

    def sample_sequences(self, batch_size, length, device):
        order = self.serial.argsort()
        order = order[self.serial[order] >= 0]
        if len(order) < length:
            raise ValueError("Insufficient continuous replay")
        windows = order.unfold(0, length, 1)
        valid = (self.episode[windows[:, 0]] == self.episode[windows[:, -1]])
        valid &= (self.serial[windows[:, -1]] - self.serial[windows[:, 0]] == length-1)
        windows = windows[valid]
        if len(windows) == 0:
            raise ValueError("No continuous episode windows")
        ids = windows[torch.randint(len(windows), (batch_size,))]
        batch = {name: getattr(self, name)[ids].to(device) for name in (
            "proprio", "image", "action", "reward", "next_proprio", "next_image", "done"
        )}
        batch["is_first"] = torch.zeros((batch_size, length), dtype=torch.bool, device=device)
        batch["is_first"][:, 0] = True
        batch["next_obs_valid"] = self.valid[ids].to(device)
        batch["terminated"] = self.terminated[ids].to(device)
        batch["reward_valid"] = torch.ones_like(batch["done"], dtype=torch.bool)
        batch["expert"] = torch.zeros_like(batch["done"], dtype=torch.bool)
        batch["events"] = self.events[ids].to(device)
        batch["events_valid"] = self.events_valid[ids].to(device)
        return batch
