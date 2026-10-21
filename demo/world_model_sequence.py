"""Lazy contiguous episode windows, with explicit masks for legacy unknowns."""
from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .world_model import resize_rgb_uint8
from .world_model_dataset import DEFAULT_JOINT_POS


class ExpertSequenceDataset(Dataset):
    def __init__(self, path, image_size=96, sequence_length=16, keys=None, burn_in=4):
        self.path = str(path)
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.burn_in = burn_in
        self._file = None
        self.windows = []
        with h5py.File(self.path) as f:
            self.keys = list(keys) if keys is not None else sorted(f.keys())
            for key in self.keys:
                g = f[key]
                aligned = "policy_input" in g
                if aligned and g["training_labels"].attrs.get("basket_success_source") not in (
                    "current_term", "recomputed_object_pos"
                ):
                    raise ValueError(f"{key}: unverified basket_success labels; repair stale-cache data first")
                if g.attrs.get("requires_actor_valid_mask", False) and "training_labels/actor_valid" not in g:
                    raise ValueError(f"{key}: recovery data require per-step actor_valid")
                n = len(g["actions"]) if aligned else len(g["actions"]) - 1
                first = 0 if aligned else 1
                # Strict lengths prevent silent truncation of malformed data.
                for field in ("obs", "qvel", "actions", "images/rgb"):
                    if len(g[field]) != len(g["actions"]):
                        raise ValueError(f"{key}: misaligned {field}")
                for start in range(first, n - sequence_length + 1):
                    self.windows.append((key, start, aligned))
        if not self.windows:
            raise ValueError("No sufficiently long episode windows")

    def __len__(self):
        return len(self.windows)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getitem__(self, index):
        if self._file is None:
            self._file = h5py.File(self.path)
        key, start, aligned = self.windows[index]
        g = self._file[key]
        sl = slice(start, start + self.sequence_length)
        if aligned:
            p = g["policy_input/proprio"][sl]
            np_ = g["next_policy_input/proprio"][sl]
            image = g["policy_input/rgb"][sl]
            next_image = g["next_policy_input/rgb"][sl]
            reward = g["reward"][sl]
            terminated = g["terminated"][sl]
            truncated = g["truncated"][sl]
            valid = g["next_obs_valid"][sl]
            reward_valid = np.ones(self.sequence_length, dtype=bool)
            labels = g["training_labels"]
            events = np.concatenate((
                labels["grasped_event"][sl], labels["in_basket_event"][sl],
                labels["basket_success"][sl][:, None],
            ), axis=-1)
            events_valid = reward_valid.copy()
        else:
            # The first saved frame follows unsaved warm-up actions: omit it.
            end = start + self.sequence_length
            q = g["obs"][start:end+1].astype(np.float32)
            v = g["qvel"][start:end+1].astype(np.float32)
            a = g["actions"][start-1:end].astype(np.float32)
            p = np.concatenate((q[:-1]-DEFAULT_JOINT_POS, v[:-1], a[:-1]), -1)
            np_ = np.concatenate((q[1:]-DEFAULT_JOINT_POS, v[1:], a[1:]), -1)
            image = g["images/rgb"][start:end]
            next_image = g["images/rgb"][start+1:end+1]
            reward = np.zeros(self.sequence_length, np.float32)
            terminated = truncated = np.zeros(self.sequence_length, dtype=bool)
            valid = np.ones(self.sequence_length, dtype=bool)
            reward_valid = np.zeros(self.sequence_length, dtype=bool)
            events = np.zeros((self.sequence_length, 7), np.float32)
            events_valid = reward_valid.copy()
        actor_valid = g["training_labels/actor_valid"][sl] if "training_labels/actor_valid" in g else np.ones(self.sequence_length, dtype=bool)
        first = np.zeros(self.sequence_length, dtype=bool)
        first[0] = True
        # Initial frames rebuild history but do not supervise a cold belief.
        loss_valid = np.ones(self.sequence_length, dtype=bool)
        loss_valid[:min(self.burn_in, self.sequence_length-1)] = False
        return {
            "proprio": torch.tensor(p, dtype=torch.float32),
            "next_proprio": torch.tensor(np_, dtype=torch.float32),
            "image": resize_rgb_uint8(torch.from_numpy(image), self.image_size),
            "next_image": resize_rgb_uint8(torch.from_numpy(next_image), self.image_size),
            "action": torch.tensor(g["actions"][sl], dtype=torch.float32),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "terminated": torch.tensor(terminated),
            "truncated": torch.tensor(truncated),
            "done": torch.tensor(terminated | truncated),
            "next_obs_valid": torch.tensor(valid & loss_valid),
            "reward_valid": torch.tensor(reward_valid & loss_valid),
            "events": torch.tensor(events, dtype=torch.float32),
            "events_valid": torch.tensor(events_valid & loss_valid),
            "expert": torch.tensor(loss_valid & actor_valid & bool(g.attrs.get("actor_supervision", True))),
            "is_first": torch.tensor(first),
        }


def episode_split(path, seed=42):
    with h5py.File(path) as f:
        keys = sorted(f.keys())
    if len(keys) < 2:
        raise ValueError("Need at least two episodes for a held-out split")
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    count = max(1, len(keys)//5)
    return keys[count:], keys[:count]
