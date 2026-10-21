"""Offline expert-transition dataset for the Task E world model."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .world_model import resize_rgb_uint8


DEFAULT_JOINT_POS = np.asarray(
    [0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035], dtype=np.float32
)


class ExpertTransitionDataset(Dataset):
    """Read successful collector HDF5 trajectories as one-step transitions.

    The collector records absolute qpos, qvel, the action taken at that state,
    and RGB.  This class reconstructs the environment's 24-D proprio vector:
    relative qpos, qvel, and previous action.
    """

    def __init__(self, path: str | Path, image_size: int = 96, only_success: bool = True):
        self.path = str(Path(path).resolve())
        self.image_size = int(image_size)
        self._file: h5py.File | None = None
        self.transitions: list[tuple[str, int, int, bool]] = []
        with h5py.File(self.path, "r") as file:
            trajectory_keys = sorted(file.keys(), key=lambda key: int(key.split("_")[-1]))
            for key in trajectory_keys:
                group = file[key]
                success = bool(group.attrs.get("success", False))
                if only_success and not success:
                    continue
                if "images" not in group or "rgb" not in group["images"]:
                    raise ValueError(f"{key} has no images/rgb; collect with --save_images.")
                length = min(
                    len(group["obs"]),
                    len(group["qvel"]),
                    len(group["actions"]),
                    len(group["images/rgb"]),
                )
                if length < 2:
                    continue
                # Collector has no post-action state for its last action, so the
                # final available pair is used as the terminal transition.
                for timestep in range(1, length - 1):
                    self.transitions.append((key, timestep, length - 2, success))
        if not self.transitions:
            raise ValueError(f"No usable expert transitions found in {self.path}")

    def __len__(self) -> int:
        return len(self.transitions)

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        key, timestep, final_timestep, success = self.transitions[index]
        group = self._get_file()[key]
        qpos = group["obs"]
        qvel = group["qvel"]
        actions = group["actions"]
        images = group["images/rgb"]

        previous_action = np.zeros(8, dtype=np.float32) if timestep == 0 else actions[timestep - 1]
        proprio = np.concatenate(
            (qpos[timestep] - DEFAULT_JOINT_POS, qvel[timestep], previous_action), axis=0
        ).astype(np.float32)
        next_proprio = np.concatenate(
            (qpos[timestep + 1] - DEFAULT_JOINT_POS, qvel[timestep + 1], actions[timestep]), axis=0
        ).astype(np.float32)
        terminal = False  # legacy collector has no final next-state/reward record
        return {
            "proprio": torch.from_numpy(proprio),
            "image": resize_rgb_uint8(torch.from_numpy(images[timestep]), self.image_size)[0],
            "action": torch.from_numpy(np.asarray(actions[timestep], dtype=np.float32)),
            "reward": torch.tensor(0.0, dtype=torch.float32),
            "reward_valid": torch.tensor(False),
            "next_obs_valid": torch.tensor(True),
            "next_proprio": torch.from_numpy(next_proprio),
            "next_image": resize_rgb_uint8(torch.from_numpy(images[timestep + 1]), self.image_size)[0],
            "done": torch.tensor(float(terminal), dtype=torch.float32),
        }
