from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from demo.world_model import (
    CEMConfig,
    CEMPlanner,
    ReplayBuffer,
    WorldModel,
    WorldModelConfig,
    load_world_model,
    resize_rgb_uint8,
    save_world_model,
)
from demo.world_model_dataset import DEFAULT_JOINT_POS, ExpertTransitionDataset
from demo.solution_world_model import AlgSolution


def small_config() -> WorldModelConfig:
    return WorldModelConfig(
        proprio_dim=24,
        action_dim=8,
        image_size=32,
        image_feature_dim=24,
        latent_dim=16,
        hidden_dim=32,
    )


class WorldModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = small_config()
        self.model = WorldModel(self.config)

    def make_batch(self, batch_size: int = 4):
        return {
            "proprio": torch.randn(batch_size, 24),
            "image": torch.randint(0, 256, (batch_size, 3, 32, 32), dtype=torch.uint8),
            "action": torch.empty(batch_size, 8).uniform_(-1, 1),
            "reward": torch.randn(batch_size),
            "next_proprio": torch.randn(batch_size, 24),
            "next_image": torch.randint(0, 256, (batch_size, 32, 32, 3), dtype=torch.uint8),
            "done": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        }

    def test_encoder_dynamics_and_heads(self):
        batch = self.make_batch()
        latent = self.model.encode(batch["proprio"], batch["image"])
        next_latent = self.model.next_latent(latent, batch["action"])
        reward = self.model.predict_reward(latent, batch["action"])
        value = self.model.predict_value(latent)
        self.assertEqual(tuple(latent.shape), (4, 16))
        self.assertEqual(tuple(next_latent.shape), (4, 16))
        self.assertEqual(tuple(reward.shape), (4,))
        self.assertEqual(tuple(value.shape), (4,))
        self.assertTrue(torch.isfinite(next_latent).all())

    def test_loss_is_finite_and_backpropagates(self):
        losses = self.model.compute_loss(self.make_batch())
        self.assertEqual(
            set(losses), {"dynamics", "reward", "value", "proprio", "inverse", "actor", "total"}
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        gradient_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_norm, 0.0)

    def test_replay_buffer_resizes_wraps_and_samples(self):
        replay = ReplayBuffer(capacity=5, config=self.config)
        for _ in range(3):
            replay.add(
                torch.randn(2, 24),
                torch.randint(0, 256, (2, 48, 64, 4), dtype=torch.uint8),
                torch.randn(2, 8),
                torch.randn(2),
                torch.randn(2, 24),
                torch.randint(0, 256, (2, 3, 40, 50), dtype=torch.uint8),
                torch.zeros(2),
            )
        self.assertEqual(len(replay), 5)
        batch = replay.sample(4, "cpu")
        self.assertEqual(tuple(batch["image"].shape), (4, 3, 32, 32))
        self.assertEqual(batch["image"].dtype, torch.uint8)
        self.assertEqual(tuple(batch["action"].shape), (4, 8))

    def test_cem_action_shape_and_bounds(self):
        planner = CEMPlanner(
            self.model,
            CEMConfig(horizon=3, population=12, elites=3, iterations=2),
        )
        action = planner.plan(torch.randn(2, 16))
        self.assertEqual(tuple(action.shape), (2, 8))
        self.assertTrue(torch.isfinite(action).all())
        self.assertLessEqual(float(action.max()), self.config.action_limit)
        self.assertGreaterEqual(float(action.min()), -self.config.action_limit)

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            save_world_model(path, self.model, step=123, metadata={"smoke_only": True})
            restored, payload = load_world_model(path)
            self.assertEqual(restored.config, self.config)
            self.assertEqual(payload["step"], 123)
            original = self.model.state_dict()
            for name, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, original[name]), name)

    def test_rgb_conversion_accepts_hwc_float(self):
        image = torch.rand(2, 40, 50, 3)
        resized = resize_rgb_uint8(image, 32)
        self.assertEqual(tuple(resized.shape), (2, 3, 32, 32))
        self.assertEqual(resized.dtype, torch.uint8)

    def test_expert_dataset_reconstructs_24d_proprio_and_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.hdf5"
            with h5py.File(path, "w") as file:
                group = file.create_group("traj_0")
                group.attrs["success"] = True
                qpos = np.tile(DEFAULT_JOINT_POS, (4, 1))
                qpos[:, 0] += np.arange(4, dtype=np.float32)
                group.create_dataset("obs", data=qpos)
                group.create_dataset("qvel", data=np.ones((4, 8), dtype=np.float32))
                group.create_dataset("actions", data=np.full((4, 8), 0.25, dtype=np.float32))
                images = group.create_group("images")
                images.create_dataset("rgb", data=np.zeros((4, 24, 30, 3), dtype=np.uint8))
            dataset = ExpertTransitionDataset(path, image_size=32)
            try:
                self.assertEqual(len(dataset), 2)
                first = dataset[0]
                final = dataset[1]
                self.assertEqual(tuple(first["proprio"].shape), (24,))
                self.assertEqual(tuple(first["image"].shape), (3, 32, 32))
                self.assertTrue(torch.equal(first["proprio"][16:], torch.full((8,), 0.25)))
                self.assertEqual(float(first["done"]), 0.0)
                self.assertEqual(float(final["done"]), 0.0)
                self.assertEqual(float(final["reward"]), 0.0)
                self.assertFalse(bool(final["reward_valid"]))
            finally:
                dataset.close()

    def test_solution_can_select_behavior_cloned_actor(self):
        solution = AlgSolution.__new__(AlgSolution)
        solution.control_mode = "actor"
        solution.model = self.model
        latent = torch.randn(2, self.config.latent_dim)
        self.assertTrue(torch.equal(solution._policy_action(latent), self.model.actor(latent)))

        class FailingPlanner:
            def plan(self, _latent):
                raise AssertionError("CEM should not run in actor mode")

        solution.planner = FailingPlanner()
        self.assertTrue(torch.equal(solution._policy_action(latent), self.model.actor(latent)))

    def test_home_action_holds_home_in_relative_position_mode(self):
        solution = AlgSolution.__new__(AlgSolution)
        solution.default_joint_pos = torch.tensor(
            [[0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035]], dtype=torch.float32
        )
        solution.teleop_home_joint_pos = torch.tensor(
            [[-0.000033, 0.924525, -1.514983, 0.000011, 1.219900, -0.000033, 0.035, -0.035]],
            dtype=torch.float32,
        )
        proprio = torch.zeros(1, 24)
        proprio[:, :8] = solution.teleop_home_joint_pos - solution.default_joint_pos
        action, reached = solution._home_action(proprio)
        expected = (solution.teleop_home_joint_pos - solution.default_joint_pos) / 0.5
        self.assertTrue(reached)
        self.assertTrue(torch.allclose(action, expected))
        self.assertGreater(float(action.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
