from __future__ import annotations

import unittest

import torch

from demo.solution_act import AlgSolution


class ActSolutionTest(unittest.TestCase):
    def test_home_action_holds_home_in_relative_position_mode(self):
        solution = AlgSolution.__new__(AlgSolution)
        solution.default_joint_pos = torch.tensor(
            [[0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035]], dtype=torch.float32
        )
        solution.teleop_home_joint_pos = torch.tensor(
            [[-0.000033, 0.924525, -1.514983, 0.000011, 1.219900, -0.000033, 0.035, -0.035]],
            dtype=torch.float32,
        )
        solution._home_qpos_tolerance = 0.10
        solution._home_stable_steps = 0

        proprio = torch.zeros(1, 24)
        proprio[:, :8] = solution.teleop_home_joint_pos - solution.default_joint_pos
        action, reached = solution._compute_home_action(proprio)
        expected = (solution.teleop_home_joint_pos - solution.default_joint_pos) / 0.5

        self.assertTrue(reached)
        self.assertTrue(torch.allclose(action, expected))
        self.assertGreater(float(action.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
