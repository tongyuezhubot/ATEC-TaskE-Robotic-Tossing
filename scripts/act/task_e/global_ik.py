"""Bounded numerical IK for the Piper scripted expert.

The DH parameters match AgileX's official Piper SDK and the Task E USD model.
This solver is used only to generate expert joint targets; learned policies do
not receive privileged state or call it at evaluation time.
"""

from dataclasses import dataclass
from math import pi

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


_A = np.array([0.0, 0.0, 0.28503, -0.02198, 0.0, 0.0])
_ALPHA = np.array([0.0, -pi / 2, 0.0, pi / 2, -pi / 2, pi / 2])
_THETA_OFFSET = np.array([0.0, -172.22 * pi / 180, -102.78 * pi / 180, 0.0, 0.0, 0.0])
_D = np.array([0.123, 0.0, 0.0, 0.25075, 0.0, 0.091])

_LOWER = np.deg2rad([-150.00035, 0.0, -169.99657, -99.98113, -69.90085, -120.00027])
_UPPER = np.deg2rad([124.21724, 179.90874, 0.0, 99.98113, 69.90085, 120.00027])
_HOME = np.array([0.0, 1.2, -1.5, 0.0, 1.2, 0.0])


@dataclass(frozen=True)
class IKResult:
    joint_pos: np.ndarray
    position_error: float
    orientation_error: float
    evaluations: int

    @property
    def usable(self) -> bool:
        return self.position_error <= 0.003 and self.orientation_error <= np.deg2rad(6.0)


def forward_kinematics(joint_pos: np.ndarray) -> np.ndarray:
    """Return base-to-gripper transform for six Piper arm joints."""
    transform = np.eye(4)
    for a, alpha, theta_offset, d, joint in zip(
        _A, _ALPHA, _THETA_OFFSET, _D, np.asarray(joint_pos, dtype=np.float64)
    ):
        theta = joint + theta_offset
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_alpha, s_alpha = np.cos(alpha), np.sin(alpha)
        link = np.array(
            [
                [c_theta, -s_theta, 0.0, a],
                [s_theta * c_alpha, c_theta * c_alpha, -s_alpha, -s_alpha * d],
                [s_theta * s_alpha, c_theta * s_alpha, c_alpha, c_alpha * d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        transform = transform @ link
    return transform


class PiperGlobalIK:
    """Solve an absolute world-frame gripper pose within Piper joint limits."""

    def __init__(self, position_weight: float = 100.0, max_evaluations: int = 300):
        self.position_weight = float(position_weight)
        self.max_evaluations = int(max_evaluations)

    def solve(
        self,
        ee_pos_w: np.ndarray,
        ee_quat_w: np.ndarray,
        root_pose_w: np.ndarray,
        current_joint_pos: np.ndarray,
    ) -> IKResult:
        root_pose_w = np.asarray(root_pose_w, dtype=np.float64)
        root_rot_w = Rotation.from_quat(
            [root_pose_w[4], root_pose_w[5], root_pose_w[6], root_pose_w[3]]
        ).as_matrix()
        target_rot_w = Rotation.from_quat(
            [ee_quat_w[1], ee_quat_w[2], ee_quat_w[3], ee_quat_w[0]]
        ).as_matrix()
        target_pos_b = root_rot_w.T @ (np.asarray(ee_pos_w, dtype=np.float64) - root_pose_w[:3])
        target_rot_b = root_rot_w.T @ target_rot_w

        def residual(joint_pos: np.ndarray) -> np.ndarray:
            current = forward_kinematics(joint_pos)
            position_error = current[:3, 3] - target_pos_b
            orientation_error = Rotation.from_matrix(
                current[:3, :3].T @ target_rot_b
            ).as_rotvec()
            return np.concatenate((self.position_weight * position_error, orientation_error))

        seeds = (np.asarray(current_joint_pos, dtype=np.float64), _HOME)
        best = None
        epsilon = 1e-6
        for seed in seeds:
            solution = least_squares(
                residual,
                np.clip(seed, _LOWER + epsilon, _UPPER - epsilon),
                bounds=(_LOWER, _UPPER),
                max_nfev=self.max_evaluations,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
            pose = forward_kinematics(solution.x)
            position_error = float(np.linalg.norm(pose[:3, 3] - target_pos_b))
            orientation_error = float(
                np.linalg.norm(Rotation.from_matrix(pose[:3, :3].T @ target_rot_b).as_rotvec())
            )
            rank = self.position_weight * position_error + orientation_error
            candidate = (rank, solution, position_error, orientation_error)
            if best is None or candidate[0] < best[0]:
                best = candidate

        _, solution, position_error, orientation_error = best
        return IKResult(
            joint_pos=solution.x.astype(np.float32),
            position_error=position_error,
            orientation_error=orientation_error,
            evaluations=int(solution.nfev),
        )
