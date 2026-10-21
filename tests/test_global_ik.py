import numpy as np

from scripts.act.task_e.global_ik import PiperGlobalIK, forward_kinematics


def test_forward_kinematics_matches_task_e_place_pose():
    joint_pos = np.array([0.747, 2.231, -1.727, 0.0, 1.154, 0.747])
    pose = forward_kinematics(joint_pos)
    np.testing.assert_allclose(pose[:3, 3], [0.324208, 0.300219, 0.150002], atol=5e-4)
    np.testing.assert_allclose(pose[:3, :3], np.diag([-1.0, 1.0, -1.0]), atol=5e-4)


def test_solver_respects_limits_for_sugar_grasp():
    solver = PiperGlobalIK()
    result = solver.solve(
        ee_pos_w=np.array([1.0318, 0.2827, 0.99166]),
        ee_quat_w=np.array([0.0, 2**-0.5, -(2**-0.5), 0.0]),
        root_pose_w=np.array([1.404254, 0.0, 0.826643, 0.0, 0.0, 0.0, 1.0]),
        current_joint_pos=np.array([0.0, 1.2, -1.5, 0.0, 1.2, 0.0]),
    )
    assert result.usable
    assert result.position_error < 0.001
    assert result.orientation_error < np.deg2rad(4.0)
