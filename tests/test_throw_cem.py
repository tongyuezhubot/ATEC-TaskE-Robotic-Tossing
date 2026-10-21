"""Small CPU-only proofs. These are NOT Isaac throwing success tests."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from demo.throw_cem.config import ThrowConfig, DEFAULT_Q, HOME_Q
from demo.throw_cem.kinematics import PiperKinematics, forward_base
from demo.throw_cem.physics import ReleaseResidual, predict_throw, Swing, flight_state, flight_apex_rise
from demo.throw_cem.planner import ThrowPlanner, NoFeasibleThrow, JointTrajectory
from demo.throw_cem.perception import numpy_observation, RGBPosePerception
from demo.throw_cem.evaluation import ThrowEvaluator
from demo.throw_cem.controller import ThrowController
from demo.throw_cem.data import pose_calibration
from demo.throw_cem.residual_fit import fit_release_residual, extract_release_samples
from demo.throw_cem.physics import physics_contract
from demo.throw_cem.joint_model import JointResponse, fit_response
from demo.throw_cem.release_world import ReleaseWorldModel, release_dynamics_contract


class GeometryOnlyPlanner(ThrowPlanner):
    """Optimizer test fixture, not available as a runtime controller option."""
    def joint_trajectory(self, q, rotation, prediction, duration):
        count = int(np.ceil(prediction.swing.end_time/self.cfg.dt))
        targets = np.tile(q, (count, 1))
        targets[:, 6:] = [-.015, .015]
        targets[prediction.open_step:, 6:] = [.035, -.035]
        return JointTrajectory(targets, targets.copy())


class ThrowTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ThrowConfig(throw_primitive="fixed_tool", planning_min_remaining_apex=0.)
        self.kin = PiperKinematics(self.cfg)

    def test_fk_and_bounded_ik(self):
        q = np.array([.747, 2.231, -1.727, 0., 1.154, .747])
        np.testing.assert_allclose(forward_base(q)[:3, 3], [.324208, .300219, .150002], atol=5e-4)
        pose = self.kin.forward(q)
        solution = self.kin.solve(pose[:3, 3], pose[:3, :3], q+.001,
                                  position_tolerance=.0005, rotation_tolerance=.01)
        np.testing.assert_allclose(self.kin.forward(solution)[:3, 3], pose[:3, 3], atol=.0005)

    def test_quintic_endpoints_and_braking(self):
        swing = Swing([0., 0., 0.], [.1, .2, .3], [.2, -.1, .7], .6)
        np.testing.assert_allclose(swing.state(0)[1], np.zeros(3))
        np.testing.assert_allclose(swing.state(.6)[0], [.1, .2, .3], atol=1e-8)
        np.testing.assert_allclose(swing.state(.6)[1], [.2, -.1, .7], atol=1e-8)
        p = predict_throw(self.cfg, 2, np.asarray(self.cfg.release_ee_xyz)+[.06, .12, -.02],
                          np.array([0., 0., -.09]), [.7, .8, .65, .06], ReleaseResidual())
        np.testing.assert_allclose(p.swing.state(p.swing.end_time+.1)[1], np.zeros(3), atol=1e-8)
        self.assertGreater(p.release_velocity[2], .15)
        self.assertGreater(p.flight_seconds, 0)

    def test_open_timing_changes_release_not_just_metadata(self):
        args = (self.cfg, 2, np.asarray(self.cfg.release_ee_xyz)+[.06, .12, -.02], np.array([0., 0., -.09]))
        early = predict_throw(*args, [.7, .8, .65, .12], ReleaseResidual())
        late = predict_throw(*args, [.7, .8, .65, .00], ReleaseResidual())
        self.assertLess(early.open_step, late.open_step)
        self.assertGreater(np.linalg.norm(early.landing_xy-late.landing_xy), .005)

    def test_cem_is_actor_free_and_has_finite_budget(self):
        planner = GeometryOnlyPlanner(self.cfg, self.kin, seed=7)
        plan = planner.plan(DEFAULT_Q, 2, np.array([0., 0., -.09]))
        self.assertEqual(plan.targets.shape[1], 8)
        self.assertEqual(plan.evaluated_candidates, self.cfg.population*self.cfg.iterations)
        safe = np.asarray(self.cfg.basket_half_xy)-np.asarray(self.cfg.object_half_xy[2])-self.cfg.collision_margin
        self.assertTrue(np.all(np.abs(plan.prediction.landing_xy-np.asarray(self.cfg.basket_xyz[:2])) <= safe))
        self.assertFalse(hasattr(planner, "actor"))

    def test_infeasible_cem_fails_without_unsafe_fallback(self):
        planner = GeometryOnlyPlanner(self.cfg, self.kin)
        with patch.object(planner, "joint_trajectory", side_effect=ValueError("infeasible")):
            with self.assertRaises(NoFeasibleThrow):
                planner.plan(DEFAULT_Q, 2, np.array([0., 0., -.09]))

    def test_real_dh_throw_plan_is_bounded_not_simulator_proof(self):
        from dataclasses import replace
        self.cfg = replace(self.cfg, throw_primitive="arm_swing")
        self.kin = PiperKinematics(self.cfg)
        start = np.asarray(self.cfg.release_ee_xyz)+self.cfg.pre_throw_delta
        rotation = self.kin.throw_rotation()
        arm = self.kin.solve(start, rotation, HOME_Q, position_tolerance=1e-6, rotation_tolerance=1e-5)
        q = np.r_[arm, 0., 0.]
        plan = ThrowPlanner(self.cfg, self.kin).plan(q, 2, np.array([-.078, 0., -.045]))
        velocity = np.diff(np.vstack((q[:6], plan.targets[:, :6])), axis=0)/self.cfg.dt
        acceleration = np.diff(np.vstack((np.zeros(6), velocity, np.zeros(6))), axis=0)/self.cfg.dt
        self.assertLessEqual(np.abs(velocity).max(), self.cfg.max_joint_speed)
        self.assertLessEqual(np.abs(acceleration).max(), self.cfg.max_joint_acceleration)

    def test_joint_swing_primitive_stays_bounded_and_opens_only_at_release(self):
        from dataclasses import replace
        cfg = replace(self.cfg, throw_primitive="joint_swing")
        planner = ThrowPlanner(cfg, PiperKinematics(cfg))
        q = np.array([.73555, 1.33661, -.96650, -.45160, .94926, .9398, -.015, .015])
        trajectory, prediction = planner.joint_swing_trajectory(
            q, np.array([.846, -.310, -.595, .584]), 2)
        self.assertTrue(np.all(trajectory.targets[:prediction.open_step, 6:] == [-.015, .015]))
        self.assertTrue(np.all(trajectory.targets[prediction.open_step:, 6:] == [.035, -.035]))
        velocity = np.diff(np.vstack((q[:6], trajectory.targets[:, :6])), axis=0)/cfg.dt
        acceleration = np.diff(np.vstack((np.zeros(6), velocity, np.zeros(6))), axis=0)/cfg.dt
        self.assertLessEqual(np.abs(velocity).max(), cfg.max_joint_speed)
        self.assertLessEqual(np.abs(acceleration).max(), cfg.max_joint_acceleration)
        np.testing.assert_allclose(trajectory.targets[-5:, :6],
                                   np.tile(trajectory.targets[-1, :6], (5, 1)))

    def test_fixed_wrist_commands_project_small_drive_deflection_not_measured_state(self):
        from dataclasses import replace
        from demo.throw_cem.config import ARM_UPPER
        cfg = replace(self.cfg,throw_primitive="joint_swing",joint_swing_duration=.60)
        planner = ThrowPlanner(cfg,PiperKinematics(cfg))
        q = HOME_Q.copy();q[4]=ARM_UPPER[4]+.0005
        original = q.copy()
        trajectory,_ = planner.joint_swing_trajectory(q,[.1,-.1,-.1,.6],1)
        np.testing.assert_array_equal(q,original)
        np.testing.assert_allclose(trajectory.targets[:,4],ARM_UPPER[4])
        self.assertGreater(q[4],ARM_UPPER[4])
        q[4]=ARM_UPPER[4]+.006
        with self.assertRaises(ValueError):
            planner.joint_swing_trajectory(q,[.1,-.1,-.1,.6],1)

    def test_grasp_axis_sign_does_not_force_an_unreachable_wrist(self):
        controller = ThrowController(self.cfg, None)
        for obj, y, quat in [(0, .225, [0., .707, 0., .707]),
                              (1, .15, [0., 0., -.707, .707]), (2, .065, [0., 0., -.707, .707])]:
            position = np.array([1.+(.05 if obj == 1 else 0.), y,
                                 self.cfg.table_z+.03+self.cfg.grasp_offsets[obj]])
            rotation, target = controller.select_grasp(HOME_Q, quat, obj, position)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)
            if obj == 2:
                actor_root = position-[0., 0., self.cfg.grasp_offsets[obj]]
                np.testing.assert_allclose(target+rotation[:, 2]*self.cfg.banana_pinch_distance, actor_root)
            else:
                np.testing.assert_allclose(target, position)

    def test_sugar_centered_grasp_retains_branch_and_pad_depth(self):
        from dataclasses import replace
        from demo.throw_cem.kinematics import quaternion_rotation, hand_table_clearance, sugar_palm_gap
        cfg = replace(self.cfg,sugar_centered_grasp=True,throw_primitive="joint_swing",pre_grasp_clearance=.06,lift_clearance=.10)
        cases = [([1.03204393,.30068287,.87207091],
                  [.090190828,.684257507,-.073999025,.719848275]),
                 ([.91451710,.28388664,.87175918],
                  [.003022069,.730788946,.002340266,.682592750])]
        for root, quat in cases:
            controller = ThrowController(cfg,None)
            rotation,target = controller.select_grasp(HOME_Q,quat,0,np.array(root)+[0.,0.,cfg.grasp_offsets[0]])
            center = np.array(root)+quaternion_rotation(quat)@np.array(cfg.object_com_offsets[0])
            pad_error = target+rotation[:,2]*.120-center
            self.assertTrue(any(abs(np.linalg.norm(pad_error[:2])-h)<1e-8 for h in (0.,.045,.065)))
            self.assertTrue(any(abs(pad_error[2]-h)<1e-8 for h in (0.,.015,.030)))
            self.assertEqual(set(controller.grasp_joint_goals),{"pre_grasp","descend","lift"})
            self.assertGreaterEqual(hand_table_clearance(target,rotation,cfg.table_z),.003)
            self.assertGreaterEqual(sugar_palm_gap(target,rotation,root,quaternion_rotation(quat)),.010)
            goals = controller.grasp_joint_goals
            np.testing.assert_array_equal(goals["descend"][3:],goals["lift"][3:])
            for joint in np.linspace(goals["pre_grasp"],goals["descend"],9):
                tool = controller.kin.forward(joint)
                self.assertGreaterEqual(sugar_palm_gap(tool[:3,3],tool[:3,:3],root,quaternion_rotation(quat)),.010)
        self.assertFalse(ThrowConfig().sugar_centered_grasp)

    def test_sugar_neighbor_screen_uses_visual_pose_and_rejects_unknown(self):
        from dataclasses import replace
        from demo.throw_cem.perception import VisualPose
        from demo.throw_cem.kinematics import open_hand_bottle_gap,quaternion_rotation
        cfg = replace(self.cfg,sugar_centered_grasp=True,throw_primitive="joint_swing",
                      pre_grasp_clearance=.06,lift_clearance=.10)
        xyz = np.array([[1.030916572,.301280916,.872634530],
                        [1.098105788,.188379750,.921435952], [0.916012824,.031204917,.841530919]])
        quat = np.array([[.093737260,.674751043,-.070327036,.728682816],
                         [-.020250278,.021625847,-.697955668,.715527892],
                         [.000542802,.006979916,-.779292166,.626621723]])
        controller = ThrowController(cfg,None)
        controller.poses = VisualPose(xyz,quat,np.ones(3))
        rotation,target = controller.select_grasp(HOME_Q,quat[0],0,xyz[0]+[0.,0.,.14])
        bottle_rotation = quaternion_rotation(quat[1])
        self.assertGreaterEqual(open_hand_bottle_gap(target,rotation,xyz[1],bottle_rotation),.010)
        goals = controller.grasp_joint_goals
        path = list(np.linspace(HOME_Q[:6],goals["pre_grasp"],9))
        path += list(np.linspace(goals["pre_grasp"],goals["descend"],9))
        path += list(np.linspace(goals["descend"],goals["lift"],9))
        for joint in path:
            tool = controller.kin.forward(joint)
            self.assertGreaterEqual(open_hand_bottle_gap(tool[:3,3],tool[:3,:3],xyz[1],bottle_rotation),.010)
        controller.poses.confidence[1] = .5
        with self.assertRaisesRegex(ValueError,"Neighbor bottle pose"):
            controller.select_grasp(HOME_Q,quat[0],0,xyz[0]+[0.,0.,.14])

    def test_sugar_upward_proposal_has_original_budget_and_limits(self):
        from dataclasses import replace
        from demo.throw_cem.config import ARM_LOWER, ARM_UPPER
        cfg = replace(self.cfg,object_order=(0,),sugar_centered_grasp=True,
                      throw_primitive="joint_swing",joint_swing_duration=.60,
                      pre_grasp_clearance=.06,lift_clearance=.10,
                      joint_swing_lower=(-.45,-.85,-.85,.40),
                      joint_response_alpha=(.18,)*6,planning_min_remaining_apex=.024)
        # Recorded held training state, not simulator input at deployment.
        q = np.array([-.558150649,2.139956427,-2.004052639,-.546047688,
                      1.141624880,1.297532916,.019801039,-.021412298])
        offset = np.array([-.022911739,.031668684,-.133770182])
        plan = ThrowPlanner(cfg,PiperKinematics(cfg)).plan(q,0,offset)
        self.assertEqual(plan.evaluated_candidates,240)
        self.assertTrue(np.all(plan.targets[:,:6]>=ARM_LOWER))
        self.assertTrue(np.all(plan.targets[:,:6]<=ARM_UPPER))
        self.assertGreater(plan.prediction.release_velocity[2],0.)

    def test_contact_approach_accepts_only_a_small_settled_vertical_gap(self):
        q = HOME_Q.copy(); actual = self.kin.forward(q)
        for gap, accepted in ((.006, True), (.03, False), (.06, False)):
            controller = ThrowController(self.cfg, None)
            controller.goal = q[:6]+.1  # joint convergence deliberately not satisfied
            controller.move_start = q[:6].copy()
            controller.move_duration = .1
            controller.move_cursor = 5
            for _ in range(3):
                target, done = controller.move(q, np.zeros(8), actual[:3, 3]-[0., 0., gap],
                                               actual[:3, :3], opened=True, contact_approach=True)
            self.assertEqual(done, accepted)
            if accepted:
                np.testing.assert_allclose(target[:6], q[:6])
        # A loose joint convergence check must not bypass banana depth, even
        # before the nominal trajectory ends.
        controller = ThrowController(self.cfg, None)
        controller.goal = q[:6].copy(); controller.move_start = q[:6].copy()
        controller.move_duration = 1.; controller.move_cursor = 0
        for _ in range(3):
            _, done = controller.move(q, np.zeros(8), actual[:3,3]-[0.,0.,.03],
                                     actual[:3,:3], opened=True, contact_approach=True)
        self.assertFalse(done)

    def test_mustard_body_grasp_clears_palm_and_preserves_legacy_default(self):
        from dataclasses import replace
        from demo.throw_cem.kinematics import quaternion_rotation, hand_table_clearance, mustard_palm_gap
        cfg = replace(self.cfg, mustard_body_grasp=True, throw_primitive="joint_swing")
        controller = ThrowController(cfg, None)
        quat = np.array([0., 0., -.707, .707])
        root = np.array([.9754, .155, .9216])
        legacy = root+[.05, 0., cfg.grasp_offsets[1]]
        rotation, target = controller.select_grasp(HOME_Q, quat, 1, legacy)
        com = root+quaternion_rotation(quat)@np.asarray(cfg.object_com_offsets[1])
        shift = target+rotation[:,2]*.120-com
        np.testing.assert_allclose(shift[:2],0.,atol=1e-10)
        self.assertTrue(.029 <= shift[2] <= .061)
        self.assertGreaterEqual(mustard_palm_gap(target,rotation,root,quaternion_rotation(quat)),.010)
        self.assertGreaterEqual(hand_table_clearance(target,rotation,cfg.table_z),.003)
        self.assertIsNotNone(controller.grasp_joint_goals)
        down = controller.kin.forward(controller.grasp_joint_goals["descend"])
        pre = controller.kin.forward(controller.grasp_joint_goals["pre_grasp"])
        np.testing.assert_allclose(pre[:3,3]-down[:3,3],-rotation[:,2]*cfg.pre_grasp_clearance,atol=.016)
        old = ThrowController(self.cfg, None)
        _, old_target = old.select_grasp(HOME_Q,quat,1,legacy)
        np.testing.assert_array_equal(old_target,legacy)

    def test_mustard_palm_projection_rejects_cap_intersection(self):
        from demo.throw_cem.kinematics import quaternion_rotation,grasp_rotations,mustard_palm_gap
        root = np.array([.9754,.155,.9216]);quat=[0.,0.,-.707,.707]
        orientation = quaternion_rotation(quat)
        com = root+orientation@np.asarray(self.cfg.object_com_offsets[1])
        rotation = next(grasp_rotations(quat,1,tilts=(-30.,)))
        self.assertLess(mustard_palm_gap(com-rotation[:,2]*.120,rotation,root,orientation),0.)
        rotation = next(grasp_rotations(quat,1,tilts=(-45.,)))
        position = com+[0.,0.,.06]-rotation[:,2]*.120
        gap = mustard_palm_gap(position,rotation,root,orientation)
        self.assertGreater(gap,.01)
        delta = np.array([1.,-2.,.5])
        self.assertAlmostEqual(gap,mustard_palm_gap(position+delta,rotation,root+delta,orientation))

    def test_mustard_high_arc_proposal_keeps_original_command_limits(self):
        from dataclasses import replace
        from demo.throw_cem.config import ARM_LOWER,ARM_UPPER
        from demo.throw_cem.physics import check_release_margin
        cfg = replace(self.cfg,mustard_body_grasp=True,throw_primitive="joint_swing",
                      joint_swing_duration=.6,joint_swing_lower=(-.45,-.85,-.85,.4),
                      joint_swing_upper=(.9,.65,.65,.8),planning_min_remaining_apex=.08,
                      joint_response_alpha=(.18,)*6)
        # Recorded legal proprio + visually measurable offset at a failed
        # planning state. No ground-truth query or privileged provider needed.
        q = np.array([-.505859017,1.675788593,-1.524509072,.470134914,.849202526,
                      -.697396517,.027061682,-.030062945])
        offset = np.array([-.088042441,.002512027,-.148133707])
        plan = ThrowPlanner(cfg,PiperKinematics(cfg)).plan(q,1,offset)
        self.assertEqual(plan.evaluated_candidates,240)
        self.assertTrue(np.all(plan.targets[:,:6]>=ARM_LOWER))
        self.assertTrue(np.all(plan.targets[:,:6]<=ARM_UPPER))
        check_release_margin(cfg,plan.prediction.release_velocity)

    def test_banana_pinch_can_clear_table_without_weakening_hold_depth(self):
        from demo.throw_cem.kinematics import grasp_rotations, hand_table_clearance
        rotation = list(grasp_rotations([1.,0.,0.,0.],2))[2]
        root = np.array([1.08,.04,self.cfg.table_z+.0144])
        original = root-rotation[:,2]*self.cfg.banana_pinch_distance
        self.assertLess(hand_table_clearance(original,rotation,self.cfg.table_z),0.)
        shifted = root-rotation@np.array([.020,0.,.125])
        self.assertGreater(hand_table_clearance(shifted,rotation,self.cfg.table_z),.003)
        local = rotation.T@(root-shifted)
        self.assertLess(np.max(np.abs(local[:2])),self.cfg.banana_held_lateral_max)
        self.assertTrue(self.cfg.banana_held_local_z_range[0] <= local[2] <= self.cfg.banana_held_local_z_range[1])

    def test_occlusion_does_not_deadlock_an_already_trusted_grasp_motion(self):
        from demo.throw_cem.perception import VisualPose
        class Occluded:
            def observe(self, p, rgb):
                return VisualPose(np.tile([1., .05, .84], (3, 1)),
                                  np.tile([1., 0., 0., 0.], (3, 1)), np.full(3, .1))
        p = np.zeros(24); p[:8] = HOME_Q-DEFAULT_Q
        for stage, advances in (("descend", True), ("plan", False)):
            controller = ThrowController(self.cfg, Occluded())
            controller.stage = stage; controller.target = HOME_Q.copy()
            controller.trusted_grasp_geometry = True
            pose = controller.kin.forward(HOME_Q)
            controller.grasp_xyz = pose[:3, 3]; controller.rotation = pose[:3, :3]
            controller.action(p, np.zeros((224, 224, 3), np.uint8))
            self.assertEqual(controller.move_cursor > 0, advances)
        controller.stage = "plan"
        controller.target[6:] = [-.015, .015]
        p[6:8] = np.array([.025, -.027])-DEFAULT_Q[6:8]
        action, _ = controller.action(p, np.zeros((224, 224, 3), np.uint8))
        command = action*self.cfg.action_scale+DEFAULT_Q
        np.testing.assert_allclose(command[6:], [-.015, .015])

    def test_collection_wrist_move_preserves_carry_rate_limits(self):
        controller = ThrowController(self.cfg, None)
        q = HOME_Q.copy(); q[5] = -.15
        start = q[:6].copy(); goal = q[:6].copy(); goal[5] = .18
        previous = q.copy()
        states = []
        for _ in range(100):
            target, done = controller.move(q, (q-previous)/self.cfg.dt, None, None, joint_goal=goal)
            states.append(target[:6].copy()); previous = q; q = target
            if done:
                break
        self.assertTrue(done)
        velocity = np.diff(np.vstack((start, states)), axis=0)/self.cfg.dt
        acceleration = np.diff(np.vstack((np.zeros(6), velocity, np.zeros(6))), axis=0)/self.cfg.dt
        self.assertLessEqual(np.abs(velocity).max(), self.cfg.carry_max_joint_speed)
        self.assertLessEqual(np.abs(acceleration).max(), self.cfg.carry_max_joint_acceleration)
        self.assertAlmostEqual(q[5], .18)

    def test_joint_response_is_a_forward_model_not_an_actor(self):
        from dataclasses import replace
        model = JointResponse(replace(self.cfg, joint_response_alpha=(.18,)*6))
        trajectory = model.rollout(np.zeros(6), np.ones((5, 8)))
        np.testing.assert_allclose(trajectory[-1], np.full(6, 1-.82**5))
        self.assertFalse(model.calibrated)
        self.assertFalse(hasattr(model, "actor"))
        rng = np.random.default_rng(43)
        def segments(count):
            data = []
            for _ in range(count):
                initial = rng.normal(size=6); commands = rng.normal(size=(5, 6)); final = initial.copy()
                for command in commands:
                    final += .18*(command-final)
                data.append((initial, commands, final))
            return data
        alpha, report = fit_response(segments(50), segments(20))
        self.assertTrue(report["passed"])
        np.testing.assert_allclose(alpha, np.full(6, .18))

    def test_release_world_requires_calibration_and_rejects_extrapolation(self):
        from dataclasses import replace
        cfg = replace(self.cfg, throw_primitive="joint_swing", joint_swing_duration=.60)
        theta = np.array([.8, -.5, -.6, .5]); offset = np.array([0., 0., -.12])
        feature = np.r_[HOME_Q[:6], offset, theta, cfg.joint_swing_duration]
        weights = np.zeros((6, 15)); weights[5, 0] = -.4
        payload = {"format": "atec_release_world_v3", "calibrated": True,
                   "state_time_basis": "first_detected_freeflight_observation",
                   "supported_object_ids": [3], "weights": weights.tolist(),
                   "feature_mean": feature.tolist(), "feature_std": np.ones(14).tolist(),
                   "feature_bounds": np.stack((feature-.1, feature+.1), axis=-1).tolist(),
                   "release_dynamics_contract": release_dynamics_contract(physics_contract(cfg))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"synthetic_world.json"
            path.write_text(json.dumps(payload))
            model = ReleaseWorldModel(path, cfg)
            self.assertEqual(model.candidate_proposals(2,HOME_Q,offset).shape,(0,4))
            low, high = model.candidate_bounds(2, HOME_Q, offset)
            np.testing.assert_allclose(low, theta-.1)
            np.testing.assert_allclose(high, theta+.1)
            with self.assertRaises(ValueError):
                model.candidate_bounds(2, HOME_Q+np.r_[.2,np.zeros(7)], offset)
            low[0] -= 10  # returned proposal bounds must not mutate model guard
            self.assertAlmostEqual(model.bounds[9,0], theta[0]-.1)
            np.testing.assert_allclose(model.correction(2, theta, offset, q=HOME_Q), [0., 0., 0., 0., 0., -.4])
            with self.assertRaises(ValueError):
                model.correction(2, theta+[.2, 0., 0., 0.], offset, q=HOME_Q)
            self.assertEqual(model.rejections, 1)
            payload["calibrated"] = False
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                ReleaseWorldModel(path, cfg)
        from demo.throw_cem.physics import check_release_margin
        observed_cfg = replace(cfg, planning_min_remaining_apex=.012)
        with self.assertRaises(ValueError):
            check_release_margin(observed_cfg, np.array([0., 0., .60]))
        check_release_margin(observed_cfg, np.array([0., 0., .60]), observation_aligned=True)

    def test_world_proposals_use_training_only_and_do_not_mutate_guards(self):
        from dataclasses import replace
        cfg = replace(self.cfg,throw_primitive="joint_swing",joint_swing_duration=.6)
        theta = np.array([.8,-.5,-.6,.5]);offset=np.array([0.,0.,-.12])
        feature=np.r_[HOME_Q[:6],offset,theta,.6]
        second=feature.copy();second[0]+=.02;second[9]+=.01
        payload={"format":"atec_release_world_v3","calibrated":True,
                 "state_time_basis":"first_detected_freeflight_observation",
                 "supported_object_ids":[3],"weights":np.zeros((6,15)).tolist(),
                 "feature_mean":feature.tolist(),"feature_std":np.ones(14).tolist(),
                 "feature_bounds":np.stack((feature-.1,feature+.1),axis=-1).tolist(),
                 "release_dynamics_contract":release_dynamics_contract(physics_contract(cfg)),
                 "calibration":{"train_seeds":[10,11],"val_seeds":[12]},
                 "proposal_examples":{"source_split":"train","seeds":[10,11],
                                      "features":[feature.tolist(),second.tolist()]}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"proposals.json";path.write_text(json.dumps(payload))
            model=ReleaseWorldModel(path,cfg)
            proposals=model.candidate_proposals(2,HOME_Q,offset)
            np.testing.assert_allclose(proposals[0],theta)
            proposals[0,0]+=10
            np.testing.assert_allclose(model.candidate_proposals(2,HOME_Q,offset)[0],theta)
            self.assertEqual(model.calls,0)  # seeds themselves are not scored plans
            payload["proposal_examples"]["seeds"][0]=12
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,"training-only"):
                ReleaseWorldModel(path,cfg)
    def test_pose_gate_never_calibrates_bad_quaternions_or_positions(self):
        xyz = np.zeros((40, 3, 3)); quat = np.zeros((40, 3, 4)); quat[..., 0] = 1.
        probabilities = np.full((40, 3), .9)
        self.assertTrue(pose_calibration(xyz, -quat, probabilities, xyz, quat)["passed"])
        self.assertFalse(pose_calibration(xyz+.10, quat, probabilities, xyz, quat)["passed"])
        self.assertFalse(pose_calibration(xyz, np.zeros_like(quat), probabilities, xyz, quat)["passed"])

    def test_release_fit_and_physics_domain_guard(self):
        rng = np.random.default_rng(43)
        def sample(obj, seed):
            theta = np.array([.35, .50, .55, .06])+rng.uniform(-.01, .01, 4)
            offset = np.array([-.078, 0., -.045])+rng.uniform(-.005, .005, 3)
            return {"object": obj, "seed": seed, "theta": theta.tolist(), "offset": offset.tolist(),
                    "delta": [.005, -.005, 0., .02, 0., .03]}
        train = [sample(obj, seed) for obj in (1, 2, 3) for seed in range(12)]
        val = [sample(obj, seed) for obj in (1, 2, 3) for seed in range(20, 24)]
        report, coeff, bounds = fit_release_residual(train, val)
        self.assertTrue(report["passed"])
        self.assertFalse(fit_release_residual(train[:1], val)[0]["passed"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"test_synthetic_residual.json"
            path.write_text(json.dumps({"format": "atec_throw_release_residual_v1", "calibrated": True,
                                        "coefficients": coeff.tolist(), "feature_bounds": bounds.tolist(),
                                        "physics": physics_contract(self.cfg)}))
            model = ReleaseResidual(path, self.cfg)
            np.testing.assert_allclose(model.correction(0, [.35, .50, .55, .06], [-.078, 0., -.045]),
                                       [.005, -.005, 0., .02, 0., .03], atol=.001)
            np.testing.assert_allclose(model.correction(0, [.7, .9, .85, .12], [-.078, 0., -.045]), np.zeros(6))
            from dataclasses import replace
            with self.assertRaises(ValueError):
                ReleaseResidual(path, replace(self.cfg, gripper_delay_prior=.1))

    def test_release_labels_are_at_the_same_physics_reference_time(self):
        from dataclasses import asdict
        cfg = self.cfg; theta = [.35, .70, .5, .02]; offset = np.array([-.078, 0., -.045])
        start = np.asarray(cfg.release_ee_xyz)+cfg.pre_throw_delta
        pred = predict_throw(cfg, 2, start, offset, theta, ReleaseResidual())
        reference = pred.open_step*cfg.dt+cfg.release_delay(2)
        actual_pos = pred.release_position+[.005, -.005, 0.]
        actual_vel = pred.release_velocity+[.02, 0., .03]
        rows = []
        for step in range(1, 40):
            t = step*cfg.dt; delta = t-reference
            pos, vel = flight_state(cfg, actual_pos, actual_vel, delta)
            rows.append({"step": step, "action_stage": "swing", "controller": {
                "cem_plans": 1, "object": 3, "swing_cursor": step, "parameters": theta,
                "object_offset": offset.tolist(), "plan_start": start.tolist()}, "evaluation_only": {
                "object_xyz": np.tile(pos, (3, 1)).tolist(), "object_velocity": np.tile(vel, (3, 1)).tolist(),
                "ee_xyz": (actual_pos-offset).tolist(), "jaws": [.035, -.035],
                "held_evidence": [True]*3, "upward_release": [t >= reference]*3}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"run_config.json").write_text(json.dumps({"seed": 43, "throw_config": asdict(cfg), "physics": physics_contract(cfg)}))
            (path/"trace.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            samples, _ = extract_release_samples(path)
            self.assertEqual(len(samples), 1)
            np.testing.assert_allclose(samples[0]["delta"], [.005, -.005, 0., .02, 0., .03], atol=1e-7)

    def test_native_drag_recurrence_and_inverse(self):
        p = np.array([1.0465, -.2178, 1.1032]); v = np.array([.0287, -.2373, .2134])
        initial = p.copy(); original_v = v.copy()
        for _ in range(4):
            v = (v+[0., 0., -9.81*self.cfg.physics_dt])*(1-2*self.cfg.physics_dt)
            p += v*self.cfg.physics_dt
        actual_p, actual_v = flight_state(self.cfg, initial, original_v, self.cfg.dt)
        np.testing.assert_allclose(actual_p, p, atol=1e-12)
        np.testing.assert_allclose(actual_v, v, atol=1e-12)
        previous_p, previous_v = flight_state(self.cfg, actual_p, actual_v, -self.cfg.dt)
        np.testing.assert_allclose(previous_p, initial, atol=1e-12)
        np.testing.assert_allclose(previous_v, original_v, atol=1e-12)
        self.assertAlmostEqual(actual_v[2], .01365, delta=.001)
        self.assertLess(flight_apex_rise(self.cfg, .421), .01)

    def test_vertical_toss_near_basket_is_not_rejected_as_placement(self):
        offset = np.asarray(self.cfg.basket_xyz)-np.asarray(self.cfg.release_ee_xyz)
        offset[2] = -.05
        start = np.asarray(self.cfg.release_ee_xyz)+self.cfg.pre_throw_delta
        prediction = predict_throw(self.cfg, 2, start, offset, [.0, .8, .6, .0], ReleaseResidual())
        np.testing.assert_allclose(prediction.release_velocity[:2], np.zeros(2), atol=1e-12)
        self.assertGreater(flight_apex_rise(self.cfg, prediction.release_velocity[2]), .01)

    def test_standard_input_and_missing_perception_are_fail_closed(self):
        obs = {"proprio": np.zeros((1, 24)), "image": {"video_rgb": np.zeros((1, 32, 32, 3), np.uint8),
                "video_depth": object()}, "object_truth": object()}
        p, rgb = numpy_observation(obs)
        self.assertEqual(p.shape, (24,)); self.assertEqual(rgb.shape, (32, 32, 3))
        with self.assertRaises(FileNotFoundError):
            RGBPosePerception(None, self.cfg)
        with self.assertRaises(ValueError):
            numpy_observation({"proprio": np.zeros(8), "image": obs["image"]})

    def test_specialist_rgb_head_cannot_control_uncalibrated_objects(self):
        from dataclasses import replace
        from demo.throw_cem.perception import _require_held_scope
        _require_held_scope(replace(self.cfg,object_order=(1,)),[2])
        for order in [(2,),(1,2),(2,1,0)]:
            with self.assertRaisesRegex(ValueError,"does not cover"):
                _require_held_scope(replace(self.cfg,object_order=order),[2])

    def test_training_stance_carry_is_opt_in_small_and_does_not_change_observation(self):
        from dataclasses import replace
        from types import SimpleNamespace
        cfg=replace(self.cfg,object_order=(1,),mustard_body_grasp=True,normalize_throw_stance=True)
        controller=ThrowController(cfg,None);q=HOME_Q.copy();original=q.copy()
        good=q[:6].copy();good[0]+=.02
        bad=q[:6].copy();bad[0]+=.5
        world=SimpleNamespace(world_model=True,
                              candidate_stances=lambda obj,state:np.array([bad,good]),
                              candidate_bounds=lambda obj,state,offset:(np.zeros(4),np.ones(4)))
        controller.planner.residual=world
        target=controller.select_training_stance(q,1,np.array([0.,0.,-.12]))
        np.testing.assert_allclose(target,good)
        np.testing.assert_array_equal(q,original)
        controller.cfg=replace(cfg,normalize_throw_stance=False)
        self.assertIsNone(controller.select_training_stance(q,1,np.array([0.,0.,-.12])))
        controller.cfg=cfg;world.candidate_stances=lambda obj,state:np.array([bad])
        self.assertIsNone(controller.select_training_stance(q,1,np.array([0.,0.,-.12])))

    def test_drop_is_not_throw_but_upward_flight_can_settle(self):
        initial = np.tile([1., .1, self.cfg.table_z+.03], (3, 1))
        monitor = ThrowEvaluator(self.cfg, initial)
        held = initial.copy(); held[0] = [1.10, -.05, 1.15]
        ee = held[0]+[0., 0., .10]
        for _ in range(6):
            monitor.update(held, np.zeros((3, 3)), ee, np.zeros(3), [0., 0.])
        dropped = held.copy(); dropped[0] = self.cfg.basket_xyz
        for _ in range(30):
            monitor.update(dropped, np.zeros((3, 3)), ee, np.zeros(3), [.035, -.035])
        self.assertFalse(monitor.success.any())
        monitor = ThrowEvaluator(self.cfg, initial)
        for _ in range(6):
            monitor.update(held, np.zeros((3, 3)), ee, np.zeros(3), [0., 0.])
        v0 = np.array([0., -.7, .8])
        for index in range(16):
            t = index*self.cfg.dt
            pos = held.copy(); vel = np.zeros((3, 3))
            pos[0] = held[0]+v0*t+[0., 0., -.5*9.81*t*t]
            vel[0] = v0+[0., 0., -9.81*t]
            monitor.update(pos, vel, ee, np.zeros(3), [.035, -.035])
        for _ in range(30):
            monitor.update(dropped, np.zeros((3, 3)), ee, np.zeros(3), [.035, -.035])
        self.assertTrue(monitor.success[0]); self.assertFalse(monitor.success.all())

    def test_default_dispatcher_does_not_import_act(self):
        with patch.dict(os.environ, {"ATEC_TASK_E_BACKEND": "throw_cem"}):
            module = importlib.import_module("demo.solution")
            self.assertEqual(module.AlgSolution.__module__, "demo.solution_throw_cem")
            self.assertNotIn("demo.solution_act", sys.modules)

    def test_actual_upward_flight_does_not_require_full_jaw_opening(self):
        initial = np.tile([1., .1, self.cfg.table_z+.03], (3, 1))
        held = initial.copy(); held[0] = [1.10, -.05, 1.15]
        ee = held[0]+[0., 0., .10]
        monitor = ThrowEvaluator(self.cfg, initial)
        for _ in range(6):
            monitor.update(held, np.zeros((3, 3)), ee, np.zeros(3), [.025, -.025])
        v0 = np.array([0., -.7, .8])
        for index in range(16):
            t = index*self.cfg.dt; pos = held.copy(); vel = np.zeros((3, 3))
            pos[0] = held[0]+v0*t+[0., 0., -.5*9.81*t*t]
            vel[0] = v0+[0., 0., -9.81*t]
            monitor.update(pos, vel, ee, np.zeros(3), [.0275, -.0275], opening_command=True)
        self.assertTrue(monitor.released_up[0])
        self.assertGreaterEqual(monitor.flight_count[0], 3)

    def test_wide_banana_hold_is_not_rejected_or_confused_with_full_open(self):
        initial = np.tile([1., .1, self.cfg.table_z+.03], (3, 1))
        held = initial.copy(); held[2] = [1.10, -.05, 1.15]
        ee = held[2]+[0., 0., .12]
        monitor = ThrowEvaluator(self.cfg, initial)
        for _ in range(6):
            monitor.update(held, np.zeros((3, 3)), ee, np.zeros(3), [.033, -.033], opening_command=False)
        self.assertTrue(monitor.held[2])
        self.assertFalse(monitor.success.any())
        monitor = ThrowEvaluator(self.cfg, initial)
        for _ in range(6):
            monitor.update(held, np.zeros((3, 3)), ee, np.zeros(3), [.035, -.035], opening_command=True)
        self.assertFalse(monitor.held.any())

    def test_plan_waits_for_stability_without_changing_the_hold_target(self):
        from types import SimpleNamespace
        from demo.throw_cem.perception import VisualPose
        q = HOME_Q.copy(); q[6:] = [.025, -.025]
        tool = self.kin.forward(q)
        xyz = np.tile(tool[:3, 3]+tool[:3, :3]@np.array([0., 0., .115]), (3, 1))
        xyz[2] -= np.asarray(self.cfg.object_com_offsets[2])
        # Tracking needs calibrated root/COM, not the acquisition orientation gate.
        pose = VisualPose(xyz, np.tile([1., 0., 0., 0.], (3, 1)), np.full(3, .1), np.ones(3))
        perception = SimpleNamespace(observe=lambda p, rgb: pose)
        controller = ThrowController(self.cfg, perception)
        controller.enter("plan"); controller.initial_object_z = xyz[2, 2]-.12
        target = q.copy(); target[0] += .002; target[6:] = [-.015, .015]
        controller.target = target.copy()
        controller.last_q = q.copy(); controller.last_q[0] -= .01
        p = np.r_[q-DEFAULT_Q, np.zeros(16)]
        with patch.object(controller.planner, "plan", return_value=SimpleNamespace(evaluated_candidates=128)) as plan:
            action, stopped = controller.action(p, np.zeros((224, 224, 3), np.uint8))
            self.assertFalse(stopped); self.assertEqual(controller.stage, "plan")
            np.testing.assert_allclose(action*self.cfg.action_scale+DEFAULT_Q, target)
            plan.assert_not_called()
            for _ in range(3):
                controller.action(p, np.zeros((224, 224, 3), np.uint8))
            self.assertEqual(controller.stage, "swing"); plan.assert_called_once()

    def test_transport_hold_geometry_uses_same_frame_pose_not_cached_rgb(self):
        from types import SimpleNamespace
        from demo.throw_cem.perception import VisualPose
        previous = HOME_Q.copy(); previous[6:] = [.025, -.025]
        current = previous.copy(); current[0] += .15
        def visual(q):
            tool = self.kin.forward(q)
            com = tool[:3, 3]+tool[:3, :3]@np.array([0., 0., .115])
            xyz = np.tile(com, (3, 1)); xyz[2] -= self.cfg.object_com_offsets[2]
            return VisualPose(xyz, np.tile([1., 0., 0., 0.], (3, 1)), np.ones(3))
        fresh = visual(current)
        perception = SimpleNamespace(observe=lambda p, rgb: fresh)
        controller = ThrowController(self.cfg, perception)
        controller.enter("pre_throw"); controller.step = controller.pose_step = 26
        controller.poses = visual(previous); controller.last_q = current.copy()
        controller.initial_object_z = fresh.positions[2, 2]-.12
        controller.held_rotation = self.kin.forward(current)[:3, :3]
        controller.target = current.copy()
        with patch.object(controller, "move", return_value=(current.copy(), False)):
            _, stopped = controller.action(np.r_[current-DEFAULT_Q, np.zeros(16)], np.zeros((224, 224, 3), np.uint8))
        self.assertFalse(stopped); self.assertEqual(controller.stage, "pre_throw")
        self.assertEqual(controller.pose_step, 27)


if __name__ == "__main__":
    unittest.main()
