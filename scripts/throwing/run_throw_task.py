"""Independent toss experiment, original Task-E geometry/actuators unchanged.

--oracle-training is explicitly privileged calibration, never RGB deployment proof.
Default policy only observes standard RGB+24-D proprio. No ACT/old WM actor loads.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True)
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--max-steps", type=int, default=6000)
parser.add_argument("--oracle-training", action="store_true", help="Privileged pose provider for calibration only, explicitly marked in results")
parser.add_argument("--save-dataset", action="store_true", help="Record same-frame RGB/proprio and privileged pose TRAINING labels")
parser.add_argument("--data-stride", type=int, default=5, help="Save one pose training frame per N simulation steps")
parser.add_argument("--throw-primitive", choices=("arm_swing", "fixed_tool", "joint_swing"), default="arm_swing",
                    help="Explicit bounded motion comparison, recorded in the physics contract")
parser.add_argument("--planner-config", help="Optional JSON of static motion/timing overrides, never scene/actuator edits")
parser.add_argument("--objects", nargs="+", type=int, choices=(1, 2, 3), default=[3, 2, 1],
                    help="Object order/subset for targeted calibration; all-three success stays separate")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.max_steps <= 6000:
    parser.error("Use a finite 1..6000 step budget")
if not 1 <= args.data_stride <= 50:
    parser.error("Use a data stride in 1..50")
if len(set(args.objects)) != len(args.objects):
    parser.error("Do not repeat objects")
motion_overrides = {} if args.planner_config is None else json.loads(Path(args.planner_config).read_text())
from demo.throw_cem.config import MOTION_FIELDS
allowed_overrides = MOTION_FIELDS
if not isinstance(motion_overrides, dict) or set(motion_overrides)-allowed_overrides:
    parser.error("Planner JSON may only override documented static motion/timing fields")
if args.save_dataset:
    import h5py  # fail before loading expensive simulator when dependency is missing
out = Path(args.output_dir)
if out.exists():
    parser.error("Choose a NEW output directory")
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import atec_rl_lab.tasks
from atec_rl_lab.tasks.task_e.env_cfg import TaskEEnvPiperCfg
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec
from demo.solution_throw_cem import AlgSolution
from demo.throw_cem.config import ThrowConfig, DEFAULT_Q
from demo.throw_cem.perception import VisualPose, numpy_observation
from demo.throw_cem.data import PoseDatasetWriter
from demo.throw_cem.evaluation import ThrowEvaluator
from demo.throw_cem.physics import physics_contract

env = writer = None
collection_complete = False
try:
    # Object layout is sampled in __post_init__. Setting seed AFTER the registry
    # helper constructs cfg is too late and made repeated seed labels unpaired.
    cfg = TaskEEnvPiperCfg(seed=args.seed)
    cfg.sim.device = args.device; cfg.scene.num_envs = 1; cfg.sim.use_fabric = True
    # Local experiment instance only. Do not edit official Task-E source/registry.
    cfg.terminations.basket_success = None
    cfg.episode_length_s = 125.
    cfg = apply_safe_action_spec(cfg, {})
    env = gym.make("ATEC-TaskE-Piper", cfg=cfg)
    obs, _ = env.reset(seed=args.seed)
    base = env.unwrapped
    robot = base.scene["robot"]
    ee_id = robot.find_bodies("gripper_base")[0][0]
    scene_geometry = {}
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/envs/env_0/Basket")
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"]).ComputeWorldBound(prim).ComputeAlignedRange()
        scene_geometry = {"basket_visual_aabb_min": list(bounds.GetMin()), "basket_visual_aabb_max": list(bounds.GetMax()),
                          "static_geometry_diagnostic_only": True, "not_contact_or_full_collision_proof": True}
    except Exception as error:
        scene_geometry = {"diagnostic_unavailable": str(error)}

    def truth():
        origin = base.scene.env_origins[0]
        pos = np.stack([(base.scene[f"object_{i}"].data.root_pos_w[0]-origin).cpu().numpy() for i in (1, 2, 3)])
        quat = np.stack([base.scene[f"object_{i}"].data.root_quat_w[0].cpu().numpy() for i in (1, 2, 3)])
        vel = np.stack([base.scene[f"object_{i}"].data.root_lin_vel_w[0].cpu().numpy() for i in (1, 2, 3)])
        return pos.copy(), quat.copy(), vel.copy()

    def com_positions():
        origin = base.scene.env_origins[0]
        return np.stack([(base.scene[f"object_{i}"].data.root_com_pos_w[0]-origin).cpu().numpy() for i in (1, 2, 3)])

    class OracleTrainingPose:
        def observe(self, p, rgb):
            pos, quat, _ = truth()
            return VisualPose(pos, quat, np.ones(3))

    config = ThrowConfig(**motion_overrides, max_steps=args.max_steps, planner_seed=args.seed, object_order=tuple(i-1 for i in args.objects),
                         throw_primitive=args.throw_primitive)
    static_com = np.stack([base.scene[f"object_{i}"].data.body_com_pos_b[0, 0].cpu().numpy() for i in (1, 2, 3)])
    if args.oracle_training:
        # Calibration mode only. Final RGB inference must use frozen asset
        # constants, never read live object states to fill pose inputs.
        config = replace(config, object_com_offsets=tuple(tuple(float(v) for v in row) for row in static_com))
    solution = AlgSolution(perception=OracleTrainingPose() if args.oracle_training else None, config=config)
    config = solution.controller.cfg  # record the loaded drive model, not the instantaneous default
    solution.reset()
    evaluator = ThrowEvaluator(config, com_positions())
    baseline_after_warmup = False
    out.mkdir(parents=True)
    (out/"run_config.json").write_text(json.dumps({"seed": args.seed, "throw_config": asdict(config), "physics": physics_contract(config),
                                                "layout_seeded_at_construction": True,
                                                "configured_spawn_positions": {f"object_{i}":list(getattr(cfg.scene,f"object_{i}").init_state.pos) for i in (1,2,3)},
                                                "oracle_training_only": args.oracle_training,
                                                "per_object_profiles": solution.profile_metadata,
                                                "training_truth_never_policy_input": not args.oracle_training,
                                                "evaluation_only_scene_geometry": scene_geometry}, indent=2))
    if args.save_dataset:
        writer = PoseDatasetWriter(out/"pose_data.hdf5", config, args.seed,
                                   max_frames=(args.max_steps+args.data_stride-1)//args.data_stride)
    reason = "step_budget"; steps = 0; max_velocity = max_effort = 0.
    completion_observation_steps = 0
    with (out/"trace.jsonl").open("x") as log:
        for step in range(args.max_steps):
            with torch.inference_mode():
                action_stage = solution.controller.stage
                # Read training labels BEFORE the action. Never pair current RGB
                # with the next state's pose, or feed these labels to RGB policy.
                if writer is not None and step % args.data_stride == 0 and action_stage != "startup":
                    p, rgb = numpy_observation(obs)
                    xyz, quat, _ = truth()
                    writer.append(step, action_stage, p, rgb, xyz, quat)
                response = solution.predicts(obs, 0.)
                if action_stage == "startup" and solution.controller.stage == "home":
                    # Reset spawns can still be falling. Lift evidence must be
                    # relative to the settled tabletop state, not a mid-air reset.
                    evaluator = ThrowEvaluator(config, com_positions())
                    baseline_after_warmup = True
                if response["giveup"]:
                    if (solution.controller.stage != "predicted_complete"
                            or completion_observation_steps >= config.settle_steps+5):
                        reason = ("completion_observation_window_finished" if completion_observation_steps
                                  else "controller_visual_stop")
                        break
                    # A visual stop can lead the independent settle count by a
                    # few frames. Hold the finished pose for a finite window,
                    # allowing evaluation without supplying truth to the policy.
                    completion_observation_steps += 1
                    action_stage = "completion_observation"
                action = torch.tensor(response["action"], device=args.device, dtype=torch.float32)
                obs, _, terminated, truncated, _ = env.step(action)
                steps = step+1
                if terminated.any() or truncated.any():
                    # No post-autoreset truth is accepted as an episode target.
                    reason = "environment_terminal_no_valid_final_truth"; break
                pos, quat, vel = truth()
                ee_xyz = (robot.data.body_pos_w[0, ee_id]-base.scene.env_origins[0]).cpu().numpy()
                ee_vel = robot.data.body_link_lin_vel_w[0, ee_id].cpu().numpy()
                ee_omega = robot.data.body_ang_vel_w[0, ee_id].cpu().numpy()
                jaws = robot.data.joint_pos[0, 6:8].cpu().numpy()
                com_xyz = com_positions()
                commanded = np.asarray(response["action"][0])*config.action_scale+DEFAULT_Q
                opening = abs(commanded[6]-commanded[7]) >= config.open_gap_min
                success = evaluator.update(com_xyz, vel, ee_xyz, ee_vel, jaws, root_positions=pos,
                                           ee_angular_velocity=ee_omega, opening_command=opening)
                max_velocity = max(max_velocity, float(robot.data.joint_vel[0, :6].abs().max()))
                max_effort = max(max_effort, float(robot.data.applied_torque[0, :6].abs().max()))
                diag = solution.get_control_diagnostics()
                item = {"step": steps, "action_stage": action_stage,
                        "action": response["action"][0], "controller": diag,
                        "evaluation_only": {"object_xyz": pos.tolist(), "object_wxyz": quat.tolist(), "object_velocity": vel.tolist(),
                                            "ee_xyz": ee_xyz.tolist(), "ee_velocity": ee_vel.tolist(), "jaws": jaws.tolist(),
                                            "object_com_xyz": com_xyz.tolist(), "ee_angular_velocity": ee_omega.tolist(),
                                            "opening_command": bool(opening),
                                            "held_evidence": evaluator.held.tolist(), "upward_release": evaluator.released_up.tolist()}}
                log.write(json.dumps(item)+"\n")
                if steps % 250 == 0:
                    log.flush(); print(f"[TOSS] steps={steps} stage={diag['stage']} actual_success={success}", flush=True)
                if bool(evaluator.success[list(config.object_order)].all()):
                    reason = "requested_objects_throw_success"; break
    report = {**evaluator.report(), "seed": args.seed, "steps": steps,
              "termination_reason": reason, "oracle_training_only": args.oracle_training,
              "deployment_success_claim": bool(evaluator.success.all()) and not args.oracle_training,
              "original_task_e_success_claim": False,
              "requested_objects": args.objects,
              "requested_objects_success": bool(evaluator.success[list(config.object_order)].all()),
              "completion_observation_steps": completion_observation_steps,
              "max_actual_joint_speed": max_velocity, "max_actual_joint_effort": max_effort,
              "controller": solution.get_control_diagnostics()}
    report["per_object_profiles_enabled"] = bool(solution.profile_metadata)
    report["pose_dataset_frames"] = writer.count if writer is not None else 0
    report["initial_positions_evaluation_only"] = evaluator.initial.tolist()
    report["baseline_after_warmup"] = baseline_after_warmup
    (out/"result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    collection_complete = True
finally:
    if writer is not None:
        writer.close(complete=collection_complete)
    if env is not None:
        env.close()
    app.close()
