"""Single-episode demo collection and success checking for Task E."""

import numpy as np
import torch
import torch.nn.functional as F
from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.utils import CartesianController
from atec_rl_lab.tasks.task_e.env_cfg import (
    TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z,
    BASKET_CENTER_X, BASKET_CENTER_Y, TABLE_HALF_X, TABLE_HALF_Y,
)

from .config import (
    ACTION_SCALE,
    GRIPPER_OPEN_POS, GRIPPER_CLOSE_POS,
    RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z,
    DEFAULT_PLACE_QUAT_W,
    BASKET_IN_X, BASKET_IN_Y,
    OBJ_SPAWN_X_MIN, OBJ_SPAWN_X_MAX, OBJ_SPAWN_Z, OBJ_SPAWN_Y_BANDS,
    OBJ_HALF_EXTENTS, OBJ_BBOX_MARGIN,
    GRASP_Z_OFFSET, GRASP_Z_OFFSETS, GRASP_LONG_AXIS_LOCAL,
    WARMUP_STEPS, SETTLE_STEPS, STEPS,
)
from .state_machine import PickPlaceStateMachine
from .global_ik import PiperGlobalIK


def _rerandomize_objects(env: ManagerBasedRLEnv, rng: np.random.Generator) -> None:
    """Place each object randomly with AABB-based overlap rejection."""
    placed: dict[int, tuple[float, float]] = {}  # obj_idx -> (x, y)

    for obj_idx in [1, 2, 3]:
        obj = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"]
        y_min, y_max = OBJ_SPAWN_Y_BANDS[obj_idx]
        hx, hy = OBJ_HALF_EXTENTS[obj_idx]

        x = y = None
        for _ in range(200):
            cx = float(rng.uniform(OBJ_SPAWN_X_MIN, OBJ_SPAWN_X_MAX))
            cy = float(rng.uniform(y_min, y_max))
            # AABB overlap check against all already-placed objects
            ok = all(
                abs(cx - px) >= hx + OBJ_HALF_EXTENTS[pi][0] + OBJ_BBOX_MARGIN or
                abs(cy - py) >= hy + OBJ_HALF_EXTENTS[pi][1] + OBJ_BBOX_MARGIN
                for pi, (px, py) in placed.items()
            )
            if ok:
                x, y = cx, cy
                break

        if x is None:  # fallback: band centre
            x = (OBJ_SPAWN_X_MIN + OBJ_SPAWN_X_MAX) / 2.0
            y = (y_min + y_max) / 2.0

        placed[obj_idx] = (x, y)
        state = obj.data.default_root_state[0:1].clone()
        state[0, 0] = x
        state[0, 1] = y
        state[0, 2] = OBJ_SPAWN_Z
        state[0, 7:] = 0.0   # zero velocities
        obj.write_root_state_to_sim(state)

    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()


_BASKET_MIN_Z = TABLE_TOP_Z
_BASKET_MAX_Z = TABLE_TOP_Z + 0.15

def check_objects_in_basket(env: ManagerBasedRLEnv, pick_objects: list[int]) -> bool:
    """Apply the evaluation bounds to every requested object in the live scene."""
    for obj_idx in pick_objects:
        pos = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"].data.root_pos_w[0]
        if (abs(pos[0].item() - BASKET_CENTER_X) > BASKET_IN_X or
                abs(pos[1].item() - BASKET_CENTER_Y) > BASKET_IN_Y or
                pos[2].item() < _BASKET_MIN_Z or
                pos[2].item() > _BASKET_MAX_Z):
            return False
    return True


def _active_termination_reasons(env: ManagerBasedRLEnv, info: dict | None = None) -> list[str]:
    """Return terms that ended env 0, including terms retained across auto-reset."""
    manager = env.unwrapped.termination_manager
    reasons = {
        name
        for name, values in manager.get_active_iterable_terms(0)
        if any(bool(value) for value in values)
    }
    if isinstance(info, dict):
        episode_log = info.get("log", {})
        prefix = "Episode_Termination/"
        reasons.update(
            key.removeprefix(prefix)
            for key, value in episode_log.items()
            if key.startswith(prefix) and bool(value)
        )
    return sorted(reasons)


def _build_result(
    qpos_buf,
    qvel_buf,
    ee_pos_buf,
    ee_quat_buf,
    action_buf,
    frames_buf,
    *,
    success: bool,
    termination_reason: str,
    pick_objects: list[int],
    object_diagnostics: dict | None = None,
    aligned_records: list | None = None,
) -> dict:
    result = {
        "qpos": np.stack(qpos_buf),
        "qvel": np.stack(qvel_buf),
        "ee_pos": np.stack(ee_pos_buf),
        "ee_quat": np.stack(ee_quat_buf),
        "action": np.stack(action_buf),
        "success": bool(success),
        "termination_reason": termination_reason,
        "pick_objects": list(pick_objects),
        "object_diagnostics": object_diagnostics or {},
        "aligned_records": aligned_records or [],
    }
    if frames_buf is not None:
        result["frames"] = np.stack(frames_buf)
    return result


def collect_one_demo(
    env:         ManagerBasedRLEnv,
    robot,
    ik_ctrl:     CartesianController,
    arm_ids:     list[int],
    gripper_ids: list[int],
    pick_objects: list[int],
    device:      str,
    default_jpos: torch.Tensor,
    rng:         np.random.Generator,
    camera=None,
    frame_size: int | None = None,
    aligned: bool = False,
    recovery_scenario=None,
    max_record_steps: int = 6000,
) -> dict | None:
    """Run one full episode and return data, preserving successful termination.

    Returns a dict with keys:
      qpos    (T, 8)        absolute joint positions
      qvel    (T, 8)        joint velocities
      ee_pos  (T, 3)        end-effector position (world frame)
      ee_quat (T, 4)        end-effector quaternion (w,x,y,z)
      action  (T, 8)        env action = (joint_target - default_jpos) / ACTION_SCALE
      frames  (T, H, W, 3)  RGB uint8 — only present when camera is given
    """
    if recovery_scenario is not None and not aligned:
        raise ValueError("Recovery data require aligned transitions")
    env.reset()
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos,
        torch.zeros_like(robot.data.default_joint_vel),
    )

    _rerandomize_objects(env, rng)      # write new object positions to sim + sim.forward()
    default_jpos = robot.data.default_joint_pos.clone()

    ee_home = torch.tensor([[RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z]],
                            dtype=torch.float32, device=device)
    eq_home = torch.tensor([DEFAULT_PLACE_QUAT_W], dtype=torch.float32, device=device)
    g_open  = torch.tensor([GRIPPER_OPEN_POS], dtype=torch.float32, device=device)

    robot.update(dt=env.unwrapped.physics_dt)
    ik_ctrl.reset()

    # Warm-up: drive arm to HOME position (not recorded)
    for _ in range(WARMUP_STEPS):
        terminated, truncated, info = _step_to(
            env, robot, ik_ctrl, arm_ids, gripper_ids,
            ee_home, eq_home, g_open, default_jpos,
        )
        if terminated.any() or truncated.any():
            reasons = _active_termination_reasons(env, info)
            print(f"[WARN] Episode ended during warm-up: {reasons}")
            return None

    # Pre-compute grasp quaternions from actual object orientations after reset
    sm = PickPlaceStateMachine(pick_objects, device)
    global_ik = PiperGlobalIK()
    for obj_idx in pick_objects:
        obj_quat = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"] \
                       .data.root_state_w[0, 3:7]
        sm.set_grasp_quat(obj_idx, obj_quat)

    # Settle
    for _ in range(SETTLE_STEPS):
        terminated, truncated, info = _step_to(
            env, robot, ik_ctrl, arm_ids, gripper_ids,
            ee_home, eq_home, g_open, default_jpos,
        )
        if terminated.any() or truncated.any():
            reasons = _active_termination_reasons(env, info)
            print(f"[WARN] Episode ended while settling: {reasons}")
            return None

    ik_ctrl.reset()

    # ---- Recording loop ---- #
    qpos_buf, qvel_buf, ee_pos_buf, ee_quat_buf, action_buf = [], [], [], [], []
    frames_buf = [] if camera is not None else None
    aligned_records = []
    recovery = None
    if recovery_scenario is not None:
        from .recovery import RecoveryController
        params = env.unwrapped.termination_manager.get_term_cfg("basket_success").params
        recovery = RecoveryController(recovery_scenario, pick_objects, params,
            (TABLE_CENTER_X,TABLE_CENTER_Y),(TABLE_HALF_X,TABLE_HALF_Y))

    def finalize(data):
        if recovery is not None:
            data["recovery_report"] = recovery.finalize(aligned_records)
            data["actor_supervision"] = True  # per-step verified actor_valid owns masking
        return data

    object_diagnostics = {}
    global_joint_goal = None
    global_goal_state = None
    for obj_idx in pick_objects:
        pos = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"].data.root_pos_w[0]
        object_diagnostics[str(obj_idx)] = {
            "initial_pos": [float(value) for value in pos],
            "max_z": float(pos[2]),
            "grasp_z_offset": GRASP_Z_OFFSETS.get(obj_idx, GRASP_Z_OFFSET),
            "grasp_long_axis_local": GRASP_LONG_AXIS_LOCAL.get(obj_idx),
            "stage_final_pos": {},
            "stage_final_ee_pos": {},
            "stage_final_ee_quat": {},
            "stage_ee_object_distance": {},
            "stage_ee_orientation_error_deg": {},
            "stage_gripper_joint_pos": {},
            "stage_target_ee_pos": {},
            "stage_target_ee_quat": {},
        }

    while not sm.done and len(action_buf) < max_record_steps:
        state_before_step = sm.state
        active_obj_idx = sm.current_object_index
        state_count = sm.step_count
        attempt_idx = recovery.retries[active_obj_idx] if recovery is not None else 0
        if recovery is not None and state_before_step == "PRE_GRASP" and state_count == 0:
            sm.set_grasp_quat(active_obj_idx, env.unwrapped.scene[sm.current_object_key].data.root_quat_w[0])
        obj_pos_w = env.unwrapped.scene.rigid_objects[sm.current_object_key] \
                        .data.root_pos_w[0].clone()
        ee_pos_des, ee_quat_des, gripper_cmd = sm.tick(obj_pos_w)
        intervention = False
        if recovery is not None:
            origin = env.unwrapped.scene.env_origins[0]
            positions = np.stack([(env.unwrapped.scene[f"object_{i}"].data.root_pos_w[0]-origin)
                                  .detach().cpu().numpy() for i in (1,2,3)])
            recovery.record_index = len(action_buf)
            adjusted, gripper_cmd, intervention = recovery.adjust(
                active_obj_idx, state_before_step, state_count,
                (ee_pos_des-origin).detach().cpu().numpy(), gripper_cmd, positions)
            ee_pos_des = torch.as_tensor(adjusted,device=device)+origin

        # Tall objects use globally solved waypoints around the grasp.  Sugar
        # approaches from above so an open finger cannot push its end face.
        use_global_ik = (
            active_obj_idx == 1
            and state_before_step in {"ALIGN", "REACH", "DESCEND", "CLOSE"}
        ) or (
            active_obj_idx == 2
            and state_before_step in {
                "ALIGN", "REACH", "DESCEND", "CLOSE", "LIFT", "TRANSPORT"
            }
        )
        goal_key = (active_obj_idx, state_before_step, attempt_idx) if recovery is not None else state_before_step
        if goal_key != global_goal_state:
            global_goal_state = goal_key
            global_joint_goal = None
            if use_global_ik:
                result = global_ik.solve(
                    ee_pos_des.detach().cpu().numpy(),
                    ee_quat_des.detach().cpu().numpy(),
                    robot.data.root_pose_w[0].detach().cpu().numpy(),
                    robot.data.joint_pos[0, arm_ids].detach().cpu().numpy(),
                )
                usable = result.usable or (
                    active_obj_idx == 1
                    and state_before_step == "ALIGN"
                    and result.position_error <= 0.015
                    and result.orientation_error <= np.deg2rad(70.0)
                ) or (
                    active_obj_idx == 1
                    and state_before_step == "REACH"
                    and result.position_error <= 0.015
                    and result.orientation_error <= np.deg2rad(70.0)
                ) or (
                    active_obj_idx == 1
                    and state_before_step in {"DESCEND", "CLOSE"}
                    and result.position_error <= 0.005
                    and result.orientation_error <= np.deg2rad(50.0)
                ) or (
                    active_obj_idx == 2
                    and state_before_step in {"REACH", "DESCEND", "CLOSE"}
                    and result.position_error <= 0.005
                    and result.orientation_error <= np.deg2rad(40.0)
                ) or (
                    active_obj_idx == 2
                    and state_before_step in {"LIFT", "TRANSPORT"}
                    and result.position_error <= 0.015
                    and result.orientation_error <= np.deg2rad(60.0)
                )
                print(
                    f"[GLOBAL_IK] object={active_obj_idx} state={state_before_step} "
                    f"pos_err={result.position_error * 1000:.2f}mm "
                    f"rot_err={np.degrees(result.orientation_error):.2f}deg "
                    f"usable={usable} nfev={result.evaluations}"
                )
                if usable:
                    global_joint_goal = torch.as_tensor(
                        result.joint_pos, dtype=torch.float32, device=device
                    ).unsqueeze(0)

        if global_joint_goal is not None:
            arm_jpos_cur = robot.data.joint_pos[:, arm_ids]
            joint_delta = torch.clamp(global_joint_goal - arm_jpos_cur, -0.1, 0.1)
            arm_jpos_des = arm_jpos_cur + joint_delta
        else:
            arm_jpos_des = ik_ctrl.compute(ee_pos_des.unsqueeze(0), ee_quat_des.unsqueeze(0))
        gripper_vals   = GRIPPER_OPEN_POS if gripper_cmd == "open" else GRIPPER_CLOSE_POS
        gripper_target = torch.tensor([gripper_vals], dtype=torch.float32, device=device)

        full_target = robot.data.joint_pos.clone()
        full_target[:, arm_ids]     = arm_jpos_des
        full_target[:, gripper_ids] = gripper_target
        env_action = (full_target - default_jpos) / ACTION_SCALE

        # Record BEFORE stepping (obs at time t, action at time t)
        qpos_buf.append(robot.data.joint_pos[0].cpu().numpy())
        qvel_buf.append(robot.data.joint_vel[0].cpu().numpy())
        ee_pos_buf.append(ik_ctrl.ee_pos_w[0].cpu().numpy())
        ee_quat_buf.append(ik_ctrl.ee_quat_w[0].cpu().numpy())
        action_buf.append(env_action[0].cpu().numpy())
        if frames_buf is not None:
            rgb = camera.data.output["rgb"][0, :, :, :3]
            if frame_size is not None:
                if frame_size < 16:
                    raise ValueError("frame_size must be at least 16 pixels.")
                rgb = F.interpolate(
                    rgb.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(frame_size, frame_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
            frames_buf.append(rgb.cpu().numpy())

        if aligned:
            from scripts.world_model.aligned_transitions import aligned_step
            step_result, record = aligned_step(
                env, env_action, frame_size or 96, state_before_step
            )
            if recovery is not None:
                recovery.annotate(record, active_obj_idx, attempt_idx, intervention)
            aligned_records.append(record)
            _, _, terminated, truncated, info = step_result
        else:
            _, _, terminated, truncated, info = env.step(env_action)

        for obj_idx in ([] if terminated.any() or truncated.any() else pick_objects):
            pos = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"].data.root_pos_w[0]
            diag = object_diagnostics[str(obj_idx)]
            diag["max_z"] = max(diag["max_z"], float(pos[2]))
            if obj_idx == active_obj_idx:
                ee_pos = ik_ctrl.ee_pos_w[0]
                diag["stage_final_pos"][state_before_step] = [float(value) for value in pos]
                diag["stage_final_ee_pos"][state_before_step] = [
                    float(value) for value in ee_pos
                ]
                ee_quat = ik_ctrl.ee_quat_w[0]
                diag["stage_final_ee_quat"][state_before_step] = [
                    float(value) for value in ee_quat
                ]
                diag["stage_ee_object_distance"][state_before_step] = float(
                    torch.linalg.vector_norm(ee_pos - pos)
                )
                quat_cos = torch.abs(torch.dot(
                    F.normalize(ee_quat, dim=0),
                    F.normalize(ee_quat_des, dim=0),
                )).clamp(max=1.0)
                diag["stage_ee_orientation_error_deg"][state_before_step] = float(
                    torch.rad2deg(2.0 * torch.acos(quat_cos))
                )
                diag["stage_gripper_joint_pos"][state_before_step] = [
                    float(value) for value in robot.data.joint_pos[0, gripper_ids]
                ]
                diag["stage_target_ee_pos"][state_before_step] = [
                    float(value) for value in ee_pos_des
                ]
                diag["stage_target_ee_quat"][state_before_step] = [
                    float(value) for value in ee_quat_des
                ]

        if terminated.any() or truncated.any():
            reasons = _active_termination_reasons(env, info)
            if aligned:
                success = bool(aligned_records[-1]["training_labels"]["basket_success"])
                return finalize(_build_result(
                    qpos_buf, qvel_buf, ee_pos_buf, ee_quat_buf, action_buf, frames_buf,
                    success=success, termination_reason="+".join(reasons) or "unknown",
                    pick_objects=pick_objects, object_diagnostics=object_diagnostics,
                    aligned_records=aligned_records,
                ))
            if "basket_success" not in reasons:
                print(f"[WARN] Episode ended before success: {reasons}")
                return None
            return _build_result(
                qpos_buf, qvel_buf, ee_pos_buf, ee_quat_buf, action_buf, frames_buf,
                success=True,
                termination_reason="basket_success",
                pick_objects=pick_objects,
                object_diagnostics=object_diagnostics,
            )

        if recovery is not None:
            positions = record["training_labels"]["object_pos"]
            ee_local = (ik_ctrl.ee_pos_w[0]-env.unwrapped.scene.env_origins[0]).detach().cpu().numpy()
            finished = state_count+1 == STEPS[state_before_step]
            decision = recovery.check(active_obj_idx, state_before_step, finished, positions, ee_local)
            if decision == "retry":
                sm.retry_object(active_obj_idx)
                ik_ctrl.reset()
                global_joint_goal = global_goal_state = None
            elif decision == "skip":
                sm.skip_object(active_obj_idx)
                ik_ctrl.reset()
                global_joint_goal = global_goal_state = None
        if len(action_buf) % 500 == 0:
            print(f"[COLLECTION_PROGRESS] steps={len(action_buf)} object={active_obj_idx} stage={state_before_step}", flush=True)

    if aligned_records and not aligned_records[-1]["terminated"]:
        aligned_records[-1]["truncated"] = True  # expert/budget stop is not a task terminal
    success = check_objects_in_basket(env, pick_objects)
    for obj_idx in pick_objects:
        pos = env.unwrapped.scene.rigid_objects[f"object_{obj_idx}"].data.root_pos_w[0]
        object_diagnostics[str(obj_idx)]["final_pos"] = [float(value) for value in pos]
        object_diagnostics[str(obj_idx)]["inside_basket"] = bool(
            abs(float(pos[0]) - BASKET_CENTER_X) <= BASKET_IN_X
            and abs(float(pos[1]) - BASKET_CENTER_Y) <= BASKET_IN_Y
            and _BASKET_MIN_Z <= float(pos[2]) <= _BASKET_MAX_Z
        )
    return finalize(_build_result(
        qpos_buf, qvel_buf, ee_pos_buf, ee_quat_buf, action_buf, frames_buf,
        success=success,
        termination_reason="collector_budget" if not sm.done else "state_machine_complete",
        aligned_records=aligned_records,
        pick_objects=pick_objects,
        object_diagnostics=object_diagnostics,
    ))


# ------------------------------------------------------------------ #
# Internal helper
# ------------------------------------------------------------------ #

def _step_to(env, robot, ik_ctrl, arm_ids, gripper_ids,
             ee_pos, ee_quat, gripper_target, default_jpos):
    """Single IK step toward a target pose (utility used during warm-up/settle)."""
    arm_des = ik_ctrl.compute(ee_pos, ee_quat)
    tgt = robot.data.joint_pos.clone()
    tgt[:, arm_ids]     = arm_des
    tgt[:, gripper_ids] = gripper_target
    _, _, terminated, truncated, info = env.step((tgt - default_jpos) / ACTION_SCALE)
    robot.update(dt=env.unwrapped.physics_dt)
    return terminated, truncated, info
