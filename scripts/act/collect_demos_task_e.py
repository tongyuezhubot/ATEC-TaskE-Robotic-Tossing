"""Scripted oracle data collection for Task E (pick-and-place).

Usage
-----
# Per-object grasp geometry is configured in task_e/config.py.
python scripts/act/collect_demos_task_e.py --pick_objects 3 --num_demos 50 --headless

"""

import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# sys.path.insert(0, os.path.dirname(__file__))

from isaaclab.app import AppLauncher
from cli_args import add_collect_demo_args

parser = argparse.ArgumentParser(description="Collect Task E demonstrations for ACT.")
add_collect_demo_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.native_physics and not args_cli.aligned:
    parser.error("--native_physics requires --aligned")
if args_cli.recovery_mix and (not args_cli.aligned or args_cli.only_success):
    parser.error("--recovery_mix requires --aligned and must keep failures")
if args_cli.recovery_mix and (not 1000 <= args_cli.max_record_steps <= 6000 or not 1 <= args_cli.num_demos <= 6):
    parser.error("Recovery budget requires 1..6 demos and 1000..6000 steps/episode")
if args_cli.recovery_mix and set(args_cli.pick_objects) != {1,2,3}:
    parser.error("Recovery mix retains all three original Task-E objects")

if args_cli.save_video or args_cli.save_images or args_cli.aligned:
    args_cli.enable_cameras = True

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import json
import numpy as np

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

from atec_rl_lab.tasks.task_e.env_cfg import TaskEEnvPiperCfg
from atec_rl_lab.utils import CartesianController

from task_e.config import (
    EE_BODY_NAME, ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES,
    ACT_STIFFNESS, ACT_DAMPING, ACT_EFFORT_LIMIT, ACT_VEL_LIMIT,
    CAM_H, CAM_W, CAM_POS, CAM_ROT,
)
from task_e.collector import collect_one_demo
from task_e.recovery import SCENARIOS


def build_env(pick_objects: list[int], need_camera: bool) -> ManagerBasedRLEnv:
    import time
    cfg = TaskEEnvPiperCfg()
    cfg.seed              = args_cli.seed
    cfg.scene.num_envs    = 1
    cfg.episode_length_s  = 40.0 * len(pick_objects) + 10.0
    if not (args_cli.recovery_mix or args_cli.native_physics):
        cfg.scene.robot.actuators["default"] = ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=ACT_EFFORT_LIMIT,
            velocity_limit=ACT_VEL_LIMIT,
            stiffness=ACT_STIFFNESS,
            damping=ACT_DAMPING,
        )
    else:
        actuator = cfg.scene.robot.actuators["default"]
        print(f"[COLLECTION] native actuator stiffness={actuator.stiffness} damping={actuator.damping}", flush=True)
    # if need_camera:
    #     cfg.scene.video_cam = CameraCfg(
    #         prim_path="{ENV_REGEX_NS}/video_cam",
    #         update_period=0.0,
    #         height=CAM_H, width=CAM_W,
    #         data_types=["rgb"],
    #         spawn=sim_utils.PinholeCameraCfg(
    #             focal_length=24.0, focus_distance=400.0,
    #             horizontal_aperture=20.955, clipping_range=(0.1, 100.0),
    #         ),
    #         offset=CameraCfg.OffsetCfg(pos=CAM_POS, rot=CAM_ROT, convention="world"),
    #     )
    return ManagerBasedRLEnv(cfg)


def init_output(output_dir: str) -> tuple[str, str]:
    """Create a new trajectory file without overwriting existing data."""
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, "trajectory.hdf5")
    json_path = os.path.join(output_dir, "trajectory.json")
    if os.path.exists(traj_path):
        raise FileExistsError(f"{traj_path} exists; use a new output directory.")
    with h5py.File(traj_path, "x"):
        pass
    with open(json_path, "w") as fh:
        json.dump({"env_info": {"env_kwargs": {"control_mode": "pd_joint_pos"}}}, fh)
    return traj_path, json_path


def save_traj(traj_path: str, traj_idx: int, data: dict,
              save_images: bool) -> None:
    """Append one trajectory group to the consolidated HDF5."""
    with h5py.File(traj_path, "a") as f:
        grp = f.create_group(f"traj_{traj_idx}")
        grp.create_dataset("obs",     data=data["qpos"],    compression="gzip")
        grp.create_dataset("actions", data=data["action"],  compression="gzip")
        grp.create_dataset("qvel",    data=data["qvel"],    compression="gzip")
        grp.create_dataset("ee_pos",  data=data["ee_pos"],  compression="gzip")
        grp.create_dataset("ee_quat", data=data["ee_quat"], compression="gzip")
        grp.attrs["success"] = bool(data["success"])
        grp.attrs["actor_supervision"] = bool(data.get("actor_supervision", data["success"]))
        if "recovery_report" in data:
            grp.attrs["requires_actor_valid_mask"] = True
            grp.attrs["recovery_report"] = json.dumps(data["recovery_report"])
            grp.attrs["collection_physics"] = "native_task"
        grp.attrs["termination_reason"] = data["termination_reason"]
        grp.attrs["pick_objects"] = data["pick_objects"]
        grp.attrs["collection_physics"] = "native_task" if (args_cli.native_physics or args_cli.recovery_mix) else "expert_override"
        grp.attrs["collection_seed"] = args_cli.seed
        if data.get("aligned_records"):
            from scripts.world_model.aligned_transitions import save_aligned_group
            save_aligned_group(grp, data["aligned_records"])
        if data.get("aligned_records"):
            grp.create_group("images")["rgb"] = grp["policy_input/rgb"]
        elif save_images and "frames" in data:
            frames = data["frames"]
            grp.create_group("images").create_dataset(
                "rgb",
                data=frames,
                compression="gzip",
                # Training samples individual frames in random order.  HDF5's
                # automatic chunk shape spans hundreds of timesteps here,
                # making each sample decompress far more data than it needs.
                chunks=(1, *frames.shape[1:]),
            )


def main() -> None:
    # Preserve the requested order.  For the full task, picking from the
    # basket-nearest band outward (3, 2, 1) prevents a carried far-side object
    # from sweeping through objects that are still waiting on the table.
    pick_objects = list(dict.fromkeys(args_cli.pick_objects))
    need_camera = args_cli.save_video or args_cli.save_images or args_cli.aligned

    env    = build_env(pick_objects, need_camera)
    dev    = env.unwrapped.device
    camera = env.unwrapped.scene["video_cam"] if need_camera else None

    robot = env.unwrapped.scene.articulations["robot"]
    arm_ids,     _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    ik_ctrl = CartesianController(
        robot=robot, ee_body_name=EE_BODY_NAME,
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1, device=dev,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.2,
    )
    default_jpos = robot.data.default_joint_pos.clone()

    video_dir = None
    imageio   = None
    if args_cli.save_video:
        video_dir = args_cli.video_dir or os.path.join(args_cli.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        import imageio as _io
        imageio = _io

    traj_path, _ = init_output(args_cli.output_dir)
    rng = np.random.default_rng(args_cli.seed)

    n_ok = 0
    attempt = 0
    max_attempts = args_cli.max_attempts or max(args_cli.num_demos * 10, 1)
    if max_attempts < args_cli.num_demos:
        raise ValueError("--max_attempts must be at least --num_demos (or zero for automatic).")

    while n_ok < args_cli.num_demos and attempt < max_attempts:
        attempt += 1
        print(f"\n[INFO] Demo {n_ok + 1}/{args_cli.num_demos}  (attempt {attempt})", flush=True)

        data = collect_one_demo(
            env, robot, ik_ctrl,
            arm_ids, gripper_ids,
            pick_objects, dev,
            default_jpos=default_jpos,
            rng=rng,
            camera=camera,
            frame_size=args_cli.image_size or None,
            aligned=args_cli.aligned,
            recovery_scenario=SCENARIOS[n_ok % len(SCENARIOS)] if args_cli.recovery_mix else None,
            max_record_steps=args_cli.max_record_steps,
        )
        if data is None:
            print("[WARN] Early termination — skipping.")
            continue

        if args_cli.recovery_mix:
            print("[RECOVERY_REPORT] " + json.dumps(data["recovery_report"]), flush=True)
        print("[DIAG] Object motion:")
        print(json.dumps(data.get("object_diagnostics", {}), indent=2))

        if args_cli.only_success and not data["success"]:
            print("[WARN] Objects not in basket — skipping (--only_success).")
            continue

        save_traj(traj_path, n_ok, data, args_cli.save_images)

        T     = len(data["qpos"])
        notes = [
            f"{T} steps",
            f"success={data['success']}",
            f"termination={data['termination_reason']}",
        ]
        if args_cli.save_video and "frames" in data:
            vp = os.path.join(video_dir, f"demo_{n_ok:04d}.mp4")
            imageio.mimwrite(vp, data["frames"], fps=50, quality=7)
            notes.append(f"video → {vp}")
        if args_cli.save_images and "frames" in data:
            notes.append("images saved")
        print(f"[INFO] traj_{n_ok}: {', '.join(notes)}")
        n_ok += 1

    if n_ok < args_cli.num_demos:
        raise RuntimeError(
            f"Collected {n_ok}/{args_cli.num_demos} demos after {attempt} attempts. "
            "Inspect the failure logs before increasing --max_attempts."
        )

    print(f"\n[INFO] Collected {n_ok} demos → {traj_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
