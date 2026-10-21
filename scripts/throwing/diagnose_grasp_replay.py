"""Read-only controller replay on recorded standard RGB/proprio, no simulator.

Privileged labels are used only in the final diagnostic report, never replay
inputs. Requires contiguous frames and checks every recorded phase transition.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--perception-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import h5py
    import numpy as np
    import torch
    from demo.throw_cem.config import ThrowConfig, DEFAULT_Q
    from demo.throw_cem.controller import ThrowController
    from demo.throw_cem.perception import RGBPosePerception
    path = Path(args.episode)
    output = Path(args.output)
    if output.exists():
        parser.error("Choose a new diagnostic output; existing reports are preserved")
    meta = json.loads((path/"run_config.json").read_text())
    cfg = ThrowConfig(**meta["throw_config"])
    controller = ThrowController(cfg, RGBPosePerception(args.perception_model, cfg))
    torch.set_num_threads(4)
    mismatches = []
    last_motion_goal = None
    with h5py.File(path/"pose_data.hdf5") as data, torch.inference_mode():
        steps = data["step"][:]
        if not len(steps) or np.any(np.diff(steps) != 1) or data["stage"][0].decode() != "home":
            raise ValueError("Replay needs contiguous frames starting at home")
        controller.step = int(steps[0]); controller.stage = "home"
        controller.target = data["proprio"][0,:8]+DEFAULT_Q
        for i in range(len(steps)):
            expected = data["stage"][i].decode()
            if controller.stage != expected:
                mismatches.append({"step":int(steps[i]), "expected":expected,
                                   "replayed":controller.stage})
            p = data["proprio"][i]
            if controller.goal is not None:
                last_motion_goal = controller.goal.copy()
            controller.action(p, data["rgb"][i])
            if controller.goal is not None:
                last_motion_goal = controller.goal.copy()
            if controller.stage == "aborted":
                break
        q = p[:8]+DEFAULT_Q
        obj = cfg.object_order[0]
        report = {"source":str(path.resolve()), "seed":meta["seed"],
                  "simulator_used":False, "truth_fed_to_controller":False,
                  "stage_mismatches":mismatches[:5], "replay_valid":not mismatches,
                  "last_frame":int(steps[i]), "diagnostics":controller.diagnostics()}
        if hasattr(controller, "grasp_xyz"):
            actual = controller.kin.forward(q)
            goal_q = q.copy(); goal_q[:6] = last_motion_goal if last_motion_goal is not None else q[:6]
            goal = controller.kin.forward(goal_q)
            report.update(grasp_target_xyz=controller.grasp_xyz.tolist(),
                          grasp_tool_z=controller.rotation[:,2].tolist(),
                          actual_ee_error_m=(actual[:3,3]-controller.grasp_xyz).tolist(),
                          goal_fk_error_m=(goal[:3,3]-controller.grasp_xyz).tolist(),
                          joint_tracking_gap_rad=(goal_q[:6]-q[:6]).tolist(),
                          final_object_xyz_evaluation_only=data["object_xyz"][i,obj].tolist(),
                          last_25_frame_max_joint_change_rad=float(np.max(np.ptp(data["proprio"][max(0,i-24):i+1,:6],axis=0))))
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
