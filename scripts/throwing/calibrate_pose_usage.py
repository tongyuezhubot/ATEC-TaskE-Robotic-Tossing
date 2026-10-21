"""Calibrate the quantities actually consumed by grasp acquisition and tracking.

Acquisition requires XYZ plus orientation. Once grasped, CEM consumes root/COM
position, not free-object orientation. Both profiles retain 15mm, >=95% precision,
>=20% coverage and >=30 accepted held-out frames. Weights are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--cache-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--held-tolerance",type=float,default=.015)
    parser.add_argument("--require-held-objects",type=int,nargs="+",choices=(1,2,3),default=[3])
    args=parser.parse_args()
    if not .001 <= args.held_tolerance <= .015:
        parser.error("Held tolerance may tighten, never exceed the existing 15mm gate")
    out=Path(args.output_dir)
    if out.exists(): parser.error("Choose a new calibration output directory")
    import numpy as np
    import torch
    from demo.throw_cem.pose_spatial import SpatialPoseHead
    from demo.throw_cem.data import pose_calibration
    from demo.throw_cem.config import ThrowConfig
    from demo.throw_cem.kinematics import quaternion_rotation
    from scripts.throwing.pose_cache import cache_key
    torch.set_num_threads(4)
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    head=SpatialPoseHead(payload["camera"],payload["proprio_mean"],payload["proprio_std"],
                         payload["table_root_z"],payload["quat_mean"],kinematic_prior=True,
                         position_quality=payload.get("position_quality_head",False)).cuda().eval()
    head.load_state_dict(payload["model"],strict=True)
    sources=json.loads(Path(args.manifest).read_text())
    sampling=payload.get("cache_sampling",{"stride":4,"dense_held_object":None})
    groups={k:[] for k in ("xyz","quat","prob","position_prob","target_xyz","target_quat","initial","held")}
    with torch.inference_mode():
        for name in sources["validation"]:
            path=Path(name)
            key=cache_key(path,sampling["stride"],sampling.get("dense_held_object"))
            data=torch.load(Path(args.cache_dir)/f"{key}.pt",map_location="cpu",weights_only=True)
            xyzs=[];quats=[];probs=[];position_probs=[]
            for start in range(0,len(data["rgb"]),32):
                inputs=[data[k][start:start+32].cuda() for k in ("patch","rgb","proprio")]
                xyz,quat,quality,_,_=head(*inputs)
                xyzs.append(xyz.cpu().numpy());quats.append(quat.cpu().numpy());probs.append(quality.sigmoid().cpu().numpy())
                position_quality = head.position_quality_logits if head.position_quality_logits is not None else quality
                position_probs.append(position_quality.sigmoid().cpu().numpy())
            groups["xyz"].append(np.concatenate(xyzs));groups["quat"].append(np.concatenate(quats));groups["prob"].append(np.concatenate(probs))
            groups["position_prob"].append(np.concatenate(position_probs))
            groups["target_xyz"].append(data["xyz"].numpy());groups["target_quat"].append(data["quat"].numpy())
            active=json.loads((path.parent/"run_config.json").read_text())["throw_config"]["object_order"][0]
            groups["initial"].append(np.array([s=="home" for s in data["stage"]]))
            groups["held"].append(np.array([[active==obj and s in ("check_hold","calibration_wrist","plan","pre_throw")
                                             for obj in range(3)] for s in data["stage"]]))
    data={k:np.concatenate(v) for k,v in groups.items()}
    m=data["initial"]
    initial=pose_calibration(data["xyz"][m],data["quat"][m],data["prob"][m],data["target_xyz"][m],data["target_quat"][m])
    offsets=np.asarray(ThrowConfig().object_com_offsets)
    root_error=np.linalg.norm(data["xyz"]-data["target_xyz"],axis=-1)
    com_error=np.empty_like(root_error)
    for i in range(len(root_error)):
        for obj in range(3):
            predicted=data["xyz"][i,obj]+quaternion_rotation(data["quat"][i,obj])@offsets[obj]
            target=data["target_xyz"][i,obj]+quaternion_rotation(data["target_quat"][i,obj])@offsets[obj]
            com_error[i,obj]=np.linalg.norm(predicted-target)
    good=(root_error<=.015)&(com_error<=.015)
    thresholds=[];tracking=[]
    for obj in range(3):
        chosen=None
        for threshold in np.linspace(.7,.99,60):
            selected=data["position_prob"][:,obj]>=threshold
            if selected.sum()>=30 and selected.mean()>=.20 and good[selected,obj].mean()>=.95:
                chosen=float(threshold);break
        thresholds.append(chosen if chosen is not None else 1.)
        selected=data["position_prob"][:,obj]>=thresholds[-1]
        tracking.append({"object":obj+1,"passed":chosen is not None,"threshold":thresholds[-1],
                         "coverage":float(selected.mean()),"precision":float(good[selected,obj].mean()) if chosen is not None else 0.,
                         "position_rmse_m":float(np.sqrt(np.mean(root_error[:,obj]**2)))})
    held_reports={}
    for object_id in sorted(set(args.require_held_objects)):
        obj=object_id-1;held=data["held"][:,obj];selected=held&(data["position_prob"][:,obj]>=thresholds[obj])
        held_good=(root_error[:,obj]<=args.held_tolerance)&(com_error[:,obj]<=args.held_tolerance)
        held_report={"frames":int(held.sum()),"accepted":int(selected.sum()),
                     "coverage":float(selected.sum()/max(held.sum(),1)),
                     "precision":float(held_good[selected].mean()) if selected.any() else 0.,
                     "root_com_tolerance_m":args.held_tolerance,
                     "position_rmse_m":float(np.sqrt(np.mean(root_error[held,obj]**2))) if held.any() else None}
        held_report["passed"]=held_report["accepted"]>=30 and held_report["coverage"]>=.20 and held_report["precision"]>=.95
        held_reports[str(object_id)]=held_report
    passed=initial["passed"] and all(r["passed"] for r in tracking) and all(r["passed"] for r in held_reports.values())
    report={"passed":bool(passed),"initial_acquisition":initial,"root_com_tracking":tracking,
            "banana_held_phase":held_reports.get("3"),"held_phase_by_object":held_reports,"weights_changed":False,
            "orientation_required_at_acquisition_degrees":12.,"root_and_com_tolerance_m":.015,
            "minimum_precision":.95,"minimum_coverage":.20,"simulator_success_claim":False}
    payload.update(format="atec_throw_rgb_pose_spatial_v6" if payload.get("position_quality_head",False) else "atec_throw_rgb_pose_spatial_v5",calibrated=bool(passed),
                   confidence_thresholds=initial["confidence_thresholds"],position_confidence_thresholds=thresholds,
                   held_calibrated_object_ids=sorted(set(args.require_held_objects)) if passed else [],
                   calibration=report)
    out.mkdir(parents=True)
    torch.save(payload,out/("rgb_pose_tracking.pt" if passed else "tracking_candidate.pt"))
    (out/"calibration.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    return 0 if passed else 2


if __name__=="__main__":raise SystemExit(main())
