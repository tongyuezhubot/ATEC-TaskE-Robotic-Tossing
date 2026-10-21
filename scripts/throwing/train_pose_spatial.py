"""One finite spatial pose training run; split by complete episode seeds.

Freeze DINO, cache selected 16x16 patch grids, train a small RGB spatial head.
Only one final held-out calibration is performed; failed models remain candidates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--kinematic-prior", action="store_true")
    parser.add_argument("--held-focus", action="store_true", help="Oversample chosen held object and supervise COM at 5mm scale")
    parser.add_argument("--held-object", type=int, choices=(1,2,3), default=3)
    parser.add_argument("--dense-held-object",type=int,choices=(1,2,3),
                        help="Retain every held-stage frame for one active object; other caches remain compatible")
    parser.add_argument("--position-quality", action="store_true", help="Separate root/COM confidence from full pose confidence")
    parser.add_argument("--warm-start", help="Existing spatial checkpoint; new branch alone starts fresh")
    args = parser.parse_args()
    if args.position_quality and not args.held_focus:
        parser.error("Separate position quality currently requires held-focus supervision")
    if not 1 <= args.steps <= 10000 or not 1 <= args.stride <= 10:
        parser.error("Finite training/stride budgets required")
    out = Path(args.output_dir)
    if out.exists():
        parser.error("Choose a new training output directory")
    sources = json.loads(Path(args.manifest).read_text())
    paths = [Path(p).resolve() for group in ("train", "validation") for p in sources[group]]
    if len(set(paths)) != len(paths):
        parser.error("Duplicate sources or train/validation path overlap")
    import h5py
    import numpy as np
    import torch
    from torch.nn import functional as F
    from demo.throw_cem.config import ThrowConfig
    from demo.throw_cem.data import POSE_FORMAT, camera_contract, pose_calibration
    from demo.throw_cem.pose_model import load_encoder, patch_features
    from demo.throw_cem.pose_spatial import SpatialPoseHead, POSE_SPATIAL_FORMAT
    from scripts.throwing.pose_cache import cache_key,dense_episode,sample_indices
    torch.set_num_threads(4)
    torch.manual_seed(43)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera = camera_contract(ThrowConfig())
    seed_sets = []
    for group in ("train", "validation"):
        seeds = set()
        for filename in sources[group]:
            with h5py.File(filename, "r") as f:
                if (f.attrs.get("format") != POSE_FORMAT or not f.attrs.get("complete", False)
                        or json.loads(f.attrs["camera"]) != camera or not len(f["rgb"])):
                    raise ValueError("Complete same-camera pose data required")
                seeds.add(int(f.attrs["source_seed"]))
        seed_sets.append(seeds)
    if seed_sets[0] & seed_sets[1] or len(seed_sets[0]) < 2 or len(seed_sets[1]) < 2:
        raise ValueError("Use independent whole-episode seeds for training and calibration")
    out.mkdir(parents=True)
    (out/"sources.json").write_text(json.dumps(sources, indent=2))
    cache = Path(args.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    encoder = load_encoder(device)
    created_cache_paths = []
    (out/"created_cache_files.json").write_text("[]")

    def cached(path):
        path = Path(path)
        key = cache_key(path,args.stride,args.dense_held_object)
        destination = cache/f"{key}.pt"
        if destination.exists():
            return torch.load(destination, map_location="cpu", weights_only=True)
        with h5py.File(path, "r") as f:
            stages = [s.decode() for s in f["stage"][:]]
            idx = np.asarray(sample_indices(stages,args.stride,dense_episode(path,args.dense_held_object)))
            rgb = torch.from_numpy(f["rgb"][idx])
            patches = []
            with torch.inference_mode():
                for start in range(0, len(rgb), 16):
                    with torch.autocast(device_type=device.type, enabled=device.type=="cuda", dtype=torch.float16):
                        feature = patch_features(encoder, rgb[start:start+16].to(device))
                    patches.append(feature.half().cpu())
            item = {"patch": torch.cat(patches), "rgb": rgb,
                    "proprio": torch.from_numpy(f["proprio"][idx]),
                    "xyz": torch.from_numpy(f["object_xyz"][idx]),
                    "quat": torch.from_numpy(f["object_wxyz"][idx]),
                    "stage": [s.decode() for s in f["stage"][idx]]}
        torch.save(item, destination)
        created_cache_paths.append(str(destination.resolve()))
        (out/"created_cache_files.json").write_text(json.dumps(created_cache_paths,indent=2))
        print(f"[SPATIAL-POSE] cached {len(rgb)} frames: {path}", flush=True)
        return item

    def merge(group):
        items = [cached(p) for p in sources[group]]
        data = {k: torch.cat([item[k] for item in items]) for k in ("patch", "rgb", "proprio", "xyz", "quat")}
        data["stage"] = [s for item in items for s in item["stage"]]
        active = []
        for path, item in zip(sources[group], items):
            meta = json.loads((Path(path).parent/"run_config.json").read_text())
            obj = meta["throw_config"]["object_order"][0]
            active.extend([obj]*len(item["rgb"]))
        data["active_obj"] = torch.tensor(active)
        return data
    train, val = merge("train"), merge("validation")
    held_train = torch.tensor([s in ("check_hold", "calibration_wrist", "plan", "pre_throw")
                               for s in train["stage"]]) & (train["active_obj"]==args.held_object-1)
    held_indices = torch.where(held_train)[0]
    if args.held_focus and (not args.kinematic_prior or len(held_indices)<30):
        raise ValueError("Held-focus requires kinematic model and sufficient TRAIN held frames")
    del encoder
    if device.type=="cuda": torch.cuda.empty_cache()
    prop_mean = train["proprio"].mean(0); prop_std = train["proprio"].std(0).clamp_min(.01)
    table_idx = torch.tensor([s in ("home", "pre_grasp", "descend") for s in train["stage"]])
    table_z = train["xyz"][table_idx, :, 2].median(0).values
    target_quat = F.normalize(train["quat"], dim=-1)
    sign = torch.where((target_quat*target_quat[:1]).sum(-1, keepdim=True) >= 0, 1., -1.)
    quat_mean = F.normalize((target_quat*sign).mean(0), dim=-1)
    initial = None
    if args.warm_start:
        initial = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        # Keep the trained feature coordinate system, and declare exactly the
        # normalization buffers that load_state_dict will restore below.
        prop_mean = torch.tensor(initial["proprio_mean"])
        prop_std = torch.tensor(initial["proprio_std"])
        table_z = torch.tensor(initial["table_root_z"])
        quat_mean = torch.tensor(initial["quat_mean"])
    head = SpatialPoseHead(camera, prop_mean, prop_std, table_z, quat_mean,
                           kinematic_prior=args.kinematic_prior,position_quality=args.position_quality).to(device)
    if initial is not None:
        missing, unexpected = head.load_state_dict(initial["model"], strict=False)
        allowed = {"object_com_offsets", "pad_distance", "attachment_gate.weight", "attachment_gate.bias",
                   "attachment_residual.weight", "attachment_residual.bias",
                   "position_quality_head.weight", "position_quality_head.bias"}
        if unexpected or any(k not in allowed for k in missing):
            raise ValueError("Incompatible warm-start spatial model")
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4 if args.held_focus else 3e-4, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(43)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    def inputs(data, idx):
        return [data[k][idx].to(device) for k in ("patch", "rgb", "proprio")]
    for step in range(args.steps):
        idx = torch.randint(len(train["rgb"]), (args.batch_size,), generator=generator)
        if args.held_focus:
            half = args.batch_size//2
            idx[:half] = held_indices[torch.randint(len(held_indices), (half,), generator=generator)]
        target_xyz = train["xyz"][idx].to(device)
        target_quat = F.normalize(train["quat"][idx].to(device), dim=-1)
        with torch.autocast(device_type=device.type, enabled=device.type=="cuda", dtype=torch.float16):
            xyz, quat, quality, heatmap, uv = head(*inputs(train, idx))
            position_loss = F.smooth_l1_loss((xyz-target_xyz)/.03, torch.zeros_like(xyz))
            dot = (quat*target_quat).sum(-1).abs().clamp(0, 1)
            rotation_loss = (1-dot.square()).mean()
            localization_loss = head.localization_loss(heatmap, target_xyz)
            good = ((xyz.detach()-target_xyz).norm(dim=-1) <= .015) & (dot.detach() >= np.cos(np.deg2rad(6)))
            quality_loss = F.binary_cross_entropy_with_logits(quality, good.float())
            loss = position_loss+rotation_loss+.15*localization_loss+.1*quality_loss
            if args.held_focus:
                held = held_train[idx].to(device)
                # Rotation into the tool frame preserves Euclidean position
                # error; supervise the actual COM consumed by hold checks.
                def com_position(pos, orientation):
                    w,x,y,z = orientation.float().unbind(-1)
                    ox,oy,oz = orientation.new_tensor(ThrowConfig().object_com_offsets).unbind(-1)
                    rotated = torch.stack(((1-2*(y*y+z*z))*ox+2*(x*y-z*w)*oy+2*(x*z+y*w)*oz,
                                           2*(x*y+z*w)*ox+(1-2*(x*x+z*z))*oy+2*(y*z-x*w)*oz,
                                           2*(x*z-y*w)*ox+2*(y*z+x*w)*oy+(1-2*(x*x+y*y))*oz),-1)
                    return pos.float()+rotated
                com_error = com_position(xyz,quat)-com_position(target_xyz,target_quat)
                if held.any():
                    focus_error = com_error[held,args.held_object-1]
                    loss = loss+F.smooth_l1_loss(focus_error/.005, torch.zeros_like(focus_error))
                if args.position_quality:
                    position_good = ((xyz.detach()-target_xyz).norm(dim=-1)<=.015)&(com_error.detach().norm(dim=-1)<=.015)
                    loss = loss+.1*F.binary_cross_entropy_with_logits(head.position_quality_logits,position_good.float())
            if args.kinematic_prior:
                loss = loss+.05*head.attachment_loss(target_xyz, train["proprio"][idx].to(device))
        if not torch.isfinite(loss): raise ValueError("Non-finite spatial pose loss")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
        scaler.step(optimizer); scaler.update()
        if (step+1)%500==0:
            print(f"[SPATIAL-POSE] step={step+1} loss={loss.item():.5f} pos={position_loss.item():.5f} rot={rotation_loss.item():.5f} loc={localization_loss.item():.5f}", flush=True)
    head.eval(); predictions = [[], [], []]
    with torch.inference_mode():
        for start in range(0, len(val["rgb"]), args.batch_size):
            idx = torch.arange(start, min(start+args.batch_size, len(val["rgb"])))
            xyz, quat, quality, _, _ = head(*inputs(val, idx))
            for bucket, value in zip(predictions, (xyz, quat, quality.sigmoid())):
                bucket.append(value.cpu().numpy())
    xyz, quat, probability = [np.concatenate(bucket) for bucket in predictions]
    calibration = pose_calibration(xyz, quat, probability, val["xyz"].numpy(), val["quat"].numpy())
    if args.kinematic_prior:
        held = np.array([s in ("check_hold", "calibration_wrist", "plan", "pre_throw")
                         for s in val["stage"]]) & (val["active_obj"].numpy()==2)
        selected = held & (probability[:,2] >= calibration["confidence_thresholds"][2])
        error = np.linalg.norm(xyz[:,2]-val["xyz"][:,2].numpy(),axis=-1)
        dot = np.abs(np.sum(quat[:,2]*val["quat"][:,2].numpy(),axis=-1)).clip(0,1)
        good = (error<=.015) & (np.rad2deg(2*np.arccos(dot))<=12.)
        held_gate = {"frames":int(held.sum()),"accepted":int(selected.sum()),
                     "coverage":float(selected.sum()/max(held.sum(),1)),
                     "precision":float(good[selected].mean()) if selected.any() else 0.,
                     "position_rmse_m":float(np.sqrt(np.mean(error[held]**2))) if held.any() else None}
        held_gate["passed"] = (held_gate["accepted"]>=30 and held_gate["coverage"]>=.20 and held_gate["precision"]>=.95)
        calibration["banana_held_phase"] = held_gate
        calibration["passed"] = calibration["passed"] and held_gate["passed"]
    report = {**calibration, "train_frames": len(train["rgb"]), "val_frames": len(val["rgb"]),
              "train_seeds": sorted(seed_sets[0]), "val_seeds": sorted(seed_sets[1]),
              "steps": args.steps, "stride": args.stride,
              "held_focus": args.held_focus, "held_focus_object": args.held_object,
              "held_train_frames": int(held_train.sum()),
              "frozen_encoder": "DINOv2 ViT-S/14 16x16 patches", "head": "RGB56 heatmap + ray/depth",
              "simulator_success_claim": False}
    deployment_ready = bool(calibration["passed"] and not args.position_quality)
    report["requires_usage_calibration"] = args.position_quality
    payload = {"format": "atec_throw_rgb_pose_spatial_v6" if args.position_quality else ("atec_throw_rgb_pose_spatial_v4" if args.kinematic_prior else POSE_SPATIAL_FORMAT),
               "position_quality_head": args.position_quality,
               "kinematic_prior": args.kinematic_prior, "calibrated": deployment_ready,
               "cache_sampling":{"stride":args.stride,"dense_held_object":args.dense_held_object},
               "camera": camera, "original_image_size": [480, 640], "input_fields": ["rgb", "proprio"],
               "proprio_mean": prop_mean.tolist(), "proprio_std": prop_std.tolist(),
               "table_root_z": table_z.tolist(), "quat_mean": quat_mean.tolist(),
               "confidence_thresholds": calibration["confidence_thresholds"],
               "model": {k: v.cpu() for k,v in head.state_dict().items()}, "calibration": report}
    torch.save(payload, out/("rgb_pose_spatial.pt" if deployment_ready else "spatial_pose_candidate.pt"))
    (out/"calibration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return 0 if deployment_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
