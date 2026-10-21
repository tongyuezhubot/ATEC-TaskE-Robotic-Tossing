"""Frozen-DINO RGB pose training; whole-seed holdout, finite head-only training.

Requires existing server Torch/h5py/Pillow and the existing local DINO bundle.
No ACT, downloads, or simulator startup. Never invent orientation labels for old
pick/place datasets that lack object_wxyz. Failed quality gates save a candidate
only; they never emit a deployable checkpoint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-files", nargs="+", required=True)
    parser.add_argument("--val-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", help="Reusable frozen feature cache; source files are never modified")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000 or not 1 <= args.batch_size <= 512 or not 1 <= args.feature_batch_size <= 64:
        parser.error("Use finite training and batch budgets")
    out = Path(args.output_dir)
    if out.exists():
        parser.error("Choose a NEW output directory")
    train_paths = [Path(x).resolve() for x in args.train_files]
    val_paths = [Path(x).resolve() for x in args.val_files]
    if len(set(train_paths+val_paths)) != len(train_paths+val_paths):
        parser.error("Duplicate files / train-validation overlap")

    import h5py
    import numpy as np
    import torch
    from torch.nn import functional as F
    from demo.throw_cem.config import ThrowConfig
    from demo.throw_cem.data import POSE_FORMAT, camera_contract, pose_calibration
    from demo.throw_cem.pose_model import load_encoder, image_features, make_head, decode

    def source_info(path):
        with h5py.File(path, "r") as f:
            if f.attrs.get("format") != POSE_FORMAT or not f.attrs.get("complete", False):
                raise ValueError(f"Not a complete time-aligned pose dataset: {path}")
            if json.loads(f.attrs["camera"]) != camera_contract(ThrowConfig()):
                raise ValueError("Camera contract mismatch")
            if len(f["rgb"]) < 1:
                raise ValueError("Empty pose dataset")
            if f["object_wxyz"].shape != (len(f["rgb"]), 3, 4):
                raise ValueError("Complete real quaternion labels required")
            return int(f.attrs["source_seed"])

    train_seeds = {source_info(p) for p in train_paths}
    val_seeds = {source_info(p) for p in val_paths}
    if len(train_seeds) < 2 or train_seeds & val_seeds:
        raise ValueError("Use >=2 training seeds and >=1 different validation seed; no frame-wise split")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_encoder(device)
    out.mkdir(parents=True)
    cache = Path(args.cache_dir) if args.cache_dir else out/"features"
    cache.mkdir(parents=True, exist_ok=True)
    bundle = Path(os.environ.get("ATEC_JEPA_BUNDLE", project/"work/jepa_prepare_2026-10-09"))
    weight_path = (bundle/"weights/dinov2_vits14_pretrain.pth").resolve()
    weight_stat = weight_path.stat()

    def features(path):
        stat = path.stat()
        key = hashlib.sha256(f"pose_features_v1:{path}:{stat.st_size}:{stat.st_mtime_ns}:{weight_path}:{weight_stat.st_size}:{weight_stat.st_mtime_ns}".encode()).hexdigest()
        dest = cache/f"{key}.pt"
        if dest.is_file():
            return torch.load(dest, map_location="cpu", weights_only=True)
        blocks = []
        with h5py.File(path, "r") as f:
            with torch.inference_mode():
                for start in range(0, len(f["rgb"]), args.feature_batch_size):
                    rgb = torch.from_numpy(f["rgb"][start:start+args.feature_batch_size]).to(device)
                    blocks.append(image_features(encoder, rgb).half().cpu())
            item = {"feature": torch.cat(blocks), "proprio": torch.from_numpy(f["proprio"][:]),
                    "xyz": torch.from_numpy(f["object_xyz"][:]), "quat": torch.from_numpy(f["object_wxyz"][:])}
        if not all(torch.isfinite(v).all() for v in item.values()):
            raise ValueError("Non-finite training data")
        torch.save(item, dest)
        print(f"[POSE] cached {len(item['feature'])} frames: {path}", flush=True)
        return item

    def merge(paths):
        items = [features(p) for p in paths]
        return {key: torch.cat([item[key] for item in items]) for key in items[0]}

    train, val = merge(train_paths), merge(val_paths)
    if len(train["feature"]) < 60 or len(val["feature"]) < 30:
        raise ValueError("Need >=60 training and >=30 validation frames across complete episodes")
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mean = train["proprio"].mean(0)
    std = train["proprio"].std(0).clamp_min(.01)
    position_mean = train["xyz"].mean(0)
    position_std = train["xyz"].std(0).clamp_min(.03)
    norm_mean, norm_std = position_mean.to(device), position_std.to(device)
    head = make_head().to(device)
    with torch.no_grad():
        # Nonzero quaternion output at initialization, avoiding normalize(0).
        bias = head[-1].bias.reshape(3, 8)
        bias.zero_(); bias[:, 3] = 1.
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(args.seed)

    def inputs(data, indices):
        prop = (data["proprio"][indices]-mean)/std
        return torch.cat((data["feature"][indices].float(), prop), dim=-1).to(device)

    for step in range(args.steps):
        idx = torch.randint(len(train["feature"]), (args.batch_size,), generator=generator)
        raw = head(inputs(train, idx))
        xyz, quat, confidence = decode(raw, norm_mean, norm_std)
        target_xyz = train["xyz"][idx].to(device)
        target_quat = F.normalize(train["quat"][idx].to(device), dim=-1)
        pos_loss = F.smooth_l1_loss((xyz-target_xyz)/norm_std, torch.zeros_like(xyz))
        dot = (quat*target_quat).sum(-1).abs().clamp(0, 1)
        rotation_loss = (1-dot.square()).mean()
        accurate = ((xyz.detach()-target_xyz).norm(dim=-1) <= .015) & (dot.detach() >= np.cos(np.deg2rad(6.)))
        # Accuracy targets are recomputed, not fake all-visible labels.
        quality_loss = F.binary_cross_entropy_with_logits(confidence, accurate.float())
        loss = pos_loss+rotation_loss+.2*quality_loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite pose loss; no deployable checkpoint emitted")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
        optimizer.step()
        if (step+1) % 250 == 0 or step+1 == args.steps:
            print(f"[POSE] step={step+1} loss={loss.item():.5f} xyz={pos_loss.item():.5f} quat={rotation_loss.item():.5f}", flush=True)

    head.eval(); predictions = [[], [], []]
    with torch.inference_mode():
        for start in range(0, len(val["feature"]), args.batch_size):
            idx = torch.arange(start, min(start+args.batch_size, len(val["feature"])))
            xyz, quat, conf = decode(head(inputs(val, idx)), norm_mean, norm_std)
            for bucket, value in zip(predictions, (xyz, quat, conf.sigmoid())):
                bucket.append(value.cpu().numpy())
    xyz, quat, probability = (np.concatenate(x) for x in predictions)
    calibration = pose_calibration(xyz, quat, probability, val["xyz"].numpy(), val["quat"].numpy())
    report = {**calibration, "train_seeds": sorted(train_seeds), "val_seeds": sorted(val_seeds),
              "train_frames": len(train["feature"]), "val_frames": len(val["feature"]),
              "steps": args.steps, "frozen_encoder": "DINOv2 ViT-S/14", "simulator_success_claim": False}
    payload = {"format": "atec_throw_rgb_pose_v1", "calibrated": False,
               "camera": camera_contract(ThrowConfig()), "input_fields": ["rgb", "proprio"],
               "proprio_mean": mean.tolist(), "proprio_std": std.tolist(),
               "position_mean": position_mean.tolist(), "position_std": position_std.tolist(),
               "confidence_thresholds": calibration["confidence_thresholds"],
               "model": {k: v.cpu() for k, v in head.state_dict().items()}, "calibration": report}
    torch.save(payload, out/"pose_candidate.pt")
    if calibration["passed"]:
        payload["calibrated"] = True
        torch.save(payload, out/"rgb_pose.pt")
    (out/"calibration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return 0 if calibration["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
