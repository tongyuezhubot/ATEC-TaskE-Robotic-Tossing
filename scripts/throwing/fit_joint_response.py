"""Fit native drive lag from existing aligned pose datasets + action traces."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
from demo.throw_cem.config import ThrowConfig, DEFAULT_Q
from demo.throw_cem.joint_model import fit_response


def segments(path):
    rows = {r["step"]: r for r in (json.loads(l) for l in (path.parent/"trace.jsonl").read_text().splitlines())}
    allowed = {"home", "pre_grasp", "lift", "pre_throw", "swing"}
    result = []
    with h5py.File(path, "r") as f:
        if not f.attrs.get("complete", False):
            raise ValueError("Incomplete source episode")
        steps = f["step"][:]; positions = f["proprio"][:, :6]+DEFAULT_Q[:6]
        seed = int(f.attrs["source_seed"])
        for index in range(len(steps)-1):
            lo, hi = int(steps[index]), int(steps[index+1])
            if hi-lo != 5:
                continue
            block = [rows.get(step) for step in range(lo+1, hi+1)]
            if any(r is None or r["action_stage"] not in allowed for r in block):
                continue
            commands = np.asarray([r["action"][:6] for r in block])*.5+DEFAULT_Q[:6]
            result.append((positions[index], commands, positions[index+1]))
    return result, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-files", nargs="+", required=True)
    parser.add_argument("--val-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        parser.error("Choose a NEW output directory")
    paths = [Path(p).resolve() for p in args.train_files+args.val_files]
    if len(set(paths)) != len(paths):
        parser.error("Duplicate / overlapping files")
    data, seeds = [[], []], [set(), set()]
    for split, files in enumerate((args.train_files, args.val_files)):
        for source in files:
            values, seed = segments(Path(source))
            data[split].extend(values); seeds[split].add(seed)
    if len(seeds[0]) < 2 or not seeds[1] or seeds[0] & seeds[1]:
        raise ValueError("Require independent training and validation seeds")
    alpha, report = fit_response(*data)
    cfg = ThrowConfig()
    payload = {"format": "atec_joint_response_v1", "calibrated": report["passed"],
               "dt": cfg.dt, "action_scale": cfg.action_scale, "default_q": DEFAULT_Q.tolist(),
               "alpha": alpha.tolist(), "calibration": report,
               "train_seeds": sorted(seeds[0]), "val_seeds": sorted(seeds[1])}
    out.mkdir(parents=True)
    (out/"calibration.json").write_text(json.dumps(payload, indent=2))
    name = "joint_response.json" if report["passed"] else "joint_response_candidate.json"
    (out/name).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
