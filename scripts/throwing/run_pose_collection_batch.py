"""Eight finite privileged pose-data episodes, then one frozen-head training run.

Failures provide valid pose labels, not successful throwing demonstrations.
No retries, actuator overrides, unchecked deployment, or residual training.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="New directory for this finite batch")
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    subprocess.run(["df", "-h", str(project)], check=True)
    jobs = [(2, 46), (1, 47), (3, 48), (2, 49), (1, 50), (3, 51), (2, 52), (1, 53)]
    import h5py
    train, val, results = [], [], []
    for obj, seed in jobs:
        directory = root/f"object{obj}_seed{seed}"
        directory.mkdir()
        with (directory/"run.log").open("x") as log:
            status = subprocess.run(["bash", str(project/"scripts/throwing/launch_collection.sh"),
                                     str(directory), str(obj), str(seed), "1000", "180"],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=project)
        path = directory/"episode/pose_data.hdf5"
        frames = 0
        if status.returncode == 0 and path.is_file():
            with h5py.File(path, "r") as f:
                if f.attrs.get("complete", False):
                    frames = len(f["rgb"])
        # Retain all files, even incomplete failures; only valid labels train.
        if frames:
            (train if seed < 52 else val).append(str(path))
        result = {"object": obj, "seed": seed, "exit_code": status.returncode,
                  "pose_frames": frames, "throwing_success_claim": False}
        results.append(result)
        (root/"collection_summary.json").write_text(json.dumps(results, indent=2))
        print("[POSE-BATCH] "+json.dumps(result), flush=True)
    # Existing data is preserved and provides motion frames from held bananas.
    old = project/"work"
    for name in ("toss_banana_2026-10-13_IDkgm5cE", "toss_banana_approach_2026-10-13_IiYRMvrE",
                 "toss_banana_settle_2026-10-13_mLYIgyLu", "toss_banana_jointbrake_2026-10-13_I5fdkUEQ",
                 "toss_banana_tilt_2026-10-13_6SH6kQoR"):
        train.append(str(old/name/"episode/pose_data.hdf5"))
    val.append(str(old/"toss_banana_release_2026-10-13_w7UJcOBy/episode/pose_data.hdf5"))
    with (root/"training.log").open("x") as log:
        result = subprocess.run([sys.executable, str(project/"scripts/throwing/train_pose.py"),
                                 "--train-files", *train, "--val-files", *val,
                                 "--steps", "1500", "--cache-dir", str(old/"toss_pose_frozen_features"),
                                 "--output-dir", str(root/"model")], stdout=log, stderr=subprocess.STDOUT, cwd=project)
    (root/"exit_code.txt").write_text(str(result.returncode)+"\n")
    print(f"[POSE-BATCH] training exit={result.returncode}; no deployment requested", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
