"""Finite fixed-code single-object batch. Keep successful AND failed outcomes.

The first episode is the startup check. An execution failure stops the batch;
physical task failures are data and never trigger retries or parameter changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def fingerprint(project):
    files = list((project / "demo/throw_cem").glob("*.py"))
    files += list((project / "demo/throw_cem").glob("*.json"))
    files += [project / name for name in (
        "demo/solution_throw_cem.py", "scripts/throwing/run_throw_task.py",
        "scripts/throwing/launch_collection.sh")]
    return {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="An existing empty directory")
    parser.add_argument("--planner-config", required=True)
    parser.add_argument("--joint-model", required=True)
    parser.add_argument("--world-model", help="Optional held-out calibrated release model")
    parser.add_argument("--perception-model", help="Calibrated RGB checkpoint; disables Oracle policy inputs")
    parser.add_argument("--object", type=int, choices=(1,2,3), default=3)
    parser.add_argument("--seed-start", type=int, default=70)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--validation-only", action="store_true",
                        help="Reserve ALL episodes for calibration before collecting; no fitting on these seeds")
    args = parser.parse_args()
    if not 1 <= args.count <= 20:
        parser.error("A finite batch of 1..20 episodes is required")
    if args.validation_only and args.world_model:
        parser.error("World-scored execution and prior-only calibration have separate roles")
    project = Path(__file__).resolve().parents[2]
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "batch_manifest.json").exists():
        parser.error("Choose a new batch directory; existing data is preserved")
    motion = root / "planner_config.json"
    model = root / "joint_response.json"
    shutil.copyfile(args.planner_config, motion)
    shutil.copyfile(args.joint_model, model)
    world = None
    if args.world_model:
        world = root / "release_world.json"
        shutil.copyfile(args.world_model, world)
        world_metadata = json.loads(world.read_text())
        if not world_metadata.get("calibrated", False):
            parser.error("Only a calibrated release world model can be evaluated")
        if args.object not in world_metadata.get("supported_object_ids", []):
            parser.error("The release world model does not support this object")
    perception = None
    if args.perception_model:
        import torch
        perception = root / "rgb_pose.pt"
        shutil.copyfile(args.perception_model, perception)
        if not torch.load(perception, map_location="cpu", weights_only=True).get("calibrated", False):
            parser.error("Only calibrated RGB perception can be evaluated")
    hashes = fingerprint(project)
    train_count = 0 if args.validation_only else max(1, args.count * 3 // 4)
    jobs = [{"object": args.object, "seed": args.seed_start + i,
             "split": ("execution_validation" if world else
                       "train" if i < train_count else "validation")}
            for i in range(args.count)]
    manifest = {"format": "atec_toss_batch_v1", "jobs": jobs,
                "code_sha256": hashes, "planner_config": json.loads(motion.read_text()),
                "joint_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "world_model_sha256": hashlib.sha256(world.read_bytes()).hexdigest() if world else None,
                "purpose": "held_out_world_execution" if world else "collection",
                "validation_only_collection": args.validation_only,
                "perception_model_sha256": hashlib.sha256(perception.read_bytes()).hexdigest() if perception else None,
                "primitive": "joint_swing", "max_steps": 1000,
                "wall_seconds_per_episode": 180, "data_stride": 1,
                "oracle_training_only": perception is None,
                "dataset_alignment": "RGB/proprio at pre-action step s pairs with trace row step s+1 action; the trace truth is post-action",
                "parameters_frozen_for_entire_batch": True}
    save(root / "batch_manifest.json", manifest)
    # Snapshot sources for exact local reconstruction after collection.
    for name in hashes:
        destination = root / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, destination)
    import os
    environment = os.environ.copy()
    environment["ATEC_THROW_JOINT_MODEL"] = str(model)
    environment.pop("ATEC_THROW_RESIDUAL", None)
    if world:
        environment["ATEC_THROW_WORLD_MODEL"] = str(world)
    else:
        environment.pop("ATEC_THROW_WORLD_MODEL", None)
    summary = {"state": "running", "planned_episodes": args.count,
               "completed_episodes": 0, "successful_episodes": 0, "episodes": []}
    save(root / "batch_summary.json", summary)
    status_code = 0
    for job in jobs:
        if fingerprint(project) != hashes:
            summary.update(state="stopped", stop_reason="Code changed during frozen collection")
            status_code = 2
            break
        if shutil.disk_usage(project).free < 2 * 1024**3:
            summary.update(state="stopped", stop_reason="Less than 2 GiB free; all recorded files preserved")
            status_code = 2
            break
        directory = root / f"object{job['object']}_seed{job['seed']}"
        directory.mkdir()
        started = time.monotonic()
        with (directory / "run.log").open("x") as log:
            result = subprocess.run([
                "bash", str(project / "scripts/throwing/launch_collection.sh"),
                str(directory), str(job["object"]), str(job["seed"]), "1000", "180", "joint_swing",
                str(motion), "1", str(perception) if perception else ""], cwd=project, env=environment,
                stdout=log, stderr=subprocess.STDOUT)
        report_path = directory / "episode/result.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        item = {**job, "exit_code": result.returncode,
                "wall_seconds": round(time.monotonic() - started, 1),
                "success": bool(report.get("requested_objects_success", False)),
                "steps": report.get("steps"), "frames": report.get("pose_dataset_frames", 0),
                "cem_plans": report.get("controller", {}).get("cem_plans", 0),
                "reason": report.get("controller", {}).get("reason", ""),
                "termination_reason": report.get("termination_reason"),
                "episode": str(directory / "episode")}
        summary["episodes"].append(item)
        summary["completed_episodes"] += 1
        summary["successful_episodes"] += int(item["success"])
        save(root / "batch_summary.json", summary)
        print("[TOSS-BATCH] " + json.dumps(item), flush=True)
        if result.returncode != 0 or not report:
            summary.update(state="stopped", stop_reason="Episode execution failed; no automatic retries")
            status_code = result.returncode or 2
            break
    else:
        summary["state"] = "complete"
    save(root / "batch_summary.json", summary)
    (root / "exit_code.txt").write_text(str(status_code) + "\n")
    print("[TOSS-BATCH] " + json.dumps({k: v for k, v in summary.items() if k != "episodes"}), flush=True)
    return status_code


if __name__ == "__main__":
    raise SystemExit(main())
