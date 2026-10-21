"""Summarize one completed held-out world-model execution batch without training."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True)
    args = parser.parse_args()
    root = Path(args.batch_dir)
    summary = json.loads((root / "batch_summary.json").read_text())
    manifest = json.loads((root / "batch_manifest.json").read_text())
    if summary["state"] != "complete" or len(summary["episodes"]) != summary["planned_episodes"]:
        parser.error("Execution validation has not completed")
    if hashlib.sha256((root / "release_world.json").read_bytes()).hexdigest() != manifest["world_model_sha256"]:
        raise ValueError("World weights changed during execution validation")
    trials = []; oracle = []
    for episode in summary["episodes"]:
        object_id = episode.get("object",3)
        result = json.loads((root / f"object{object_id}_seed{episode['seed']}" / "episode/result.json").read_text())
        c = result["controller"]
        oracle.append(result["oracle_training_only"])
        if not c.get("release_world_model_scoring_used", False):
            raise ValueError("A supposed world-model evaluation did not load the world model")
        trials.append({"seed": episode["seed"], "object":object_id, "role": "held_out_execution_validation",
                       "success": result["requested_objects_success"], "steps": result["steps"],
                       "cem_plans": c["cem_plans"], "world_calls": c["release_world_model_calls"],
                       "world_domain_rejections": c["release_world_model_domain_rejections"],
                       "apex_rise_m": result["apex_rise_since_detected_release_m"][object_id-1],
                       "flight_frames": result["gravity_consistent_flight_frames"][object_id-1],
                       "settled_frames": result["settled_frames"][object_id-1], "reason": c["reason"]})
    report = {"purpose": "held_out_world_execution", "episodes": len(trials),
              "successes": sum(t["success"] for t in trials),
              "success_fraction": sum(t["success"] for t in trials)/len(trials),
              "weights_frozen": True, "weights_sha256": manifest["world_model_sha256"],
              "none_of_these_runs_used_for_training": True, "oracle_training_only": any(oracle),
              "rgb_policy_inputs_only": not any(oracle),
              "rgb_deployment_success_claim": False, "trials": trials}
    (root / "world_execution_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
