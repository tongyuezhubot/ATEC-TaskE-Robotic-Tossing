"""Train the throwing residual from held-out-seed finite simulator traces."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demo.throw_cem.residual_fit import extract_release_samples, fit_release_residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-runs", nargs="+", required=True)
    parser.add_argument("--val-runs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        parser.error("Choose a NEW output directory")
    paths = [Path(x).resolve() for x in args.train_runs+args.val_runs]
    if len(set(paths)) != len(paths):
        parser.error("Duplicate runs / train-validation overlap")
    train, val, contract = [], [], None
    seeds = [set(), set()]
    for split, runs in enumerate((args.train_runs, args.val_runs)):
        for run in runs:
            samples, physics = extract_release_samples(run)
            if contract is not None and contract != physics:
                raise ValueError("Cannot mix incompatible throw physics/configurations")
            contract = physics
            seeds[split].add(json.loads((Path(run)/"run_config.json").read_text())["seed"])
            (train if split == 0 else val).extend(samples)
    if seeds[0] & seeds[1]:
        raise ValueError("Validation requires different episode seeds")
    report, coefficients, bounds = fit_release_residual(train, val)
    payload = {"format": "atec_throw_release_residual_v1", "calibrated": report["passed"],
               "coefficients": coefficients.tolist(), "feature_bounds": bounds.tolist(), "physics": contract,
               "calibration": report, "train_seeds": sorted(seeds[0]), "val_seeds": sorted(seeds[1])}
    out.mkdir(parents=True)
    (out/"samples.jsonl").write_text("".join(json.dumps(s)+"\n" for s in train+val))
    (out/"calibration.json").write_text(json.dumps(report, indent=2))
    name = "release_residual.json" if report["passed"] else "residual_candidate.json"
    (out/name).write_text(json.dumps(payload, indent=2))
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
