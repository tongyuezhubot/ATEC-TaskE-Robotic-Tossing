"""Analyze a completed frozen batch once and fit a small release dynamics model.

Inputs are initial six arm joints, measured object-to-tool offset, four CEM
parameters and swing duration. Targets are the physical prior's six release-state errors. This is
a structured state/action world model; RGB localization remains a separate head.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demo.throw_cem.residual_fit import extract_release_samples
from demo.throw_cem.config import ThrowConfig, DEFAULT_Q
from demo.throw_cem.physics import projectile_landing, physics_contract, predict_driven_release, ReleaseResidual
from demo.throw_cem.kinematics import PiperKinematics
from demo.throw_cem.joint_model import JointResponse
from demo.throw_cem.planner import ThrowPlanner


def features(sample):
    return np.r_[sample["q_start"][:6], sample["offset"], sample["theta"], sample["duration"]]


def reference_contract(contract, delay):
    result = dict(contract)
    if delay is not None:
        result["gripper_delay_prior"] = float(delay)
    return result


def replay_release_reference(sample, delay):
    """Recompute a forecast on IDENTICAL executed commands, preserving source data.

    Only its gripper-delay reference changes. Labels remain the same independent
    first detected free-flight observation, never a newly invented rollout.
    """
    if not 0. <= delay <= .12:
        raise ValueError("Reference delay must be within 0..120ms")
    directory = Path(sample["source"])
    meta = json.loads((directory/"run_config.json").read_text())
    source_cfg = ThrowConfig(**meta["throw_config"])
    if meta["physics"] != physics_contract(source_cfg):
        raise ValueError("Source physics provenance mismatch")
    obj = sample["object"]-1
    if source_cfg.throw_primitive != "joint_swing" or obj not in (0,1):
        raise ValueError("Reference replay only supports sugar/mustard joint-swing data")
    old_cfg = replace(source_cfg,planning_min_remaining_apex=0.)
    new_cfg = replace(old_cfg,gripper_delay_prior=float(delay))
    q = np.asarray(sample["q_start"]); theta = np.asarray(sample["theta"])
    offset = np.asarray(sample["offset"])
    kin = PiperKinematics(old_cfg)
    original, old_nominal = ThrowPlanner(old_cfg,kin).joint_swing_trajectory(q,theta,obj)
    new, new_nominal = ThrowPlanner(new_cfg,kin).joint_swing_trajectory(q,theta,obj)
    if not np.array_equal(original.targets,new.targets):
        raise ValueError("Reference replay would change commanded actions")
    rows = [json.loads(line) for line in (directory/"trace.jsonl").read_text().splitlines()]
    selected = [r for r in rows if r["action_stage"] == "swing"
                and r["controller"].get("cem_plans") == sample["plan"]
                and r["controller"].get("object") == sample["object"]]
    commands = np.array([r["action"] for r in selected])*source_cfg.action_scale+DEFAULT_Q
    needed = max(int(np.floor(new_nominal.open_step+delay/source_cfg.dt)),
                 int(np.floor(old_nominal.open_step+source_cfg.release_delay(obj)/source_cfg.dt)))+2
    if (len(commands)<needed or len(commands)>len(original.targets)
            or not np.allclose(commands,original.targets[:len(commands)],atol=1e-7,rtol=0.)):
        raise ValueError("Recorded executed commands do not match reference trajectory")
    model = JointResponse(new_cfg)
    old_prediction,_ = predict_driven_release(old_cfg,obj,q,offset,theta,old_nominal,
                                             commands,kin,model,ReleaseResidual())
    prediction,_ = predict_driven_release(new_cfg,obj,q,offset,theta,new_nominal,
                                         commands,kin,model,ReleaseResidual())
    old_state = np.r_[old_prediction.release_position,old_prediction.release_velocity]
    saved_state = np.asarray(sample["nominal_release_state"])
    if not np.allclose(old_state,saved_state,atol=1e-6,rtol=0.):
        raise ValueError("Saved nominal forecast cannot be reproduced")
    actual = saved_state+np.asarray(sample["delta"])
    state = np.r_[prediction.release_position,prediction.release_velocity]
    result = dict(sample)
    result.update(original_nominal_release_state=saved_state.tolist(),
                  original_delta=list(sample["delta"]),nominal_release_state=state.tolist(),
                  delta=(actual-state).tolist(),
                  prior_reference_replay={"source_delay":source_cfg.gripper_delay_prior,
                                          "reference_delay":float(delay),
                                          "identical_executed_commands":True,
                                          "source_data_modified":False,
                                          "feasibility_floor_disabled_for_offline_diagnostic_only":True})
    return result


def release_dynamics_contract(contract):
    # Kept independent of the runtime module so frozen collection can finish
    # and train before a new runtime model module is deployed.
    return {k: v for k, v in contract.items()
            if k not in ("planning_min_remaining_apex", "banana_held_local_z_range",
                         "joint_swing_duration", "joint_swing_lower", "joint_swing_upper")}


def observation_label(directory, sample):
    """Use one consistent FREE-FLIGHT observation, rather than a virtual earlier state.

    Existing extraction still proves three gravity-consistent frames. The
    target below is the evaluator's first upward gravity-consistent observation.
    """
    obj = sample["object"]-1
    for line in (directory / "trace.jsonl").read_text().splitlines():
        row = json.loads(line); c = row["controller"]; state = row["evaluation_only"]
        if (c.get("cem_plans") == sample["plan"] and c.get("object") == sample["object"]
                and state["upward_release"][obj]):
            actual = np.r_[state["object_com_xyz"][obj], state["object_velocity"][obj]]
            sample["effective_reference_delta"] = sample["delta"]
            sample["delta"] = (actual-np.asarray(sample["nominal_release_state"])).tolist()
            sample["label"] = "first evaluator-confirmed upward damped free-flight observation"
            sample["observed_step"] = row["step"]
            return
    raise ValueError("A validated release label lacks an upward free-flight observation")


def fit(samples, cfg, contract):
    object_ids = sorted({s["object"] for s in samples})
    if len(object_ids)>1:
        raise ValueError("Fit one object per model; inputs do not encode object identity")
    train = [s for s in samples if s["split"] == "train"]
    val = [s for s in samples if s["split"] == "validation"]
    if {s["seed"] for s in train} & {s["seed"] for s in val}:
        raise ValueError("Release training/validation seed overlap")
    report = {"passed": False, "train_samples": len(train), "val_samples": len(val),
              "train_seeds": sorted({s["seed"] for s in train}),
              "val_seeds": sorted({s["seed"] for s in val})}
    payload = {"format": "atec_release_world_v3", "calibrated": False,
               "physics": contract, "supported_object_ids": object_ids or [cfg.object_order[0]+1],
               "input_fields": ["initial_arm_q6", "object_tool_offset_xyz3", "cem_parameters4", "swing_duration1"],
               "target_fields": ["first_observed_freeflight_com_error_xyz3", "first_observed_freeflight_velocity_error_xyz3"],
               "state_time_basis": "first_detected_freeflight_observation",
               "calibration": report}
    sufficient = len(report["train_seeds"]) >= 8 and len(report["val_seeds"]) >= 3
    if len(train) < 3 or len(val) < 1:
        report["reason"] = "Too few independent releases to fit/evaluate even a candidate"
        return report, payload
    x, xv = np.array([features(s) for s in train]), np.array([features(s) for s in val])
    y, yv = np.array([s["delta"] for s in train]), np.array([s["delta"] for s in val])
    mean = x.mean(0)
    std = np.maximum(x.std(0), np.r_[np.full(6, .10), np.full(3, .02), [.08, .08, .08, .04, .05]])
    design = np.column_stack((np.ones(len(x)), (x - mean) / std))
    # Fixed regularization; no repeated tuning on the reserved validation set.
    penalty = np.eye(design.shape[1]) * 10.; penalty[0, 0] = .01
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y).T
    margin = np.maximum(.20 * np.ptp(x, axis=0),
                        np.r_[np.full(6, .15), np.full(3, .02), [.08, .08, .08, .04, .05]])
    bounds = np.stack((x.min(0) - margin, x.max(0) + margin), axis=-1)
    supported = np.all((xv >= bounds[:, 0]) & (xv <= bounds[:, 1]), axis=1)
    prediction = np.column_stack((np.ones(len(xv)), (xv - mean) / std)) @ weights.T
    prediction[~supported] = 0.
    error = yv - prediction
    scales = np.r_[np.full(3, .02), np.full(3, .20)]
    baseline = float(np.mean((yv / scales) ** 2))
    fitted = float(np.mean((error / scales) ** 2))
    position_rmse = float(np.sqrt(np.mean(np.sum(error[:, :3] ** 2, axis=1))))
    velocity_rmse = float(np.sqrt(np.mean(np.sum(error[:, 3:] ** 2, axis=1))))
    baseline_landing, fitted_landing, validation = [], [], []
    valid_landing = True
    for s, correction, covered in zip(val, prediction, supported):
        nominal = np.asarray(s["nominal_release_state"])
        actual = nominal + np.asarray(s["delta"])
        predicted = nominal + correction
        try:
            target, _ = projectile_landing(cfg, actual[:3], actual[3:], cfg.basket_xyz[2])
            before, _ = projectile_landing(cfg, nominal[:3], nominal[3:], cfg.basket_xyz[2])
            after, _ = projectile_landing(cfg, predicted[:3], predicted[3:], cfg.basket_xyz[2])
            baseline_landing.append(float(np.linalg.norm(before - target)))
            fitted_landing.append(float(np.linalg.norm(after - target)))
            validation.append({"seed": s["seed"], "supported": bool(covered),
                               "policy_inputs": s.get("policy_inputs", "oracle"),
                               "predicted_velocity": predicted[3:].tolist(),
                               "label_velocity": actual[3:].tolist(),
                               "landing_error_m": fitted_landing[-1]})
        except ValueError:
            valid_landing = False
    landing_rmse = float(np.sqrt(np.mean(np.square(fitted_landing)))) if valid_landing else None
    report.update(baseline_normalized_mse=baseline, fitted_normalized_mse=fitted,
                  position_rmse_m=position_rmse, velocity_rmse_m_s=velocity_rmse,
                  supported_validation_fraction=float(supported.mean()),
                  baseline_landing_rmse_m=float(np.sqrt(np.mean(np.square(baseline_landing)))) if valid_landing else None,
                  fitted_landing_rmse_m=landing_rmse, validation=validation)
    # Mixed Oracle/RGB data must not hide poor new-RGB accuracy behind older
    # Oracle validation examples. Apply the same existing error limits per cohort.
    cohorts = {}
    for kind in sorted({s.get("policy_inputs", "oracle") for s in val}):
        mask = np.array([s.get("policy_inputs", "oracle")==kind for s in val])
        rows = [r for r in validation if r["policy_inputs"]==kind]
        landing = float(np.sqrt(np.mean([r["landing_error_m"]**2 for r in rows]))) if len(rows)==int(mask.sum()) else None
        position = float(np.sqrt(np.mean(np.sum(error[mask,:3]**2,axis=1))))
        velocity = float(np.sqrt(np.mean(np.sum(error[mask,3:]**2,axis=1))))
        cohorts[kind] = {"samples":int(mask.sum()), "position_rmse_m":position,
                         "velocity_rmse_m_s":velocity, "landing_rmse_m":landing,
                         "supported_fraction":float(supported[mask].mean()),
                         "passed":bool(position<=.03 and velocity<=.20 and landing is not None
                                        and landing<=.06 and supported[mask].mean()>=.8)}
    report["validation_by_policy_inputs"] = cohorts
    # Model limits are data/physics validity checks, not artificially small
    # +/-0.25 m/s legacy clamps that conceal the measured release bias (~0.4).
    bounded = np.all(np.abs(prediction) <= np.r_[np.full(3, .05), np.full(3, .80)])
    report["passed"] = bool(sufficient and baseline > 1e-8 and fitted < .8 * baseline
                            and position_rmse <= .03 and velocity_rmse <= .20
                            and supported.mean() >= .8 and bounded and valid_landing
                            and landing_rmse <= .06 and all(r["passed"] for r in cohorts.values()))
    if any(s.get("policy_inputs")=="rgb" for s in train) and "rgb" not in cohorts:
        report["passed"] = False
        report["reason"] = "New RGB training releases lack independent RGB validation releases"
    elif not sufficient:
        report["reason"] = "Candidate fitted, but need at least 8 training and 3 validation releases before deployment"
    elif not report["passed"]:
        report["reason"] = "Held-out release state/landing accuracy or coverage gate failed"
    payload.update(calibrated=report["passed"], feature_mean=mean.tolist(),
                   feature_std=std.tolist(), feature_bounds=bounds.tolist(), weights=weights.tolist(),
                   proposal_examples={"source_split":"train",
                                      "features":x[:256].tolist(),
                                      "seeds":[s["seed"] for s in train[:256]]})
    return report, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--historical-train-runs", nargs="*", default=None)
    parser.add_argument("--historical-validation-runs", nargs="*", default=[])
    parser.add_argument("--reference-gripper-delay", type=float,
                        help="Explicit same-command forecast replay; source trajectories remain untouched")
    args = parser.parse_args()
    if args.reference_gripper_delay is not None and not 0. <= args.reference_gripper_delay <= .12:
        parser.error("Reference delay must be finite and within 0..120ms")
    root, out = Path(args.batch_dir), Path(args.output_dir)
    summary = json.loads((root / "batch_summary.json").read_text())
    if summary["state"] != "complete":
        parser.error("Analyze only after the fixed batch has completed")
    if out.exists():
        parser.error("Choose a new output directory")
    buckets = {}; samples = []; contract = cfg = None; episodes = []
    oracle_inputs = []
    for item in summary["episodes"]:
        object_id = item.get("object",3)
        directory = root / f"object{object_id}_seed{item['seed']}" / "episode"
        result = json.loads((directory / "result.json").read_text())
        oracle_inputs.append(bool(result.get("oracle_training_only", True)))
        if result["controller"].get("release_world_model_scoring_used", False):
            raise ValueError("This fitter handles prior-only collection; keep execution validation separate")
        extracted, physics = extract_release_samples(directory)
        if contract is not None and contract != physics:
            raise ValueError("Frozen batch contains incompatible physics contracts")
        contract = physics
        cfg = ThrowConfig(**json.loads((directory / "run_config.json").read_text())["throw_config"])
        c = result["controller"]
        for sample in extracted:
            sample.update(split=item["split"], q_start=c["plan_q_start"], duration=cfg.joint_swing_duration,
                          policy_inputs="oracle" if result.get("oracle_training_only", True) else "rgb",
                          nominal_release_state=c["predicted_release_xyz"] + c["predicted_release_velocity"])
            observation_label(directory, sample)
        samples += extracted
        apex = result["apex_rise_since_detected_release_m"][object_id-1]
        if result["requested_objects_success"]:
            category = "success"
        elif not c.get("cem_plans"):
            category = "planning_failure" if "CEM found no" in c.get("reason", "") else "grasp_or_transport_failure"
        elif apex is None:
            category = "no_valid_upward_release"
        elif apex < .01:
            category = "insufficient_upward_rise"
        elif result["gravity_consistent_flight_frames"][object_id-1] < 3:
            category = "insufficient_free_flight"
        else:
            category = "landing_or_observation_timeout"
        buckets[category] = buckets.get(category, 0) + 1
        episodes.append({**item, "category": category, "release_samples": len(extracted),
                         "apex_rise_m": apex, "flight_frames": result["gravity_consistent_flight_frames"][object_id-1],
                         "settled_frames": result["settled_frames"][object_id-1]})
    historical = args.historical_train_runs
    sidecar = root / "historical_train_runs.json"
    if historical is None:
        historical = json.loads(sidecar.read_text()) if sidecar.exists() else []
    seed_splits = {i["seed"]: i["split"] for i in summary["episodes"]}
    existing_sources = {str(Path(s["source"]).resolve()) for s in samples}
    historical_sources = [(run, "train") for run in historical]
    historical_sources += [(run, "validation") for run in args.historical_validation_runs]
    for run, split in historical_sources:
        directory = Path(run).resolve()
        if str(directory) in existing_sources:
            raise ValueError("Duplicate historical training source")
        result = json.loads((directory / "result.json").read_text())
        if seed_splits.get(result["seed"], split) != split:
            raise ValueError("Historical source crosses the reserved train/validation seed split")
        seed_splits[result["seed"]] = split
        if result["controller"].get("release_world_model_scoring_used", False):
            raise ValueError("Historical world-scored rollouts are not prior-error training labels")
        extracted, physics = extract_release_samples(directory)
        if (release_dynamics_contract(reference_contract(physics,args.reference_gripper_delay))
                != release_dynamics_contract(reference_contract(contract,args.reference_gripper_delay))):
            raise ValueError("Historical dynamics differ; do not mix labels")
        c = result["controller"]
        source_duration = json.loads((directory / "run_config.json").read_text())["throw_config"]["joint_swing_duration"]
        for sample in extracted:
            sample.update(split=split, q_start=c["plan_q_start"], historical=True, duration=source_duration,
                          policy_inputs="oracle" if result.get("oracle_training_only", True) else "rgb",
                          nominal_release_state=c["predicted_release_xyz"] + c["predicted_release_velocity"])
            observation_label(directory, sample)
        samples += extracted
        existing_sources.add(str(directory))
    if args.reference_gripper_delay is not None:
        samples = [replay_release_reference(s,args.reference_gripper_delay) for s in samples]
        cfg = replace(cfg,gripper_delay_prior=args.reference_gripper_delay)
        contract = physics_contract(cfg)
    report, payload = fit(samples, cfg, contract)
    payload["reference_gripper_delay_replay"] = args.reference_gripper_delay
    payload["release_dynamics_contract"] = release_dynamics_contract(contract)
    payload["historical_training_runs"] = historical
    payload["historical_validation_runs"] = args.historical_validation_runs
    analysis = {"episodes": len(episodes), "successes": summary["successful_episodes"],
                "categories": buckets, "release_labels": len(samples),
                "model": report, "per_episode": episodes,
                "historical_training_runs": historical,
                "historical_validation_runs": args.historical_validation_runs,
                "reference_gripper_delay_replay": args.reference_gripper_delay,
                "oracle_training_only": any(oracle_inputs),
                "collection_rgb_policy_inputs_only": not any(oracle_inputs),
                "rgb_deployment_success_claim": False}
    out.mkdir(parents=True)
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (out / "release_samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    name = "release_world.json" if report["passed"] else "release_world_candidate.json"
    (out / name).write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in analysis.items() if k != "per_episode"}), flush=True)


if __name__ == "__main__":
    main()
