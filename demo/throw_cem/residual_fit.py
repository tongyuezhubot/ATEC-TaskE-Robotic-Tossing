"""Effective release labels from evaluation traces and small ridge dynamics.

These are training-only kinematic free-flight proxies, NOT contact measurements.
Fit at the prior's reference time, so delay errors become consistent ballistic
state corrections without pretending the exact physical detachment was observed.
"""
import json
from pathlib import Path
import numpy as np
from .config import ThrowConfig
from .physics import ReleaseResidual, predict_throw, physics_contract, predict_driven_release, flight_state
from .joint_model import JointResponse
from .kinematics import PiperKinematics
from .planner import ThrowPlanner


def extract_release_samples(directory):
    directory = Path(directory)
    meta = json.loads((directory/"run_config.json").read_text())
    cfg = ThrowConfig(**meta["throw_config"])
    if meta.get("physics") != physics_contract(cfg):
        raise ValueError("Run lacks the matching release/braking contract; do not relabel old trajectories")
    groups = {}
    for line in (directory/"trace.jsonl").read_text().splitlines():
        row = json.loads(line)
        diag = row["controller"]
        if row["action_stage"] not in ("swing", "wait_landing") or not diag.get("cem_plans"):
            continue
        key = (diag["cem_plans"], diag["object"])
        groups.setdefault(key, []).append(row)
    samples = []
    for (plan_id, object_id), rows in groups.items():
        obj = object_id-1
        first = rows[0]
        if first["action_stage"] != "swing" or first["controller"]["swing_cursor"] != 1:
            continue
        diag = first["controller"]
        theta, offset = np.asarray(diag["parameters"]), np.asarray(diag["object_offset"])
        try:
            if cfg.throw_primitive == "joint_swing":
                kin = PiperKinematics(cfg); model = JointResponse(cfg); q_start = np.asarray(diag["plan_q_start"], float)
                trajectory, prediction = ThrowPlanner(cfg, kin, joint_model=model).joint_swing_trajectory(q_start, theta, obj)
                prediction, _ = predict_driven_release(cfg, obj, q_start, offset, theta, prediction,
                                                       trajectory.targets, kin, model, ReleaseResidual())
            else:
                prediction = predict_throw(cfg, obj, np.asarray(diag["plan_start"]), offset, theta, ReleaseResidual())
            if cfg.throw_primitive != "joint_swing" and (not np.allclose(cfg.joint_response_alpha, 1.) or cfg.throw_primitive == "arm_swing"):
                kin = PiperKinematics(cfg); model = JointResponse(cfg)
                q_start = np.asarray(diag["plan_q_start"], float)
                trajectory = ThrowPlanner(cfg, kin, joint_model=model).joint_trajectory(q_start, kin.forward(q_start)[:3, :3], prediction, theta[2])
                prediction, _ = predict_driven_release(cfg, obj, q_start, offset, theta, prediction,
                                                      trajectory.targets, kin, model, ReleaseResidual())
        except ValueError:
            continue
        reference_t = prediction.open_step*cfg.dt+cfg.release_delay(obj)
        run, previous = [], None
        for row in rows:
            t = (row["step"]-first["step"]+1)*cfg.dt
            state = row["evaluation_only"]
            pos, vel = np.asarray(state.get("object_com_xyz", state["object_xyz"]))[obj], np.asarray(state["object_velocity"])[obj]
            gap = abs(np.diff(state["jaws"])[0])
            separated = np.linalg.norm(pos-(np.asarray(state["ee_xyz"])+offset)) > .02
            held = state["held_evidence"][obj]
            upward = state["upward_release"][obj]
            expected = None if previous is None else flight_state(cfg, np.zeros(3), previous[1], cfg.dt)[1]
            ballistic = (previous is not None and row["step"] == previous[0]+1
                         and np.linalg.norm(vel-expected)/cfg.dt < 3.)
            valid = (held and upward and separated and state.get("opening_command", gap >= cfg.open_gap_min) and ballistic
                     and 0 <= t-reference_t <= .15)
            run = run+[(t, pos, vel)] if valid else []
            previous = (row["step"], vel)
            if len(run) == 3:
                effective = []
                for time, position, velocity in run:
                    delta = reference_t-time
                    release_position, release_velocity = flight_state(cfg, position, velocity, delta)
                    effective.append(np.r_[release_position, release_velocity])
                effective = np.asarray(effective)
                if np.max(np.std(effective[:, :3], axis=0)) <= .005 and np.max(np.std(effective[:, 3:], axis=0)) <= .05:
                    observed = effective.mean(0)
                    nominal = np.r_[prediction.release_position, prediction.release_velocity]
                    samples.append({"object": object_id, "seed": meta["seed"], "plan": plan_id,
                                    "source": str(directory.resolve()), "theta": theta.tolist(), "offset": offset.tolist(),
                                    "delta": (observed-nominal).tolist(),
                                    "label": "damped kinematic flight inversion at nominal release time"})
                break
    return samples, physics_contract(cfg)


def fit_release_residual(train, val, min_train=8, min_val=3):
    coefficients = np.zeros((3, 6, 8)); bounds = np.zeros((3, 7, 2)); reports = []
    scales = np.r_[np.full(3, .02), np.full(3, .15)]
    for obj in range(3):
        training = [s for s in train if s["object"] == obj+1]
        validation = [s for s in val if s["object"] == obj+1]
        train_seeds = {s["seed"] for s in training}; val_seeds = {s["seed"] for s in validation}
        if train_seeds & val_seeds:
            raise ValueError("Release training/validation seed overlap")
        report = {"object": obj+1, "train_samples": len(training), "val_samples": len(validation),
                  "train_unique_seeds": len(train_seeds), "val_unique_seeds": len(val_seeds), "passed": False}
        if len(train_seeds) < min_train or len(val_seeds) < min_val:
            report["reason"] = "insufficient independent release samples"
            reports.append(report); continue
        x = np.array([s["theta"]+s["offset"] for s in training], float)
        y = np.array([s["delta"] for s in training], float)
        xv = np.array([s["theta"]+s["offset"] for s in validation], float)
        yv = np.array([s["delta"] for s in validation], float)
        if not all(np.isfinite(a).all() for a in (x, y, xv, yv)):
            raise ValueError("Non-finite release data")
        mean = x.mean(0); std = np.maximum(x.std(0), [.05, .05, .05, .02, .02, .02, .02])
        design = np.column_stack((np.ones(len(x)), (x-mean)/std))
        penalty = np.eye(8)*2.; penalty[0, 0] = .1
        weights = np.linalg.solve(design.T@design+penalty, design.T@y).T
        coeff = weights.copy(); coeff[:, 1:] /= std
        coeff[:, 0] -= coeff[:, 1:]@mean
        margin = np.maximum(.1*(x.max(0)-x.min(0)), [.02, .02, .02, .01, .01, .01, .01])
        domain = np.stack((x.min(0)-margin, x.max(0)+margin), -1)
        correction = np.column_stack((np.ones(len(xv)), xv))@coeff.T
        supported = np.all((xv >= domain[:, 0]) & (xv <= domain[:, 1]), axis=1)
        correction[~supported] = 0.
        error = yv-correction
        baseline = float(np.mean((yv/scales)**2)); fitted = float(np.mean((error/scales)**2))
        position_rmse = float(np.sqrt(np.mean(np.sum(error[:, :3]**2, axis=-1))))
        velocity_rmse = float(np.sqrt(np.mean(np.sum(error[:, 3:]**2, axis=-1))))
        bounded = np.all(np.abs(correction) <= np.r_[np.full(3, .03), np.full(3, .25)])
        passed = (baseline > 1e-8 and fitted < .9*baseline and position_rmse <= .03
                  and velocity_rmse <= .20 and supported.mean() >= .8 and bool(bounded))
        coefficients[obj] = coeff; bounds[obj] = domain
        report.update(passed=bool(passed), baseline_normalized_mse=baseline, fitted_normalized_mse=fitted,
                      position_rmse_m=position_rmse, velocity_rmse_m_s=velocity_rmse,
                      supported_validation_fraction=float(supported.mean()))
        reports.append(report)
    return {"passed": all(r["passed"] for r in reports), "objects": reports}, coefficients, bounds
