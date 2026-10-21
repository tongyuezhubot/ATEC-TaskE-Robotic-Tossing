"""Small learned native position-drive model, not an action-generating actor.

Predict actual joint positions from bounded position commands. Independent-seed
calibration is mandatory for loading weights. No simulator truth in inference.
"""
import json
from pathlib import Path
import numpy as np
from .config import DEFAULT_Q


class JointResponse:
    def __init__(self, cfg, path=None):
        self.dt = cfg.dt
        self.alpha = np.asarray(cfg.joint_response_alpha, float)
        self.calibrated = False
        if path:
            payload = json.loads(Path(path).read_text())
            if payload.get("format") != "atec_joint_response_v1" or not payload.get("calibrated", False):
                raise ValueError("Uncalibrated native joint response model")
            if payload["dt"] != cfg.dt or payload["action_scale"] != cfg.action_scale or payload["default_q"] != DEFAULT_Q.tolist():
                raise ValueError("Joint model action/timing contract mismatch")
            alpha = np.asarray(payload["alpha"], float)
            if alpha.shape != (6,) or not np.isfinite(alpha).all() or np.any(alpha < .05) or np.any(alpha > .5):
                raise ValueError("Invalid fitted joint response")
            self.alpha = alpha
            self.calibrated = True

    def rollout(self, start, targets):
        q = np.asarray(start, float)[:6].copy()
        result = []
        for command in np.asarray(targets):
            q = q+self.alpha*(command[:6]-q)
            result.append(q.copy())
        return np.asarray(result)


def fit_response(training, validation):
    """Fit a diagonal one-step response on multi-step measured transitions.

    A segment is (initial six joints, command sequence, final six joints).
    Static/blocked descent segments are excluded by the dataset reader.
    """
    grid = np.linspace(.05, .50, 91)
    alpha, reports = [], []
    for joint in range(6):
        def moving(data):
            return [s for s in data if abs(s[2][joint]-s[0][joint]) >= .005
                    and max(abs(u[joint]-s[0][joint]) for u in s[1]) >= .015]
        train, val = moving(training), moving(validation)
        report = {"joint": joint+1, "train_segments": len(train), "val_segments": len(val), "passed": False}
        if len(train) < 20 or len(val) < 5:
            report["reason"] = "insufficient moving segments"
            alpha.append(.2); reports.append(report); continue
        errors = []
        for initial, commands, final in train:
            state = np.full(len(grid), initial[joint])
            for command in commands:
                state += grid*(command[joint]-state)
            errors.append(state-final[joint])
        errors = np.asarray(errors)
        # Robust against occasional contact projections; no fit to val frames.
        absolute = np.abs(errors)
        loss = np.where(absolute <= .05, .5*errors**2, .05*(absolute-.025)).mean(0)
        chosen = float(grid[np.argmin(loss)])
        fitted, instant = [], []
        for initial, commands, final in val:
            state = initial[joint]
            for command in commands:
                state += chosen*(command[joint]-state)
            fitted.append(state-final[joint]); instant.append(commands[-1][joint]-final[joint])
        fitted_rmse = float(np.sqrt(np.mean(np.square(fitted))))
        instant_rmse = float(np.sqrt(np.mean(np.square(instant))))
        passed = fitted_rmse < .8*instant_rmse and fitted_rmse <= .06
        report.update(passed=bool(passed), alpha=chosen, time_constant_s=float(-.02/np.log(1-chosen)),
                      fitted_rmse_rad=fitted_rmse, instantaneous_rmse_rad=instant_rmse)
        alpha.append(chosen); reports.append(report)
    return np.asarray(alpha), {"passed": all(r["passed"] for r in reports), "joints": reports}
