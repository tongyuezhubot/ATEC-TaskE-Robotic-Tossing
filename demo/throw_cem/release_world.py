"""Calibrated state/action release dynamics used inside bounded CEM.

The physics prior supplies the nominal release. This model predicts its six
position/velocity errors; candidates outside the learned domain are rejected.
"""
import json
from pathlib import Path

import numpy as np

from .physics import physics_contract


def release_dynamics_contract(contract):
    return {k: v for k, v in contract.items()
            if k not in ("planning_min_remaining_apex", "banana_held_local_z_range",
                         "joint_swing_duration", "joint_swing_lower", "joint_swing_upper")}


class ReleaseWorldModel:
    def __init__(self, path, cfg):
        payload = json.loads(Path(path).read_text())
        if (payload.get("format") != "atec_release_world_v3" or not payload.get("calibrated", False)
                or payload.get("state_time_basis") != "first_detected_freeflight_observation"):
            raise ValueError("Release world model must pass held-out calibration before use")
        if cfg.throw_primitive != "joint_swing":
            raise ValueError("Release world model requires its trained joint_swing primitive")
        if payload.get("release_dynamics_contract") != release_dynamics_contract(physics_contract(cfg)):
            raise ValueError("Release world model physical contract mismatch")
        self.weights = np.asarray(payload["weights"], float)
        self.mean = np.asarray(payload["feature_mean"], float)
        self.std = np.asarray(payload["feature_std"], float)
        self.bounds = np.asarray(payload["feature_bounds"], float)
        if (self.weights.shape != (6, 15) or self.mean.shape != (14,) or self.std.shape != (14,)
                or self.bounds.shape != (14, 2) or np.any(self.std <= 0)
                or not all(np.isfinite(a).all() for a in (self.weights, self.mean, self.std, self.bounds))
                or np.any(self.bounds[:, 0] > self.bounds[:, 1])):
            raise ValueError("Invalid calibrated release world model arrays")
        self.object_ids = payload["supported_object_ids"]
        if not self.object_ids or any(o not in (1, 2, 3) for o in self.object_ids):
            raise ValueError("Invalid release world model object domain")
        self.duration = cfg.joint_swing_duration
        self.proposal_features = np.empty((0,14))
        examples = payload.get("proposal_examples")
        if examples is not None:
            features = np.asarray(examples.get("features"),float)
            seeds = examples.get("seeds",[])
            calibration = payload.get("calibration",{})
            training = set(calibration.get("train_seeds",[]))
            validation = set(calibration.get("val_seeds",[]))
            if (examples.get("source_split") != "train" or features.ndim != 2
                    or features.shape[1:] != (14,) or not 1 <= len(features) <= 256
                    or len(seeds) != len(features) or not np.isfinite(features).all()
                    or any(seed not in training or seed in validation for seed in seeds)
                    or np.any(features<self.bounds[:,0]) or np.any(features>self.bounds[:,1])):
                raise ValueError("CEM proposal examples must be bounded training-only features")
            self.proposal_features = features
        self.calibrated = True
        self.world_model = True
        self.observation_aligned = True
        self.calls = self.rejections = 0

    def candidate_proposals(self,obj,q,offset):
        """Four nearby TRAIN actions seed CEM, never bypassing its scorer/guards.

        Legacy checkpoints have no examples and preserve their random proposals.
        No validation trajectories, actor weights or live simulator truth used.
        """
        if not len(self.proposal_features):
            return np.empty((0,4))
        self.candidate_bounds(obj,q,offset)
        index = np.r_[0:9,13]
        state = np.r_[np.asarray(q)[:6],offset,self.duration]
        distance = np.sum(((self.proposal_features[:,index]-state)/self.std[index])**2,axis=1)
        nearest = np.argsort(distance,kind="stable")[:4]
        return self.proposal_features[nearest,9:13].copy()

    def candidate_stances(self,obj,q):
        """Training arm poses for a physical carry, not substituted observations.

        Controller must validate the path, execute the carry and re-observe.
        Current state need not already be in-domain; final planning still must.
        """
        if obj+1 not in self.object_ids or not len(self.proposal_features):
            return np.empty((0,6))
        distance=np.sum(((self.proposal_features[:,:6]-np.asarray(q)[:6])/self.std[:6])**2,axis=1)
        nearest=np.argsort(distance,kind="stable")
        return self.proposal_features[nearest,:6].copy()

    def candidate_bounds(self, obj, q, offset):
        """Proposal domain only: never enlarge the calibrated model support."""
        state = np.r_[np.asarray(q)[:6], offset, self.duration]
        bounds = self.bounds[np.r_[0:9, 13]]
        if (obj+1 not in self.object_ids or state.shape != (10,)
                or not np.isfinite(state).all() or np.any(state < bounds[:,0])
                or np.any(state > bounds[:,1])):
            raise ValueError("Release world model current state outside calibrated domain")
        return self.bounds[9:13,0].copy(), self.bounds[9:13,1].copy()

    def correction(self, obj, theta, offset, q=None):
        if q is None or obj+1 not in self.object_ids:
            raise ValueError("Release world model lacks the requested state/object domain")
        features = np.r_[np.asarray(q)[:6], offset, theta, self.duration]
        self.calls += 1
        if (features.shape != (14,) or not np.isfinite(features).all()
                or np.any(features < self.bounds[:, 0]) or np.any(features > self.bounds[:, 1])):
            self.rejections += 1
            raise ValueError("Release world model candidate outside calibrated domain")
        correction = self.weights @ np.r_[1., (features-self.mean)/self.std]
        if np.any(np.abs(correction) > np.r_[np.full(3, .05), np.full(3, .80)]):
            self.rejections += 1
            raise ValueError("Release world model correction exceeds physical validity bounds")
        return correction
