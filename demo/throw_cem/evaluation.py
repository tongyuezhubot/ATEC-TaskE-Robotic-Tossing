"""Privileged NEW-TASK evaluator; never supplied to the deployed controller.

Kinematic evidence: lifted near closed jaws, upward opening, gravity-consistent
separated flight, then stable basket occupancy. This is not a force-contact sensor.
"""
import numpy as np
from .physics import flight_state


class ThrowEvaluator:
    def __init__(self, cfg, initial_positions):
        self.cfg = cfg; self.initial = np.asarray(initial_positions).copy()
        self.held_count = np.zeros(3, int); self.held = np.zeros(3, bool)
        self.held_age = np.full(3, 10000, int); self.flight_run = np.zeros(3, int)
        self.released_up = np.zeros(3, bool); self.flight_count = np.zeros(3, int)
        self.settled_count = np.zeros(3, int); self.success = np.zeros(3, bool)
        self.release_xyz = np.full((3, 3), np.nan); self.apex = self.initial[:, 2].copy()
        self.last_velocity = None
        self.held_gap = np.full(3, np.inf)

    def update(self, positions, velocities, ee_xyz, ee_velocity, jaws, *, root_positions=None,
               ee_angular_velocity=None, opening_command=None):
        cfg = self.cfg
        pos, vel = np.asarray(positions), np.asarray(velocities)
        if pos.shape != (3, 3) or vel.shape != (3, 3) or not np.isfinite(pos).all() or not np.isfinite(vel).all():
            raise ValueError("Invalid evaluator truth")
        distance = np.linalg.norm(pos-np.asarray(ee_xyz), axis=1)
        gap = abs(float(jaws[0]-jaws[1]))
        omega = np.zeros(3) if ee_angular_velocity is None else np.asarray(ee_angular_velocity)
        held_velocity = np.asarray(ee_velocity)+np.cross(omega, pos-np.asarray(ee_xyz))
        held_now = ((pos[:, 2]-self.initial[:, 2] >= .06) & (distance <= .20)
                    & (np.linalg.norm(vel-held_velocity, axis=1) < .30) & (gap <= np.asarray(cfg.held_gap_max))
                    & (opening_command is None or not bool(opening_command)))
        self.held_count = np.where(held_now, self.held_count+1, 0)
        self.held_age = np.where(held_now, 0, self.held_age+1)
        self.held |= self.held_count >= 5
        self.held_gap = np.where(held_now, np.minimum(self.held_gap, gap), self.held_gap)
        opening = (gap >= self.held_gap+.002) & (opening_command is None or bool(opening_command))
        # Full jaw opening can occur after the object's upward apex. Use an
        # opening increment plus ACTUAL gravity-consistent motion, not jaw width
        # alone. The apex/flight/basket thresholds themselves remain unchanged.
        ballistic = np.zeros(3, bool)
        if self.last_velocity is not None:
            expected_velocity = flight_state(cfg, np.zeros_like(vel), self.last_velocity, cfg.dt)[1]
            ballistic = np.linalg.norm(vel-expected_velocity, axis=1)/cfg.dt < 3.
        upward_open = self.held & (self.held_age <= 5) & ~self.released_up & opening & (vel[:, 2] > .15) & ballistic
        self.release_xyz[upward_open] = pos[upward_open]
        self.apex[upward_open] = pos[upward_open, 2]
        self.released_up |= upward_open
        self.apex = np.where(self.released_up, np.maximum(self.apex, pos[:, 2]), self.apex)
        if self.last_velocity is not None:
            relative_shift = np.linalg.norm(pos-self.release_xyz, axis=1) > .04
            flying = self.released_up & relative_shift & opening & ballistic
            self.flight_run = np.where(flying, self.flight_run+1, 0)
            self.flight_count = np.maximum(self.flight_count, self.flight_run)
        roots = pos if root_positions is None else np.asarray(root_positions)
        inside = (np.all(np.abs(roots[:, :2]-np.asarray(cfg.basket_xyz[:2])) <= cfg.basket_half_xy, axis=1)
                  & (roots[:, 2] >= cfg.table_z) & (roots[:, 2] <= cfg.table_z+.15))
        still = np.linalg.norm(vel, axis=1) <= .10
        self.settled_count = np.where(inside & still, self.settled_count+1, 0)
        apex_rise = self.apex-self.release_xyz[:, 2]
        self.success = (self.held & self.released_up & (apex_rise >= .01)
                        & (self.flight_count >= 3) & (self.settled_count >= cfg.settle_steps))
        self.last_velocity = vel.copy()
        return bool(self.success.all())

    def report(self):
        rise = self.apex-self.release_xyz[:, 2]
        return {"task": "TaskE-Toss-v1", "evaluation_only_truth": True,
                "evaluation_version": "COM-damped-freeflight-v4", "flight_position_basis": "mass center",
                "flight_linear_damping": self.cfg.object_linear_damping,
                "held_gap_limits_m": list(self.cfg.held_gap_max),
                "basket_position_basis": "actor root",
                "evidence": "kinematic held/flight proxy; no contact-force proof",
                "throw_success": bool(self.success.all()), "per_object_success": self.success.tolist(),
                "held_evidence": self.held.tolist(), "upward_release": self.released_up.tolist(),
                "apex_rise_since_detected_release_m": [float(x) if np.isfinite(x) else None for x in rise],
                "minimum_apex_rise_m": .01,
                "release_height_basis": "first gravity-consistent upward observation, not exact contact separation",
                "gravity_consistent_flight_frames": self.flight_count.tolist(),
                "settled_frames": self.settled_count.tolist()}
