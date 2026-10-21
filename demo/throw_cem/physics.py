"""Quintic motion, actual open-command timing, and ballistic landing prior."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


def physics_contract(cfg):
    keys = ("dt", "physics_dt", "object_linear_damping", "planning_min_remaining_apex", "release_ee_xyz", "pre_throw_delta", "throw_tilt_delta_deg", "position_lead_seconds", "joint_response_alpha", "throw_primitive", "gripper_delay_prior", "banana_delay_prior", "follow_through",
            "brake_seconds", "table_z", "basket_xyz", "basket_outer_half_xy", "basket_rim_z", "held_rim_clearance", "root_xyz", "object_com_offsets", "grasp_offsets", "banana_pinch_distance", "banana_keep_grasp_orientation", "banana_held_local_z_range", "banana_held_lateral_max", "lower", "upper",
            "joint_swing_duration", "joint_swing_lower", "joint_swing_upper", "lift_clearance",
            "max_joint_speed", "max_joint_acceleration", "carry_max_joint_speed", "carry_max_joint_acceleration")
    return json.loads(json.dumps({**{key: getattr(cfg, key) for key in keys}, "braking_mode": "bounded_joint_after_detach_v2",
                                 "flight_model": "discrete_linear_drag_v1"}))


class Swing:
    def __init__(self, start, end, velocity, duration, start_velocity=None):
        self.start, self.end, self.velocity = (np.asarray(x, float) for x in (start, end, velocity))
        self.duration = float(duration)
        self.start_velocity = np.zeros(3) if start_velocity is None else np.asarray(start_velocity, float)
        t = self.duration
        matrix = np.array([[t**3, t**4, t**5], [3*t**2, 4*t**3, 5*t**4], [6*t, 12*t**2, 20*t**3]])
        self.coeff = np.linalg.solve(matrix, np.stack((self.end-self.start-self.start_velocity*t, self.velocity-self.start_velocity, np.zeros(3))))

    def state(self, time):
        t = np.clip(float(time), 0., self.duration)
        pos = self.start+self.start_velocity*t + np.array([t**3, t**4, t**5]) @ self.coeff
        vel = self.start_velocity+np.array([3*t**2, 4*t**3, 5*t**4]) @ self.coeff
        if time > self.duration:
            pos = self.end+(time-self.duration)*self.velocity; vel = self.velocity
        return pos, vel


class ThrowMotion:
    """C2-continuous acceleration, follow-through and braking; no abrupt hold."""
    def __init__(self, start, end, velocity, duration, cfg):
        self.swing = Swing(start, end, velocity, duration)
        self.duration = duration
        self.brake_start = duration+cfg.follow_through
        self.end_time = self.brake_start+cfg.brake_seconds
        brake_xyz = np.asarray(end)+np.asarray(velocity)*cfg.follow_through
        self.brake = Swing(brake_xyz, brake_xyz+.5*np.asarray(velocity)*cfg.brake_seconds,
                           np.zeros(3), cfg.brake_seconds, start_velocity=velocity)

    def state(self, time):
        if time <= self.brake_start:
            return self.swing.state(time)
        if time <= self.end_time:
            return self.brake.state(time-self.brake_start)
        return self.brake.end.copy(), np.zeros(3)


def ballistic_landing(position, velocity, target_z, gravity=9.81):
    position, velocity = np.asarray(position), np.asarray(velocity)
    discriminant = velocity[2]**2 + 2*gravity*(position[2]-target_z)
    if not np.isfinite(discriminant) or discriminant < 0:
        raise ValueError("Projectile cannot cross target plane")
    time = (velocity[2]+np.sqrt(discriminant))/gravity
    if time <= .04 or velocity[2]-gravity*time >= 0:
        raise ValueError("Need a future descending basket crossing")
    xy = position[:2]+velocity[:2]*time
    return xy, float(time)


def flight_state(cfg, position, velocity, time):
    """Native semi-implicit free-flight prior, including original linear drag.

    At each physics step: v=(v+g*h)*(1-d*h); x=x+v*h.
    Closed-form integer steps, linear substep interpolation. Negative times
    invert this SAME recurrence for effective release labels. No contact model.
    """
    p, v = np.asarray(position, float), np.asarray(velocity, float)
    h = cfg.physics_dt; steps = float(time)/h
    n = int(np.floor(steps+1e-9)); fraction = steps-n
    gravity = np.array([0., 0., -9.81])
    a = 1-cfg.object_linear_damping*h
    if abs(1-a) < 1e-12:
        vn = v+n*h*gravity
        pn = p+n*h*v+.5*h*h*n*(n+1)*gravity
    else:
        steady = a*gravity*h/(1-a)
        factor = a**n
        vn = steady+factor*(v-steady)
        pn = p+h*((v-steady)*a*(1-factor)/(1-a)+n*steady)
    following = a*(vn+gravity*h)
    return pn+fraction*h*following, vn+fraction*(following-vn)


def flight_apex_rise(cfg, vz):
    if vz <= 0:
        return 0.
    h = cfg.physics_dt; a = 1-cfg.object_linear_damping*h
    if abs(1-a) < 1e-12:
        peak_step = vz/(9.81*h)
    else:
        steady = -a*9.81*h/(1-a)
        peak_step = np.log(-steady/(vz-steady))/np.log(a)
    times = [max(0, int(np.floor(peak_step)))*h, max(0, int(np.ceil(peak_step)))*h]
    return max(0., *(float(flight_state(cfg, np.zeros(3), [0., 0., vz], t)[0][2]) for t in times))


def projectile_landing(cfg, position, velocity, target_z):
    p, v = np.asarray(position, float), np.asarray(velocity, float)
    # Start at the apex, then find a descending plane crossing, never ascent.
    h = cfg.physics_dt; a = 1-cfg.object_linear_damping*h
    if v[2] <= 0:
        lo = 0.
    elif abs(1-a) < 1e-12:
        lo = max(0., np.floor(v[2]/(9.81*h))*h)
    else:
        steady = -a*9.81*h/(1-a)
        lo = max(0., np.floor(np.log(-steady/(v[2]-steady))/np.log(a))*h)
    if flight_state(cfg, p, v, lo)[0][2] < target_z:
        raise ValueError("Projectile cannot cross target plane")
    hi = lo+2.
    if flight_state(cfg, p, v, hi)[0][2] > target_z:
        raise ValueError("No bounded-time descending plane crossing")
    for _ in range(32):
        mid = (lo+hi)/2
        if flight_state(cfg, p, v, mid)[0][2] > target_z:
            lo = mid
        else:
            hi = mid
    if hi <= .04:
        raise ValueError("Need a future descending basket crossing")
    return flight_state(cfg, p, v, hi)[0][:2], hi


def check_release_margin(cfg, velocity, observation_aligned=False):
    if velocity[2] < .15 or flight_apex_rise(cfg, velocity[2]) < .01:
        raise ValueError("Driven model predicts insufficient upward release")
    # A calibrated world prediction already targets the first confirmed free-
    # flight frame. Advancing it by another control frame double-counts latency.
    first_observed_velocity = (velocity if observation_aligned else
                               flight_state(cfg, np.zeros(3), velocity, cfg.dt)[1])
    if flight_apex_rise(cfg, first_observed_velocity[2]) < cfg.planning_min_remaining_apex:
        raise ValueError("Insufficient post-release observed apex margin")


class ReleaseResidual:
    """Optional per-object linear corrections to release position and velocity.

    Absent weights mean an explicitly uncalibrated physics prior, not trained WM.
    File format: coefficients[3,6,8], delta_xyz+delta_velocity from [1,theta4,offset3].
    """
    def __init__(self, path=None, cfg=None):
        self.coefficients = np.zeros((3, 6, 8)); self.calibrated = False
        self.bounds = None
        if path:
            payload = json.loads(Path(path).read_text())
            if payload.get("format") != "atec_throw_release_residual_v1" or not payload.get("calibrated", False):
                raise ValueError("Residual checkpoint must be calibrated for throwing")
            coeff = np.asarray(payload["coefficients"], float)
            if coeff.shape != (3, 6, 8) or not np.isfinite(coeff).all():
                raise ValueError("Invalid residual parameters")
            if cfg is None or payload.get("physics") != physics_contract(cfg):
                raise ValueError("Residual physics contract does not match this controller")
            bounds = np.asarray(payload["feature_bounds"], float)
            if bounds.shape != (3, 7, 2) or not np.isfinite(bounds).all() or np.any(bounds[..., 0] > bounds[..., 1]):
                raise ValueError("Invalid residual support bounds")
            self.bounds = bounds
            self.coefficients = coeff; self.calibrated = True

    def correction(self, obj, theta, offset, q=None):
        features = np.r_[theta, offset]
        if self.bounds is not None and (np.any(features < self.bounds[obj, :, 0]) or np.any(features > self.bounds[obj, :, 1])):
            return np.zeros(6)  # no uncalibrated extrapolation
        value = self.coefficients[obj] @ np.r_[1., features]
        return np.clip(value, -np.r_[np.full(3, .03), np.full(3, .25)], np.r_[np.full(3, .03), np.full(3, .25)])


@dataclass
class ThrowPrediction:
    swing: Swing
    release_position: np.ndarray
    release_velocity: np.ndarray
    landing_xy: np.ndarray
    entry_xy: np.ndarray
    flight_seconds: float
    open_step: int
    release_delay: float


def predict_throw(cfg, obj, start, offset, theta, residual):
    vh, vz, duration, lead = np.asarray(theta, float)
    endpoint = np.asarray(cfg.release_ee_xyz)
    direction = np.asarray(cfg.basket_xyz[:2])-(endpoint+offset)[:2]
    distance = np.linalg.norm(direction)
    # Upward free flight distinguishes a throw from a placement. Requiring
    # 5 cm horizontal travel wrongly excluded vertical tosses near the basket.
    direction = direction/distance if distance >= 1e-4 else np.zeros(2)
    swing = ThrowMotion(start, endpoint, np.r_[direction*vh, vz], duration, cfg)
    open_step = max(0, int(np.ceil((duration-lead)/cfg.dt)))
    detach_time = open_step*cfg.dt+cfg.release_delay(obj)
    pos, velocity = swing.state(detach_time)
    correction = residual.correction(obj, theta, offset)
    pos = pos+offset+correction[:3]; velocity = velocity+correction[3:]
    if velocity[2] < .15 or flight_apex_rise(cfg, velocity[2]) < .01:
        raise ValueError("Must release upward, not simply drop")
    landing, time = projectile_landing(cfg, pos, velocity, cfg.basket_xyz[2])
    entry, _ = projectile_landing(cfg, pos, velocity, cfg.table_z+.15)
    return ThrowPrediction(swing, pos, velocity, landing, entry, time, open_step, cfg.release_delay(obj))


def predict_driven_release(cfg, obj, q, offset, theta, nominal, targets, kin, joint_model, residual):
    """Release from learned ACTUAL joint motion, with a rigid held-object proxy.

    Accounts for tool rotation and drive lag; no real object truth is read.
    Actual slip, opening delay and free-flight residuals remain to be calibrated.
    """
    rollout = joint_model.rollout(q, targets)
    states = np.vstack((q[:6], rollout))
    local_offset = kin.forward(q)[:3, :3].T@offset
    xyz = []
    for state in states:
        pose = kin.forward(state)
        xyz.append(pose[:3, 3]+pose[:3, :3]@local_offset)
    xyz = np.asarray(xyz)
    index = nominal.open_step+cfg.release_delay(obj)/cfg.dt
    lo = int(np.floor(index)); fraction = index-lo
    if lo < 1 or lo+2 >= len(states):
        raise ValueError("Insufficient modeled motion around detachment")
    pos = xyz[lo]*(1-fraction)+xyz[lo+1]*fraction
    v0 = (xyz[lo+1]-xyz[lo-1])/(2*cfg.dt)
    v1 = (xyz[lo+2]-xyz[lo])/(2*cfg.dt)
    velocity = v0*(1-fraction)+v1*fraction
    correction = residual.correction(obj, theta, offset, q=q)
    pos = pos+correction[:3]; velocity = velocity+correction[3:]
    check_release_margin(cfg, velocity, observation_aligned=getattr(residual, "observation_aligned", False))
    landing, time = projectile_landing(cfg, pos, velocity, cfg.basket_xyz[2])
    entry, _ = projectile_landing(cfg, pos, velocity, cfg.table_z+.15)
    return ThrowPrediction(nominal.swing, pos, velocity, landing, entry, time, nominal.open_step, nominal.release_delay), rollout
