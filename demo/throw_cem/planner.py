from dataclasses import dataclass
import numpy as np
from .config import ARM_LOWER, ARM_UPPER, DEFAULT_Q
from .physics import ReleaseResidual, ThrowPrediction, predict_throw, predict_driven_release, check_release_margin
from .joint_model import JointResponse


class NoFeasibleThrow(RuntimeError):
    pass


@dataclass
class JointTrajectory:
    targets: np.ndarray
    reference: np.ndarray  # desired physical motion, separate from lead commands


@dataclass
class ThrowPlan:
    theta: np.ndarray
    targets: np.ndarray  # absolute eight joint targets; not env action deltas
    prediction: object
    cost: float
    evaluated_candidates: int
    tracking_targets: np.ndarray


class ThrowPlanner:
    """CEM over four motion parameters; no ACT/actor initialization or weights."""
    def __init__(self, cfg, kinematics, residual=None, seed=43, joint_model=None):
        self.cfg, self.kin = cfg, kinematics
        self.residual = residual or ReleaseResidual()
        self.rng = np.random.default_rng(seed)
        self.joint_model = joint_model or JointResponse(cfg)

    def nominal(self, offset):
        position = np.asarray(self.cfg.release_ee_xyz)+offset
        driven = not np.allclose(self.joint_model.alpha, 1.) or self.cfg.throw_primitive == "arm_swing"
        vz = .85 if self.cfg.planning_min_remaining_apex > 0 else (.60 if driven else .45)
        from .physics import projectile_landing
        _, flight = projectile_landing(self.cfg, position, [0., 0., vz], self.cfg.basket_xyz[2])
        vh = np.linalg.norm(np.asarray(self.cfg.basket_xyz[:2])-position[:2])/flight
        return np.clip([vh, vz, .65 if driven else .50, .0 if driven else self.cfg.gripper_delay_prior], self.cfg.lower, self.cfg.upper)

    def joint_trajectory(self, q, rotation, prediction, duration):
        cfg = self.cfg
        # Preserve Cartesian release kinematics until even the latest nominal
        # detachment. Afterwards the hand no longer needs fixed orientation:
        # Cartesian braking extended it outside the bounded robot workspace.
        # After nominal detachment, the complete original swing endpoint is
        # unnecessary. Early opening must also advance braking; otherwise CEM's
        # timing parameter cannot recover stopping space near extension limits.
        follow_end = max(2*cfg.dt, prediction.open_step*cfg.dt+prediction.release_delay+cfg.follow_through)
        count = int(np.ceil(follow_end/cfg.dt))
        seed = np.asarray(q[:6], float).copy(); joints = []
        for index in range(count):
            pos, _ = prediction.swing.state((index+1)*cfg.dt)
            if pos[2] < cfg.table_z+.08:
                raise ValueError("Swing too close to support plane")
            if cfg.throw_primitive == "arm_swing":
                seed = self.kin.solve_position(pos, seed, max_iterations=35)
            else:
                seed = self.kin.solve(pos, rotation, seed, max_iterations=35,
                                      position_tolerance=1e-6, rotation_tolerance=1e-5)
            joints.append(seed.copy())
        arm = np.asarray(joints)
        terminal_velocity = (arm[-1]-arm[-2])/cfg.dt
        arm = np.vstack((arm, self.joint_brake(arm[-1], terminal_velocity)))
        desired_velocity = np.diff(np.vstack((q[:6], arm)), axis=0)/cfg.dt
        command = arm+cfg.position_lead_seconds*desired_velocity
        # Finite lead compensation of measured actuator lag, standard position
        # actions only. Both the reference and actual commands stay bounded.
        for values in (arm, command):
            if not np.isfinite(values).all() or np.any(values < ARM_LOWER) or np.any(values > ARM_UPPER):
                raise ValueError("Joint limits violated")
            velocity = np.diff(np.vstack((q[:6], values)), axis=0)/cfg.dt
            acceleration = np.diff(np.vstack((np.zeros(6), velocity, np.zeros(6))), axis=0)/cfg.dt
            if np.max(np.abs(velocity)) > cfg.max_joint_speed:
                raise ValueError("Throw target velocity limit")
            if np.max(np.abs(acceleration)) > cfg.max_joint_acceleration:
                raise ValueError("Throw target acceleration limit")
        targets = np.tile(DEFAULT_Q, (len(arm), 1)); targets[:, :6] = arm
        targets[:, 6:] = [-.015, .015]
        targets[prediction.open_step:, 6:] = [.035, -.035]
        reference = targets.copy()
        targets[:, :6] = command
        return JointTrajectory(targets, reference)

    def joint_brake(self, q, velocity):
        cfg = self.cfg
        speed = np.abs(velocity)
        available = np.where(velocity >= 0, ARM_UPPER-q, q-ARM_LOWER)-.005
        max_time = 2*np.maximum(available, 0)/np.maximum(speed, 1e-9)
        times = np.minimum(cfg.brake_seconds, .95*max_time)
        minimum = 1.5*speed/cfg.max_joint_acceleration+cfg.dt
        if np.any((speed > 1e-6) & (times < minimum)):
            raise ValueError("Insufficient bounded stopping distance")
        times = np.where(speed <= 1e-6, cfg.brake_seconds, times)
        samples = []
        for index in range(int(np.ceil(cfg.brake_seconds/cfg.dt))):
            s = np.clip((index+1)*cfg.dt/times, 0, 1)
            # Integrate v=v0*(1-3*s^2+2*s^3): zero initial/final acceleration.
            target = q+velocity*times*(s-s**3+.5*s**4)
            if self.kin.forward(target)[2, 3] < cfg.table_z+.08:
                raise ValueError("Joint braking too close to support plane")
            samples.append(target)
        return np.asarray(samples)

    def plan(self, q, obj, object_offset):
        if self.cfg.throw_primitive == "joint_swing":
            return self.plan_joint_swing(q, obj, object_offset)
        cfg = self.cfg
        rejections = {}
        current = self.kin.forward(q); start, rotation = current[:3, 3], current[:3, :3]
        offset = np.asarray(object_offset, float)
        if offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > .20:
            raise NoFeasibleThrow("Unreliable object-to-gripper offset")
        safe_half = np.asarray(cfg.basket_half_xy)-np.asarray(cfg.object_half_xy[obj])-cfg.collision_margin
        if np.any(safe_half <= 0):
            raise NoFeasibleThrow("No clearance for this object")
        lower, upper = np.asarray(cfg.lower), np.asarray(cfg.upper)
        mean = self.nominal(offset); std = (upper-lower)*.20
        best, evaluated = None, 0
        driven = not np.allclose(self.joint_model.alpha, 1.) or cfg.throw_primitive == "arm_swing"
        def score(prediction, theta):
            error = (prediction.landing_xy-np.asarray(cfg.basket_xyz[:2]))/safe_half
            outside = np.maximum(np.abs(error)-1, 0)
            entry_error = (prediction.entry_xy-np.asarray(cfg.basket_xyz[:2]))/safe_half
            entry_outside = np.maximum(np.abs(entry_error)-1, 0)
            return float(error@error+40*(outside@outside)+40*(entry_outside@entry_outside)+.02*(theta[0]**2+theta[1]**2)+.02*theta[2])
        for _ in range(cfg.iterations):
            candidates = np.clip(self.rng.normal(mean, std, (cfg.population, 4)), lower, upper)
            candidates[0] = mean
            if best is not None:
                candidates[1] = best.theta
            ranked = []
            for theta in candidates:
                evaluated += 1
                try:
                    prediction = predict_throw(cfg, obj, start, offset, theta, ReleaseResidual() if driven else self.residual)
                    if not driven:
                        check_release_margin(cfg, prediction.release_velocity)
                    cost = score(prediction, theta)
                    # Root-centre margin is not full rigid-body collision validation.
                    ranked.append((cost, theta.copy(), prediction))
                except ValueError as error:
                    rejections[str(error)] = rejections.get(str(error), 0)+1
                    continue
            ranked.sort(key=lambda item: item[0])
            feasible = []
            for cost, theta, prediction in (ranked if driven else ranked[:cfg.elites]):
                try:
                    trajectory = self.joint_trajectory(q, rotation, prediction, theta[2])
                    tracking = trajectory.reference
                    if driven:
                        prediction, rollout = predict_driven_release(cfg, obj, q, offset, theta, prediction,
                                                                     trajectory.targets, self.kin, self.joint_model, self.residual)
                        tracking = trajectory.reference.copy(); tracking[:, :6] = rollout
                        cost = score(prediction, theta)
                except ValueError as error:
                    rejections[str(error)] = rejections.get(str(error), 0)+1
                    continue
                feasible.append((cost, theta))
                inside = np.all(np.abs(prediction.landing_xy-np.asarray(cfg.basket_xyz[:2])) <= safe_half)
                inside &= np.all(np.abs(prediction.entry_xy-np.asarray(cfg.basket_xyz[:2])) <= safe_half)
                if inside and (best is None or cost < best.cost):
                    best = ThrowPlan(theta, trajectory.targets, prediction, cost, evaluated, tracking)
            if feasible:
                selected = np.asarray([theta for _, theta in sorted(feasible, key=lambda row: row[0])[:cfg.elites]])
                mean = np.mean(selected, axis=0)
                std = np.maximum(np.std(selected, axis=0), .03*(upper-lower))
            else:
                # No silent fallback to an unsafe trajectory or an actor policy.
                std = np.minimum(std*1.3, .4*(upper-lower))
        if best is None:
            raise NoFeasibleThrow(f"CEM found no joint-feasible upward landing plan within budget: {rejections}")
        best.evaluated_candidates = evaluated
        return best

    def joint_swing_trajectory(self, q, theta, obj):
        """Three bounded joint displacements + opening fraction. End at rest.

        The same four-feature residual layout is retained, but the explicit
        primitive/physics contract prevents use of Cartesian-param weights.
        """
        cfg = self.cfg; duration = cfg.joint_swing_duration
        delta = np.asarray(theta[:3], float)
        count = int(np.ceil(duration/cfg.dt))
        s = np.clip(np.arange(1, count+1)*cfg.dt/duration, 0., 1.)
        blend = 10*s**3-15*s**4+6*s**5
        targets = np.tile(q, (count+5, 1))
        # A position drive can settle a fraction of a milliradian beyond the
        # planner's 5mrad inset while still inside the native joint range.
        # Preserve measured q as rollout/rate-check start; only COMMAND the
        # fixed wrist back into the existing planner bounds. Never widen them.
        wrist = np.clip(q[3:6],ARM_LOWER[3:6],ARM_UPPER[3:6])
        if np.max(np.abs(wrist-q[3:6]))>.005:
            raise ValueError("Measured wrist exceeds the native safety reserve")
        targets[:,3:6] = wrist
        targets[:count, :3] = q[:3]+blend[:, None]*delta
        targets[count:, :3] = q[:3]+delta
        targets[:, 6:] = [-.015, .015]
        opened = int(np.ceil(theta[3]*duration/cfg.dt))
        targets[opened:, 6:] = [.035, -.035]
        arm = targets[:, :6]
        if not np.isfinite(arm).all() or np.any(arm < ARM_LOWER) or np.any(arm > ARM_UPPER):
            raise ValueError("Joint limits violated")
        v = np.diff(np.vstack((q[:6], arm)), axis=0)/cfg.dt
        acceleration = np.diff(np.vstack((np.zeros(6), v, np.zeros(6))), axis=0)/cfg.dt
        if np.max(np.abs(v)) > cfg.max_joint_speed or np.max(np.abs(acceleration)) > cfg.max_joint_acceleration:
            raise ValueError("Joint swing rate limit")
        for state in arm:
            if self.kin.forward(state)[2, 3] < cfg.table_z+.08:
                raise ValueError("Joint swing too close to support plane")
        nominal = ThrowPrediction(None, np.zeros(3), np.zeros(3), np.zeros(2), np.zeros(2), 0., opened, cfg.release_delay(obj))
        return JointTrajectory(targets, targets.copy()), nominal

    def plan_joint_swing(self, q, obj, object_offset):
        cfg = self.cfg; offset = np.asarray(object_offset, float)
        if offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > .20:
            raise NoFeasibleThrow("Unreliable object-to-gripper offset")
        safe_half = np.asarray(cfg.basket_half_xy)-np.asarray(cfg.object_half_xy[obj])-cfg.collision_margin
        if np.any(safe_half <= 0):
            raise NoFeasibleThrow("No clearance for this object")
        lower = np.asarray(cfg.joint_swing_lower, float).copy(); upper = np.asarray(cfg.joint_swing_upper, float).copy()
        lower[:3] = np.maximum(lower[:3], ARM_LOWER[:3]-q[:3])
        upper[:3] = np.minimum(upper[:3], ARM_UPPER[:3]-q[:3])
        if getattr(self.residual, "world_model", False):
            try:
                model_lower, model_upper = self.residual.candidate_bounds(obj, q, offset)
            except ValueError as error:
                raise NoFeasibleThrow(str(error)) from error
            lower = np.maximum(lower, model_lower)
            upper = np.minimum(upper, model_upper)
        if np.any(lower >= upper):
            raise NoFeasibleThrow("No bounded joint swing domain")
        tool = self.kin.forward(q); local = tool[:3, :3].T@offset
        start_point = tool[:3, 3]+offset
        jac = np.empty((3, 3))
        for j in range(3):
            other = q.copy(); other[j] += 1e-5; pose = self.kin.forward(other)
            jac[:, j] = (pose[:3, 3]+pose[:3, :3]@local-start_point)/1e-5
        direction = np.asarray(cfg.basket_xyz[:2])-start_point[:2]
        direction /= max(np.linalg.norm(direction), 1e-6)
        speed = np.linalg.lstsq(jac, np.r_[direction*.75, .85], rcond=None)[0]
        mean = np.clip(np.r_[speed*cfg.joint_swing_duration/1.875, .60], lower, upper)
        if (obj == 2 or (obj == 1 and cfg.mustard_body_grasp)
                or (obj == 0 and cfg.sugar_centered_grasp)):
            # The unconstrained Jacobian solution saturates shoulder pitch and
            # then puts the elbow in the wrong direction. Seed the second arm
            # displacement with the actual upward Jacobian signs; CEM still
            # scores every candidate on predicted body flight. The same arm
            # direction is needed for the optional upper-body bottle grasp;
            # the unconstrained linear seed also misses its high-arc proposals.
            # The optional box grasp has the same elbow-sign failure. This
            # changes only the initial proposal, never feasibility or budget.
            mean[:3] = np.clip([max(mean[0], .65), min(mean[1], -.45), min(mean[2], -.40)], lower[:3], upper[:3])
        std = .25*(upper-lower); best = None; evaluated = 0; rejections = {}
        proposal_method = getattr(self.residual,"candidate_proposals",None)
        proposals = proposal_method(obj,q,offset) if proposal_method else np.empty((0,4))
        proposal_count = min(4,len(proposals),max(0,cfg.population-2))
        for _ in range(cfg.iterations):
            candidates = np.clip(self.rng.normal(mean, std, (cfg.population, 4)), lower, upper)
            candidates[0] = mean
            if best is not None:
                candidates[1] = best.theta
            # Replace random slots, never add evaluations or directly execute a
            # demonstration. All proposals still need bounded world scoring.
            if proposal_count:
                candidates[2:2+proposal_count] = np.clip(proposals[:proposal_count],lower,upper)
            feasible = []
            for theta in candidates:
                evaluated += 1
                try:
                    trajectory, nominal = self.joint_swing_trajectory(q, theta, obj)
                    prediction, rollout = predict_driven_release(cfg, obj, q, offset, theta, nominal,
                                                                 trajectory.targets, self.kin, self.joint_model, self.residual)
                    # Point clearance only; not mesh collision validation.
                    held_path = []
                    for state in rollout[:nominal.open_step]:
                        pose = self.kin.forward(state)
                        held_path.append(pose[:3, 3]+pose[:3, :3]@local)
                    if any(p[2] < cfg.table_z+.06 for p in held_path):
                        raise ValueError("Held point too close to support plane")
                    for p in held_path:
                        near_rim = np.all(np.abs(p[:2]-np.asarray(cfg.basket_xyz[:2]))
                                          <= np.asarray(cfg.basket_outer_half_xy)+cfg.held_rim_clearance)
                        if near_rim and p[2] < cfg.basket_rim_z+cfg.held_rim_clearance:
                            raise ValueError("Held point lacks basket-rim clearance")
                    error = (prediction.landing_xy-np.asarray(cfg.basket_xyz[:2]))/safe_half
                    entry = (prediction.entry_xy-np.asarray(cfg.basket_xyz[:2]))/safe_half
                    cost = float(error@error+40*np.sum(np.maximum(np.abs(error)-1, 0)**2)
                                 +40*np.sum(np.maximum(np.abs(entry)-1, 0)**2)+.01*(theta[:3]@theta[:3]))
                    feasible.append((cost, theta))
                    if np.all(np.abs(error) <= 1) and np.all(np.abs(entry) <= 1) and (best is None or cost < best.cost):
                        tracking = trajectory.reference.copy(); tracking[:, :6] = rollout
                        best = ThrowPlan(theta.copy(), trajectory.targets, prediction, cost, evaluated, tracking)
                except ValueError as error:
                    rejections[str(error)] = rejections.get(str(error), 0)+1
            if feasible:
                selected = np.asarray([t for _,t in sorted(feasible, key=lambda r:r[0])[:cfg.elites]])
                mean = selected.mean(0); std = np.maximum(selected.std(0), .03*(upper-lower))
            else:
                std = np.minimum(std*1.3, .4*(upper-lower))
        if best is None:
            raise NoFeasibleThrow(f"CEM found no bounded joint-space upward landing plan: {rejections}")
        best.evaluated_candidates = evaluated
        return best
