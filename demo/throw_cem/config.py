from dataclasses import dataclass
import numpy as np

DEFAULT_Q = np.array([0., 1.2, -1.5, 0., 1.2, 0., .035, -.035])
HOME_Q = np.array([0., .924525, -1.514983, 0., 1.20, 0., .035, -.035])
ARM_LOWER = np.deg2rad([-150.00035, 0., -169.99657, -99.98113, -69.90085, -120.00027]) + .005
ARM_UPPER = np.deg2rad([124.21724, 179.90874, 0., 99.98113, 69.90085, 120.00027]) - .005

MOTION_FIELDS = frozenset({"release_ee_xyz", "pre_throw_delta", "throw_tilt_delta_deg", "lower", "upper",
                         "gripper_delay_prior", "banana_delay_prior", "follow_through", "planning_min_remaining_apex",
                         "banana_keep_grasp_orientation", "banana_pinch_distance", "lift_clearance", "joint_swing_duration",
                         "joint_swing_lower", "joint_swing_upper", "collection_wrist_q6", "static_hand_clearance",
                         "mustard_body_grasp", "normalize_throw_stance", "sugar_centered_grasp", "pre_grasp_clearance"})


@dataclass(frozen=True)
class ThrowConfig:
    # Static task calibration, not live object truth. Derived from Task-E geometry.
    table_z: float = .826642648
    root_xyz: tuple = (1.404253903, 0., .826642648)
    basket_xyz: tuple = (1.08, -.30, .926642648)
    basket_half_xy: tuple = (.20, .11)
    basket_outer_half_xy: tuple = (.237356, .158346)  # static original-asset visual AABB
    basket_rim_z: float = .97982264
    held_rim_clearance: float = .10  # conservative point proxy, not mesh collision proof
    release_ee_xyz: tuple = (1.16, -.23, 1.176642648)
    pre_throw_delta: tuple = (.04, .08, -.07)
    throw_tilt_delta_deg: float = -10.
    # Measured lag ~.082..093 s. Lead remains OFF: .085 produced no bounded
    # command-feasible plan in the focused check. Do not bypass constraints.
    position_lead_seconds: float = 0.
    joint_response_alpha: tuple = (1., 1., 1., 1., 1., 1.)  # instantaneous legacy prior
    throw_primitive: str = "arm_swing"  # fixed wrist angles, not fixed world tool orientation
    joint_swing_duration: float = .75
    joint_swing_lower: tuple = (-.45, -.65, -.65, .40)
    joint_swing_upper: tuple = (.90, .65, .65, .80)
    lift_clearance: float = .12
    collection_wrist_q6: float | None = None  # optional static wrist alignment, RGB/proprio inputs only
    object_half_xy: tuple = ((.050, .044), (.050, .030), (.100, .040))
    # Static asset-local actor->COM offsets captured from the original assets.
    # RGB deployment uses constants, never a live simulator object query.
    object_com_offsets: tuple = ((.0004768353537656367, .000664581370074302, -.0002919909602496773),
                                 (.0002076068049063906, .01623924821615219, .00018429271585773677),
                                 (-.004321056883782148, .00022860686294734478, -.0004944115644320846))
    grasp_offsets: tuple = (.140, .165, .090)
    banana_pinch_distance: float = .12  # wrist-link origin to finger-pad grasp point
    banana_keep_grasp_orientation: bool = True
    static_hand_clearance: bool = False  # experimental grasp selector; preserve validated baseline by default
    mustard_body_grasp: bool = False  # experimental COM-centered pads; legacy and banana unchanged
    sugar_centered_grasp: bool = False  # optional pad-centered box grasp; legacy default unchanged
    normalize_throw_stance: bool = False  # at most one bounded training-stance carry after a planning rejection
    # Successful holds span 116..129.5 mm; the known pad-tip slip was ~140 mm.
    banana_held_local_z_range: tuple = (.095, .132)
    banana_held_lateral_max: float = .030
    pre_grasp_clearance: float = .12
    dt: float = .02
    physics_dt: float = .005
    object_linear_damping: float = 2.0  # original Task-E assets, not an environment modification
    action_scale: float = .5
    # Conservative planner limits; original simulator actuator settings stay intact.
    # Target-trajectory caps, NOT a measured safe hardware operating envelope.
    # 1.5/8 excluded the upward motions in the bounded-DH calculation.
    max_joint_speed: float = 3.
    max_joint_acceleration: float = 20.
    carry_max_joint_speed: float = 1.5
    carry_max_joint_acceleration: float = 8.
    max_tracking_error: float = .15
    tracking_error_steps: int = 3
    pose_position_tolerance: float = .008
    pose_rotation_tolerance: float = .18
    # v_horizontal, v_upward, swing_duration, early_open_seconds
    lower: tuple = (.00, .45, .40, .00)
    upper: tuple = (.70, .90, .85, .12)
    gripper_delay_prior: float = .06
    # Seed65: opening command at swing cursor 21; gravity-consistent free flight
    # begins at cursor 23. Model that measured two-control-frame jaw delay.
    banana_delay_prior: float = .04
    # The calibrated joint prior over-predicts object vertical speed: seed66
    # passed at 1.05 m/s predicted, while seed67 failed at 0.82 m/s. Require
    # enough predicted rise after the first observable free-flight frame.
    planning_min_remaining_apex: float = .024  # evaluator threshold remains .010 m
    follow_through: float = .02  # joint trajectory also covers per-object nominal detachment
    brake_seconds: float = .30
    population: int = 48
    elites: int = 10
    iterations: int = 5
    planner_seed: int = 43
    object_order: tuple = (2, 1, 0)  # zero-based; subset only for targeted calibration
    collision_margin: float = .02
    confidence_min: float = .70
    held_gap_max: tuple = (.065, .065, .068)  # banana held at 65.1..66.3 mm; full-open is 70 mm
    open_gap_min: float = .068
    max_steps: int = 6000
    stage_timeout: int = 350
    settle_steps: int = 25
    max_flight_steps: int = 150

    def release_delay(self, obj):
        return self.banana_delay_prior if obj == 2 else self.gripper_delay_prior

    def __post_init__(self):
        if self.collection_wrist_q6 is not None and not ARM_LOWER[5] < self.collection_wrist_q6 < ARM_UPPER[5]:
            raise ValueError("Collection wrist angle must stay inside native joint limits")
        if self.throw_primitive not in ("arm_swing", "fixed_tool", "joint_swing"):
            raise ValueError("Unknown throw primitive")
        if np.asarray(self.object_com_offsets).shape != (3, 3) or not np.isfinite(self.object_com_offsets).all():
            raise ValueError("Three finite static object COM offsets required")
        if np.asarray(self.held_gap_max).shape != (3,) or any(not 0 < gap < .07 for gap in self.held_gap_max):
            raise ValueError("Per-object hold gaps must remain below full opening")
        if (len(self.banana_held_local_z_range) != 2 or
                not 0 < self.banana_held_local_z_range[0] < self.banana_held_local_z_range[1] < .20 or
                not 0 < self.banana_held_lateral_max <= .05):
            raise ValueError("Invalid banana pad-depth guard")
        if len(self.joint_response_alpha) != 6 or not np.isfinite(self.joint_response_alpha).all() or any(not 0 < a <= 1 for a in self.joint_response_alpha):
            raise ValueError("Six finite joint response coefficients required")
        if not self.object_order or len(set(self.object_order)) != len(self.object_order) or any(i not in (0, 1, 2) for i in self.object_order):
            raise ValueError("Object order must be a nonempty unique subset of (0,1,2)")
        if not 0 < self.elites <= self.population or not 1 <= self.iterations <= 10:
            raise ValueError("Invalid finite CEM budget")
        if self.dt <= 0 or self.max_joint_speed <= 0 or self.max_joint_acceleration <= 0:
            raise ValueError("Invalid execution limits")
        if not 0 < self.physics_dt <= self.dt or not 0 <= self.object_linear_damping*self.physics_dt < 1:
            raise ValueError("Invalid discrete flight parameters")
        if not 0 <= self.planning_min_remaining_apex <= .10 or min(self.gripper_delay_prior, self.banana_delay_prior) < 0:
            raise ValueError("Invalid release timing / height margin")
        if len(self.lower) != 4 or any(a >= b for a, b in zip(self.lower, self.upper)):
            raise ValueError("Four bounded throw parameters required")
        if (not .4 <= self.joint_swing_duration <= 1.2 or not .10 <= self.lift_clearance <= .25
                or len(self.joint_swing_lower) != 4 or len(self.joint_swing_upper) != 4
                or any(a >= b for a,b in zip(self.joint_swing_lower, self.joint_swing_upper))
                or not 0 < self.joint_swing_lower[3] < self.joint_swing_upper[3] < 1):
            raise ValueError("Invalid bounded joint swing / lift parameters")
