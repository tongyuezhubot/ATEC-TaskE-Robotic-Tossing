"""Task E demo-collection constants.
"""

from atec_rl_lab.tasks.task_e.env_cfg import (
    BASKET_CENTER_X, BASKET_CENTER_Y,
    TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z, TABLE_HALF_X,
    BASKET_EXCL_HALF_X, BASKET_EXCL_HALF_Y,
)

__all__ = [
    "BASKET_CENTER_X", "BASKET_CENTER_Y",
    "TABLE_CENTER_X", "TABLE_CENTER_Y", "TABLE_TOP_Z", "TABLE_HALF_X",
    "BASKET_EXCL_HALF_X", "BASKET_EXCL_HALF_Y",
    # robot
    "EE_BODY_NAME", "ARM_JOINT_NAMES", "GRIPPER_JOINT_NAMES",
    "GRIPPER_OPEN_POS", "GRIPPER_CLOSE_POS", "ACTION_SCALE",
    # state machine
    "STEPS", "STATE_ORDER",
    # geometry
    "PRE_GRASP_CLEARANCE", "GRASP_Z_OFFSET", "GRASP_Z_OFFSETS",
    "GRASP_LONG_AXIS_LOCAL", "GRASP_LONG_AXIS_SIGN",
    "GRASP_ALIGN_X", "GRASP_ALIGN_Y", "GRASP_APPROACH_OFFSETS", "GRASP_TILT_Y_DEG",
    "GRASP_TARGET_X_OFFSETS",
    "CARRY_Z", "PLACE_HEIGHT",
    "RETRACT_POS_X", "RETRACT_POS_Y",
    "DEFAULT_PLACE_QUAT_W",
    # success check
    "BASKET_IN_X", "BASKET_IN_Y",
    # spawn regions
    "OBJ_SPAWN_X_MIN", "OBJ_SPAWN_X_MAX", "OBJ_SPAWN_Z", "OBJ_SPAWN_Y_BANDS",
    "OBJ_HALF_EXTENTS", "OBJ_BBOX_MARGIN",
    # camera
    "CAM_POS", "CAM_ROT", "CAM_H", "CAM_W",
    # actuator
    "ACT_STIFFNESS", "ACT_DAMPING", "ACT_EFFORT_LIMIT", "ACT_VEL_LIMIT",
    # warm-up
    "WARMUP_STEPS", "SETTLE_STEPS",
]

# ------------------------------------------------------------------ #
# Robot
# ------------------------------------------------------------------ #
EE_BODY_NAME        = "gripper_base"
ARM_JOINT_NAMES     = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
GRIPPER_OPEN_POS    = [0.035, -0.035]   # joint7, joint8
GRIPPER_CLOSE_POS   = [-0.015,  0.015]

# Must match ActionsCfg: scale=0.5, use_default_offset=True
#   env_action = (joint_target - default_joint_pos) / ACTION_SCALE
ACTION_SCALE = 0.5

# ------------------------------------------------------------------ #
# State-machine
# ------------------------------------------------------------------ #
STEPS: dict[str, int] = {
    "INIT":         100,
    "PRE_GRASP":    200,
    "ALIGN":        120,
    "REACH":        100,
    "DESCEND":       80,
    "CLOSE":         40,
    "LIFT":         160,
    "TRANSPORT":    200,
    "PLACE":         80,
    "OPEN":          60,
    "LIFT_RETRACT":  80,
    "RETRACT":       80,
}
STATE_ORDER = ["INIT", "PRE_GRASP", "ALIGN", "REACH", "DESCEND", "CLOSE", "LIFT",
               "TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT", "RETRACT"]

# ------------------------------------------------------------------ #
# Geometry
# ------------------------------------------------------------------ #
PRE_GRASP_CLEARANCE = 0.12   # metres above object before descent
GRASP_Z_OFFSET      = 0.09   # metres: gripper approach height above object centre

# The assets do not have the same upright height.  Keep roughly the same
# gripper-base clearance above each object's top surface as the known-good
# banana grasp instead of applying one centre-relative height to every asset.
# Values are metres above the live rigid-body centre.
GRASP_Z_OFFSETS: dict[int, float] = {
    1: 0.140,  # sugar box: finger-root clearance while retaining jaw overlap
    2: 0.165,  # mustard bottle: vertical half-extent ~= 0.096 m
    3: 0.090,  # banana: validated by the object-3 smoke demonstration
}

# Local asset axis that gripper local X should follow.  Choosing an
# axis merely because it is horizontal is ambiguous for upright assets: the
# sugar box has two horizontal axes, but its 176 mm long side cannot fit
# between the ~70 mm jaws.  Aligning the gripper with the long horizontal axis
# makes the local-Y jaws close across the short side.
GRASP_LONG_AXIS_LOCAL: dict[int, int] = {
    1: 1,  # sugar: local Y (176 mm)
    2: 0,  # mustard: local X (96 mm; local Y is vertical)
    3: 0,  # banana: local X (197 mm)
}

# The jaw geometry is invariant to reversing local X.  The negative direction
# keeps sugar's wrist joint6 away from its lower limit.
GRASP_LONG_AXIS_SIGN: dict[int, float] = {1: -1.0}

# Pitch about the jaw-opening axis makes far-side grasps reachable while
# keeping both fingers at equal height.  Mustard uses the same modest tilt to
# avoid the one-finger-first contact that destabilizes a vertical bottle.
GRASP_TILT_Y_DEG: dict[int, float] = {1: -30.0, 2: -30.0}

# Tilting mustard about the jaw axis shifts its finger contact region toward
# world -X.  Move the wrist base +X so the fingers, rather than the wrist
# origin, remain centred on the bottle.
GRASP_TARGET_X_OFFSETS: dict[int, float] = {2: 0.05}

# Optional lateral alignment waypoints.  The current tall objects instead use
# overhead approach offsets below, avoiding any low horizontal finger sweep.
GRASP_ALIGN_X: dict[int, float] = {}
GRASP_ALIGN_Y: dict[int, float] = {}

# Sugar and mustard must be approached from directly above.  Their high
# waypoints finish rotating before a vertical descent; horizontal motion near
# grasp height can otherwise push the object away.
GRASP_APPROACH_OFFSETS: dict[int, float] = {1: 0.25, 2: 0.30}

CARRY_Z      = TABLE_TOP_Z + 0.40   # safe carry height
PLACE_HEIGHT = TABLE_TOP_Z + 0.15   # height at which to release into basket

RETRACT_POS_X = TABLE_CENTER_X + TABLE_HALF_X - 0.05
RETRACT_POS_Y = TABLE_CENTER_Y
DEFAULT_PLACE_QUAT_W = [0.0, 1.0, 0.0, 0.0]   # top-down orientation (w,x,y,z)

# ------------------------------------------------------------------ #
# Object spawn regions
#
# These ranges intentionally match TaskEEnvCfg's evaluation distribution.
#   object_1  Y ∈ [0.25, 0.29]   (top band)
#   object_2  Y ∈ [0.14, 0.20]   (middle band)
#   object_3  Y ∈ [0.03, 0.09]   (bottom band, closest to basket)
# ------------------------------------------------------------------ #
OBJ_SPAWN_X_MIN = TABLE_CENTER_X - 0.10
OBJ_SPAWN_X_MAX = TABLE_CENTER_X + 0.10
OBJ_SPAWN_Z     = TABLE_TOP_Z + 0.05

# Per-object Y-bands: {object_idx: (y_min, y_max)}
OBJ_SPAWN_Y_BANDS = {
    1: (TABLE_CENTER_Y + 0.25, TABLE_CENTER_Y + 0.29),
    2: (TABLE_CENTER_Y + 0.14, TABLE_CENTER_Y + 0.20),
    3: (TABLE_CENTER_Y + 0.03, TABLE_CENTER_Y + 0.09),
}

# Per-object 2-D bounding-box half-extents (metres, world XY plane, scale=1).
# Used for AABB overlap rejection during randomisation.
OBJ_HALF_EXTENTS: dict[int, tuple[float, float]] = {
    1: (0.050, 0.044),   # Sugar box   (half_x, half_y)
    2: (0.050, 0.030),   # Mustard bottle
    3: (0.100, 0.040),   # Banana
}
OBJ_BBOX_MARGIN = 0.015  # extra clearance between object bounding boxes

# ------------------------------------------------------------------ #
# Success / basket bounds
# ------------------------------------------------------------------ #
BASKET_IN_X = 0.20   # ± metres in X around BASKET_CENTER_X
BASKET_IN_Y = 0.11   # ± metres in Y around BASKET_CENTER_Y

# ------------------------------------------------------------------ #
# Camera (for --save_video / --save_images)
# ------------------------------------------------------------------ #
CAM_H, CAM_W = 480, 640
CAM_POS = (TABLE_CENTER_X - 1.2, TABLE_CENTER_Y, TABLE_TOP_Z + 0.8)
CAM_ROT = (0.957, 0.0, 0.290, 0.0)   # ~34° around +Y → looks forward-down

# ------------------------------------------------------------------ #
# Actuator overrides for data collection (high stiffness for fast tracking)
# ------------------------------------------------------------------ #
ACT_STIFFNESS    = 800.0
ACT_DAMPING      = 80.0
ACT_EFFORT_LIMIT = 100.0
ACT_VEL_LIMIT    = 100.0

# ------------------------------------------------------------------ #
# Warm-up / settle steps before recording starts
# ------------------------------------------------------------------ #
WARMUP_STEPS = 150
SETTLE_STEPS = 30
