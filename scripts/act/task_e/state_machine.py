"""Pick-place state machine and grasp-quaternion solver for Task E."""

import math

import torch
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from .config import (
    STEPS, STATE_ORDER,
    CARRY_Z, PLACE_HEIGHT,
    RETRACT_POS_X, RETRACT_POS_Y,
    GRASP_Z_OFFSET, GRASP_Z_OFFSETS, GRASP_LONG_AXIS_LOCAL,
    GRASP_LONG_AXIS_SIGN, GRASP_TILT_Y_DEG, GRASP_TARGET_X_OFFSETS,
    GRASP_ALIGN_X, GRASP_ALIGN_Y, GRASP_APPROACH_OFFSETS,
    BASKET_CENTER_X, BASKET_CENTER_Y,
    DEFAULT_PLACE_QUAT_W,
)


def _build_grasp_matrix(long_axis: torch.Tensor, grip_z: torch.Tensor) -> torch.Tensor:
    """Build a right-handed gripper rotation matrix given the object's long axis.

    The Piper gripper jaw opens along its LOCAL Y axis, so local Y must be
    perpendicular to the object's long axis.

    Frame layout (columns of R_grip):
      col 0 (local X) = align_dir  ∥ long_axis
      col 1 (local Y) = jaw_dir    ⊥ long_axis  ← jaw opening direction
      col 2 (local Z) = grip_z     pointing down

    Right-hand check: col0 × col1 = align_dir × jaw_dir = grip_z ✓
    """
    jaw_dir   = torch.linalg.cross(long_axis, grip_z)   # ⊥ long_axis, in XY plane
    jaw_dir   = jaw_dir / jaw_dir.norm().clamp(min=1e-6)
    align_dir = torch.linalg.cross(jaw_dir, grip_z)     # ∥ long_axis
    align_dir = align_dir / align_dir.norm().clamp(min=1e-6)
    return torch.stack([align_dir, jaw_dir, grip_z], dim=1)   # (3, 3)


def compute_grasp_quat(
    obj_quat_w: torch.Tensor,
    device: str,
    long_axis_local: int | None = None,
    long_axis_sign: float = 1.0,
    tilt_y_deg: float = 0.0,
) -> torch.Tensor:
    """Compute a top-down grasp quaternion for the object.

    Uses the configured local long axis when supplied.  Otherwise it falls back
    to the most horizontal object axis.  It then builds a gripper frame where
    the jaw (local Y) is perpendicular to that axis.

    Parameters
    ----------
    obj_quat_w : (4,) tensor, (w, x, y, z)
    device     : torch device string

    Returns
    -------
    grasp_quat : (4,) tensor, (w, x, y, z)
    """
    R_obj        = matrix_from_quat(obj_quat_w.unsqueeze(0)).squeeze(0)   # (3, 3)
    grip_z       = torch.tensor([0.0, 0.0, -1.0], device=device)
    default_quat = torch.tensor(DEFAULT_PLACE_QUAT_W, dtype=torch.float32, device=device)

    if long_axis_local is not None:
        if long_axis_local not in (0, 1, 2):
            raise ValueError(f"long_axis_local must be 0, 1, or 2; got {long_axis_local}")
        selected = R_obj[:, long_axis_local].clone() * long_axis_sign
        selected[2] = 0.0
        if selected.norm().item() >= 1e-3:
            long_axis = selected / selected.norm().clamp(min=1e-6)
            R_grip = _build_grasp_matrix(long_axis, grip_z)
            if tilt_y_deg:
                angle = math.radians(tilt_y_deg)
                c, s = math.cos(angle), math.sin(angle)
                tilt = torch.tensor(
                    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                    dtype=R_grip.dtype,
                    device=device,
                )
                R_grip = R_grip @ tilt
            return quat_from_matrix(R_grip.unsqueeze(0)).squeeze(0)

    # Fallback for unconfigured assets: project each object axis onto XY and
    # keep the most horizontal candidate(s).
    norms, axes_xy = [], []
    for col in range(3):
        ax = torch.tensor([R_obj[0, col].item(), R_obj[1, col].item(), 0.0], device=device)
        norms.append(ax.norm().item())
        axes_xy.append(ax)

    best_norm  = max(norms)
    candidates = [
        axes_xy[c] / max(norms[c], 1e-6)
        for c in range(3)
        if norms[c] >= best_norm - 1e-3
    ]

    # Among candidates, pick the one whose grasp frame is closest to the default orientation
    best_cos  = -2.0
    long_axis = candidates[0]
    for cand in candidates:
        q_cand  = quat_from_matrix(_build_grasp_matrix(cand, grip_z).unsqueeze(0)).squeeze(0)
        cos_sim = torch.abs((q_cand * default_quat).sum()).item()
        if cos_sim > best_cos:
            best_cos  = cos_sim
            long_axis = cand

    R_grip = _build_grasp_matrix(long_axis, grip_z)                  # (3, 3)
    return quat_from_matrix(R_grip.unsqueeze(0)).squeeze(0)           # (4,) w,x,y,z


class PickPlaceStateMachine:
    """Finite state machine that sequences pick-and-place for multiple objects.

    States (in order): INIT → PRE_GRASP → ALIGN → REACH → DESCEND
    → CLOSE → LIFT →
                       TRANSPORT → PLACE → OPEN → RETRACT → (next object or done)
    """

    def __init__(self, object_indices: list[int], device: str):
        self._obj_indices = object_indices
        self._device      = device
        self._grasp_quat_cache: dict[int, torch.Tensor] = {}
        self.reset()

    def reset(self) -> None:
        self._ptr             = 0
        self._state_idx       = 0
        self._count           = 0
        self.done             = False
        self._cached_obj_pos: torch.Tensor | None = None
        self._grasp_quat_cache.clear()

    def set_grasp_quat(self, obj_idx: int, obj_quat_w: torch.Tensor) -> None:
        """Pre-compute and cache the grasp quaternion for one object."""
        self._grasp_quat_cache[obj_idx] = compute_grasp_quat(
            obj_quat_w,
            self._device,
            long_axis_local=GRASP_LONG_AXIS_LOCAL.get(obj_idx),
            long_axis_sign=GRASP_LONG_AXIS_SIGN.get(obj_idx, 1.0),
            tilt_y_deg=GRASP_TILT_Y_DEG.get(obj_idx, 0.0),
        )

    @property
    def step_count(self) -> int:
        return self._count

    def retry_object(self, obj_idx: int) -> None:
        """Retry from open-gripper HOME so a dropped object can settle."""
        self._ptr = self._obj_indices.index(obj_idx)
        self._state_idx = STATE_ORDER.index("INIT")
        self._count = 0
        self.done = False
        self._cached_obj_pos = None

    def skip_object(self, obj_idx: int) -> None:
        """Advance even if the preceding tick already advanced the pointer."""
        self._ptr = self._obj_indices.index(obj_idx)+1
        self._count = 0
        self._cached_obj_pos = None
        self.done = self._ptr >= len(self._obj_indices)
        if not self.done:
            self._state_idx = STATE_ORDER.index("PRE_GRASP")

    def tick(self, obj_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Advance the state machine by one step.

        Parameters
        ----------
        obj_pos : (3,) tensor — current object position in world frame

        Returns
        -------
        ee_pos_des  : (3,) target EE position
        ee_quat_des : (4,) target EE orientation (w,x,y,z)
        gripper_cmd : "open" | "close"
        """
        s = self.state
        d = self._device

        # Freeze object position at start of PRE_GRASP to avoid drift during descent
        if s == "PRE_GRASP" and self._count == 0:
            self._cached_obj_pos = obj_pos.clone()
        if s in ("ALIGN", "REACH", "DESCEND", "CLOSE") and self._cached_obj_pos is not None:
            obj_pos = self._cached_obj_pos

        ee_pos, gripper = self._get_target_pos_gripper(s, obj_pos, d)
        ee_quat         = self._get_target_quat(s, d)

        self._count += 1
        if self._count >= STEPS[s]:
            self._count = 0
            if s == "RETRACT":
                self._ptr          += 1
                self._cached_obj_pos = None
                if self._ptr >= len(self._obj_indices):
                    self.done = True
                    return ee_pos, ee_quat, gripper
                self._state_idx = STATE_ORDER.index("PRE_GRASP")
            else:
                self._state_idx += 1

        return ee_pos, ee_quat, gripper

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> str:
        return STATE_ORDER[self._state_idx]

    @property
    def current_object_key(self) -> str:
        return f"object_{self._obj_indices[self._ptr]}"

    @property
    def current_object_index(self) -> int:
        return self._obj_indices[self._ptr]

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _get_target_pos_gripper(
        self, s: str, obj_pos: torch.Tensor, d: str
    ) -> tuple[torch.Tensor, str]:
        if s == "INIT":
            return torch.tensor([RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z], device=d), "open"
        elif s == "PRE_GRASP":
            p = obj_pos.clone(); p[2] = CARRY_Z
            return p, "open"
        elif s == "ALIGN":
            if self.current_object_index in GRASP_APPROACH_OFFSETS:
                p = obj_pos.clone()
                p[0] += GRASP_TARGET_X_OFFSETS.get(self.current_object_index, 0.0)
                p[2] += GRASP_APPROACH_OFFSETS[self.current_object_index]
                return p, "open"
            if (self.current_object_index not in GRASP_ALIGN_X and
                    self.current_object_index not in GRASP_ALIGN_Y):
                p = obj_pos.clone(); p[2] = CARRY_Z
                return p, "open"
            offset = GRASP_APPROACH_OFFSETS.get(
                self.current_object_index,
                GRASP_Z_OFFSETS.get(self.current_object_index, GRASP_Z_OFFSET),
            )
            p = obj_pos.clone()
            if self.current_object_index in GRASP_ALIGN_X:
                p[0] = GRASP_ALIGN_X[self.current_object_index]
            if self.current_object_index in GRASP_ALIGN_Y:
                p[1] = GRASP_ALIGN_Y[self.current_object_index]
            p[2] += offset
            return p, "open"
        elif s == "REACH":
            offset = GRASP_APPROACH_OFFSETS.get(
                self.current_object_index,
                GRASP_Z_OFFSETS.get(self.current_object_index, GRASP_Z_OFFSET),
            )
            p = obj_pos.clone()
            p[0] += GRASP_TARGET_X_OFFSETS.get(self.current_object_index, 0.0)
            p[2] += offset
            return p, "open"
        elif s == "DESCEND":
            offset = GRASP_Z_OFFSETS.get(self.current_object_index, GRASP_Z_OFFSET)
            p = obj_pos.clone()
            p[0] += GRASP_TARGET_X_OFFSETS.get(self.current_object_index, 0.0)
            p[2] += offset
            return p, "open"
        elif s == "CLOSE":
            offset = GRASP_Z_OFFSETS.get(self.current_object_index, GRASP_Z_OFFSET)
            p = obj_pos.clone()
            p[0] += GRASP_TARGET_X_OFFSETS.get(self.current_object_index, 0.0)
            p[2] += offset
            return p, "close"
        elif s == "LIFT":
            p = obj_pos.clone()
            p[0] += GRASP_TARGET_X_OFFSETS.get(self.current_object_index, 0.0)
            p[2] = CARRY_Z
            return p, "close"
        elif s == "TRANSPORT":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, CARRY_Z], device=d), "close"
        elif s == "PLACE":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, PLACE_HEIGHT], device=d), "close"
        elif s == "OPEN":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, PLACE_HEIGHT], device=d), "open"
        elif s == "LIFT_RETRACT":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, CARRY_Z], device=d), "open"
        elif s == "RETRACT":
            return torch.tensor([RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z], device=d), "open"
        else:
            raise ValueError(f"Unknown state: {s}")

    def _get_target_quat(self, s: str, d: str) -> torch.Tensor:
        default_quat = torch.tensor(DEFAULT_PLACE_QUAT_W, dtype=torch.float32, device=d)
        # Finish rotating while safely above the object.  Switching from the
        # default pose only during REACH makes the open fingers sweep sideways
        # through tall objects while the arm is descending.
        cur_idx = self._obj_indices[min(self._ptr, len(self._obj_indices) - 1)]
        has_alignment = (
            cur_idx in GRASP_ALIGN_X
            or cur_idx in GRASP_ALIGN_Y
            or cur_idx in GRASP_APPROACH_OFFSETS
        )
        use_grasp_quat = s in ("REACH", "DESCEND", "CLOSE", "LIFT")
        use_grasp_quat |= s == "PRE_GRASP" and not has_alignment
        use_grasp_quat |= s == "ALIGN" and has_alignment
        if use_grasp_quat:
            return self._grasp_quat_cache.get(cur_idx, default_quat)
        return default_quat
