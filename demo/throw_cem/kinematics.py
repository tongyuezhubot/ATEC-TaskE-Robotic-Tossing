"""Piper modified-DH model reused from the existing expert, NumPy-only IK.

Uses joint observations and static robot calibration, never scene/object state.
Planning feasibility is kinematic; motor torque/tracking still need simulator proof.
"""
import numpy as np
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from .config import ARM_LOWER, ARM_UPPER, HOME_Q

A = (0., 0., .28503, -.02198, 0., 0.)
ALPHA = (0., -np.pi/2, 0., np.pi/2, -np.pi/2, np.pi/2)
OFFSET = (0., np.deg2rad(-172.22), np.deg2rad(-102.78), 0., 0., 0.)
D = (.123, 0., 0., .25075, 0., .091)


@lru_cache(maxsize=1)
def open_gripper_vertices():
    """Static original hand hulls at fully open jaws; not whole-robot collision."""
    payload = json.loads(Path(__file__).with_name("gripper_geometry.json").read_text())
    if payload.get("format") != "atec_static_gripper_hulls_v1":
        raise ValueError("Invalid static hand geometry")
    project = Path(__file__).resolve().parents[2]
    for name, expected in payload["source_sha256"].items():
        if hashlib.sha256((project/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Static hand hulls do not match the original robot assets")
    parts = []
    for name, shift in (("gripper_base",0.),("link7",-.035),("link8",.035)):
        points = np.asarray(payload["parts"][name]["vertices"], float).copy()
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Invalid hand hull vertices")
        points[:,1] += shift
        parts.append(points)
    points = np.concatenate(parts); points.setflags(write=False)
    return points


def hand_table_clearance(position, rotation, table_z):
    """Lowest point of the open hand above the known table plane."""
    return float(position[2]+np.min(open_gripper_vertices()@np.asarray(rotation)[2])-table_z)


def open_hand_bottle_gap(position, rotation, actor_root, object_rotation):
    """Conservative tool-frame AABB separation from a non-target bottle.

    A positive gap proves separation of the original hulls. Overlapping boxes
    are rejected even when exact meshes might separate. Includes both open
    fingers and palm, not the arm or other objects.
    """
    bottle = mustard_palm_geometry()[0]
    local = (bottle@np.asarray(object_rotation).T+actor_root-position)@np.asarray(rotation)
    hand = open_gripper_vertices()
    return float(max(np.max(local.min(0)-hand.max(0)),
                     np.max(hand.min(0)-local.max(0))))


@lru_cache(maxsize=1)
def mustard_palm_geometry():
    """Original static convex hulls; SciPy/USD are needed only for export."""
    open_gripper_vertices()  # also verify the underlying original robot USDs
    payload = json.loads(Path(__file__).with_name("mustard_grasp_geometry.json").read_text())
    if payload.get("format") != "atec_mustard_palm_hulls_v1":
        raise ValueError("Invalid mustard/palm geometry")
    project = Path(__file__).resolve().parents[2]
    for name, expected in payload["source_sha256"].items():
        if hashlib.sha256((project/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Mustard/palm hulls do not match original assets")
    return tuple(np.asarray(payload[part][key],float)
                 for part,key in (("bottle","vertices"),("palm","vertices"),("palm","normals")))


def mustard_palm_gap(position, rotation, actor_root, object_rotation):
    """Proven separating projection gap, not exact distance/full collision QA.

    Using only palm face normals is conservative: ambiguous configurations are
    rejected, not accepted. Live poses must come from the active pose provider.
    Finger contact, other objects and whole-arm collisions are not modeled.
    """
    bottle,palm,normals = mustard_palm_geometry()
    axes = normals@np.asarray(rotation).T
    a = (bottle@np.asarray(object_rotation).T+actor_root)@axes.T
    b = (palm@np.asarray(rotation).T+position)@axes.T
    return float(max(np.max(a.min(0)-b.max(0)),np.max(b.min(0)-a.max(0))))


@lru_cache(maxsize=1)
def sugar_palm_geometry():
    """Original convex hulls, with original-asset hash guards."""
    open_gripper_vertices()
    payload = json.loads(Path(__file__).with_name("sugar_grasp_geometry.json").read_text())
    if payload.get("format") != "atec_sugar_palm_hulls_v1":
        raise ValueError("Invalid sugar/palm geometry")
    project = Path(__file__).resolve().parents[2]
    for name, expected in payload["source_sha256"].items():
        if hashlib.sha256((project/name).read_bytes()).hexdigest() != expected:
            raise ValueError("Sugar/palm hulls do not match original assets")
    return tuple(np.asarray(payload[part][key],float)
                 for part,key in (("sugar","vertices"),("palm","vertices"),("palm","normals")))


def sugar_palm_gap(position, rotation, actor_root, object_rotation):
    """Conservative separating projection; not whole-arm collision proof."""
    sugar,palm,normals = sugar_palm_geometry()
    axes = normals@np.asarray(rotation).T
    a = (sugar@np.asarray(object_rotation).T+actor_root)@axes.T
    b = (palm@np.asarray(rotation).T+position)@axes.T
    return float(max(np.max(a.min(0)-b.max(0)),np.max(b.min(0)-a.max(0))))


def forward_base(q):
    transform = np.eye(4)
    for a, alpha, offset, d, joint in zip(A, ALPHA, OFFSET, D, q):
        c, s = np.cos(joint+offset), np.sin(joint+offset)
        ca, sa = np.cos(alpha), np.sin(alpha)
        transform = transform @ np.array([[c, -s, 0., a],
            [s*ca, c*ca, -sa, -sa*d], [s*sa, c*sa, ca, ca*d], [0., 0., 0., 1.]])
    return transform


def rotation_log(rotation):
    theta = np.arccos(np.clip((np.trace(rotation)-1)/2, -1., 1.))
    skew = np.array([rotation[2, 1]-rotation[1, 2], rotation[0, 2]-rotation[2, 0], rotation[1, 0]-rotation[0, 1]])
    if theta < 1e-6:
        return skew/2
    if np.pi-theta < 1e-4:
        _, vectors = np.linalg.eigh((rotation+np.eye(3))/2)
        return theta*vectors[:, -1]
    return theta*skew/(2*np.sin(theta))


def quaternion_rotation(quat):
    quat = np.asarray(quat, float)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("Invalid predicted object quaternion")
    w, x, y, z = quat/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def grasp_rotation(quat, obj):
    rotation = quaternion_rotation(quat)
    axis = rotation[:, 1 if obj == 0 else 0].copy()
    axis *= -1 if obj == 0 else 1
    axis[2] = 0.
    if np.linalg.norm(axis) < 1e-3:
        raise ValueError("Cannot infer a horizontal grasp axis")
    x = axis/np.linalg.norm(axis); z = np.array([0., 0., -1.])
    y = np.cross(z, x)
    tool = np.stack((x, y, z), axis=1)
    if obj in (0, 1):
        c, s = np.cos(np.deg2rad(-30)), np.sin(np.deg2rad(-30))
        tool = tool @ np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    return tool


def grasp_rotations(quat, obj, tilts=None):
    """Jaw-symmetric alternatives, shallow tilt preferred over large tilt.

    Sign of the long axis is not observable grasp semantics. Flipping it before
    tilting (not after) allows bounded wrist solutions without changing which
    axis the jaws squeeze. Every alternative remains physically unvalidated.
    """
    axis = quaternion_rotation(quat)[:, 1 if obj == 0 else 0].copy()
    axis[2] = 0.
    if np.linalg.norm(axis) < 1e-3:
        raise ValueError("Cannot infer a horizontal grasp axis")
    axis /= np.linalg.norm(axis)
    z = np.array([0., 0., -1.])
    if tilts is None:
        tilts = (0., -30., 30.) if obj == 2 else (-30., 30., -45., 45.)
    for tilt in tilts:
        for sign in (1., -1.):
            x = axis*sign
            tool = np.stack((x, np.cross(z, x), z), axis=1)
            a = np.deg2rad(tilt); c, s = np.cos(a), np.sin(a)
            yield tool @ np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


class PiperKinematics:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = np.eye(4)
        self.root[:3, :3] = np.diag([-1., -1., 1.])
        self.root[:3, 3] = cfg.root_xyz

    def forward(self, q):
        return self.root @ forward_base(np.asarray(q)[:6])

    def throw_rotation(self):
        angle = np.deg2rad(self.cfg.throw_tilt_delta_deg)
        c, s = np.cos(angle), np.sin(angle)
        return self.forward(HOME_Q)[:3, :3] @ np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])

    def jacobian(self, q):
        pose = self.forward(q); jac = np.empty((6, 6))
        for j in range(6):
            other = np.asarray(q[:6], float).copy(); other[j] += 1e-5
            target = self.forward(other)
            jac[:3, j] = (target[:3, 3]-pose[:3, 3])/1e-5
            jac[3:, j] = rotation_log(target[:3, :3] @ pose[:3, :3].T)/1e-5
        return jac

    def solve_position(self, position, seed, max_iterations=60, tolerance=1e-6):
        """Three arm joints reach XYZ; wrist angles stay at the held posture.

        The hand rotates naturally with the arm. Release scoring must propagate
        this rotation to the held-object point; never use a constant world offset.
        """
        q = np.asarray(seed[:6], float).copy()
        if np.any(q < ARM_LOWER) or np.any(q > ARM_UPPER):
            raise ValueError("Initial arm pose outside planner bounds")
        for _ in range(max_iterations):
            current = self.forward(q)[:3, 3]
            error = np.asarray(position)-current
            if np.linalg.norm(error) <= tolerance:
                return q
            jac = np.empty((3, 3))
            for joint in range(3):
                other = q.copy(); other[joint] += 1e-5
                jac[:, joint] = (self.forward(other)[:3, 3]-current)/1e-5
            delta = jac.T@np.linalg.solve(jac@jac.T+1e-6*np.eye(3), error)
            q[:3] = np.clip(q[:3]+np.clip(delta, -.10, .10), ARM_LOWER[:3], ARM_UPPER[:3])
        raise ValueError("Position-only arm swing unreachable")

    def solve(self, position, rotation, seed, max_iterations=60, position_tolerance=None, rotation_tolerance=None):
        q = np.clip(np.asarray(seed[:6], float), ARM_LOWER, ARM_UPPER)
        pos_tol = self.cfg.pose_position_tolerance if position_tolerance is None else position_tolerance
        rot_tol = self.cfg.pose_rotation_tolerance if rotation_tolerance is None else rotation_tolerance
        for _ in range(max_iterations):
            pose = self.forward(q)
            dp = np.asarray(position)-pose[:3, 3]
            dr = rotation_log(rotation @ pose[:3, :3].T)
            if np.linalg.norm(dp) <= pos_tol and np.linalg.norm(dr) <= rot_tol:
                return q
            error = np.r_[dp, .15*dr]
            jac = self.jacobian(q); jac[3:] *= .15
            delta = jac.T @ np.linalg.solve(jac @ jac.T + .0004*np.eye(6), error)
            q = np.clip(q+np.clip(delta, -.10, .10), ARM_LOWER, ARM_UPPER)
        raise ValueError("Unreachable/poorly conditioned pose; do not execute")
