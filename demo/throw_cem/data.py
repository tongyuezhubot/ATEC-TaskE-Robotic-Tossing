"""Versioned, time-aligned privileged TRAINING labels; never policy inputs."""
import json
from pathlib import Path
import numpy as np

POSE_FORMAT = "atec_throw_pose_dataset_v1"


def camera_contract(cfg):
    return {"position": [-.20, 0., cfg.table_z+.8], "rotation_wxyz": [.957, 0., .290, 0.],
            "focal_length": 24., "horizontal_aperture": 20.955}


def canonical_rgb(rgb):
    from PIL import Image
    return np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BILINEAR)).copy()


class PoseDatasetWriter:
    def __init__(self, path, cfg, seed, max_frames=1500):
        import h5py
        self.file = h5py.File(Path(path), "x")
        self.max_frames = int(max_frames)
        self.count = 0
        self.file.attrs.update(format=POSE_FORMAT, camera=json.dumps(camera_contract(cfg)),
                               source_seed=int(seed), input_fields=json.dumps(["rgb", "proprio"]),
                               label_timing="same pre-action state as RGB/proprio",
                               training_only_truth=True, complete=False)
        shapes = {"rgb": ((224, 224, 3), np.uint8), "proprio": ((24,), np.float32),
                  "object_xyz": ((3, 3), np.float32), "object_wxyz": ((3, 4), np.float32),
                  "step": ((), np.int32), "stage": ((), "S32")}
        for name, (shape, dtype) in shapes.items():
            self.file.create_dataset(name, shape=(0,)+shape, maxshape=(None,)+shape,
                                     chunks=(1,)+shape, dtype=dtype, compression="gzip", compression_opts=1)

    def append(self, step, stage, proprio, rgb, positions, quaternions):
        if self.count >= self.max_frames:
            return
        p, xyz, quat = (np.asarray(x) for x in (proprio, positions, quaternions))
        if p.shape != (24,) or xyz.shape != (3, 3) or quat.shape != (3, 4):
            raise ValueError("Pose dataset shape mismatch")
        if not all(np.isfinite(x).all() for x in (p, xyz, quat)) or np.any(np.abs(np.linalg.norm(quat, axis=-1)-1) > .01):
            raise ValueError("Invalid training pose labels")
        values = {"rgb": canonical_rgb(rgb), "proprio": p, "object_xyz": xyz,
                  "object_wxyz": quat, "step": step, "stage": stage.encode()}
        for name, value in values.items():
            data = self.file[name]
            data.resize(self.count+1, axis=0)
            data[self.count] = value
        self.count += 1
        if self.count % 100 == 0:
            self.file.flush()

    def close(self, complete=False):
        self.file.attrs["complete"] = bool(complete)
        self.file.close()


def pose_calibration(xyz, quat, probabilities, target_xyz, target_quat,
                     position_tolerance=.015, angle_tolerance_degrees=12., min_accepted=30):
    """Held-out quality gate; confidence labels are pose accuracy, not visibility.

    Reject when no threshold >= .7 achieves >=95% joint XYZ/rotation precision
    and >=20% coverage for EACH object. A failed candidate is never deployable.
    """
    error = np.linalg.norm(np.asarray(xyz)-target_xyz, axis=-1)
    pred = np.asarray(quat); target = np.asarray(target_quat)
    pred = pred/np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-6)
    target = target/np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-6)
    angle = np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(pred*target, axis=-1)), 0, 1)))
    good = (error <= position_tolerance) & (angle <= angle_tolerance_degrees)
    thresholds, reports = [], []
    for obj in range(3):
        chosen = None
        for threshold in np.linspace(.70, .99, 60):
            selected = probabilities[:, obj] >= threshold
            n = int(selected.sum()); coverage = float(selected.mean())
            precision = float(good[selected, obj].mean()) if n else 0.
            if n >= min_accepted and coverage >= .20 and precision >= .95:
                chosen = float(threshold)
                break
        thresholds.append(chosen if chosen is not None else 1.)
        reports.append({"object": obj+1, "passed": chosen is not None,
                        "position_rmse_m": float(np.sqrt(np.mean(error[:, obj]**2))),
                        "rotation_median_degrees": float(np.median(angle[:, obj])),
                        "accepted": int((probabilities[:, obj] >= thresholds[-1]).sum()),
                        "coverage": float((probabilities[:, obj] >= thresholds[-1]).mean()),
                        "precision": float(good[probabilities[:, obj] >= thresholds[-1], obj].mean()) if chosen is not None else 0.,
                        "threshold": thresholds[-1]})
    return {"passed": all(r["passed"] for r in reports), "objects": reports,
            "confidence_thresholds": thresholds, "position_tolerance_m": position_tolerance,
            "angle_tolerance_degrees": angle_tolerance_degrees,
            "quality_is_not_visibility_or_contact_proof": True}
