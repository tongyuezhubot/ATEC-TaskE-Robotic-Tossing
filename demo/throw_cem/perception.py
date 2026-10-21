"""RGB+24-D calibrated pose head. Neither depth nor live simulator truth is read."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .data import camera_contract, canonical_rgb

SPATIAL_FORMATS = ("atec_throw_rgb_pose_spatial_v2", "atec_throw_rgb_pose_spatial_v4",
                   "atec_throw_rgb_pose_spatial_v5", "atec_throw_rgb_pose_spatial_v6")


def numpy_observation(obs):
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if "proprio" not in obs or "video_rgb" not in obs.get("image", {}):
        raise ValueError("Throw controller requires standard RGB and 24-D proprio")
    p = array(obs["proprio"])
    if p.shape == (1, 24):
        p = p[0]
    if p.shape != (24,) or not np.isfinite(p).all():
        raise ValueError("Single environment with finite 24-D proprio required")
    rgb = array(obs["image"]["video_rgb"])
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim != 3:
        raise ValueError("Invalid RGB layout")
    if rgb.shape[0] in (3, 4):
        rgb = rgb.transpose(1, 2, 0)
    if rgb.shape[-1] not in (3, 4):
        raise ValueError("RGB channels missing")
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("Float RGB must be in [0,1]")
        rgb = (rgb*255).round().astype(np.uint8)
    return p.astype(float), np.ascontiguousarray(rgb)


@dataclass
class VisualPose:
    positions: np.ndarray  # three environment-local actor-root XYZ positions
    quaternions: np.ndarray  # three wxyz object rotations
    confidence: np.ndarray
    position_confidence: np.ndarray | None = None  # root AND COM, not orientation accuracy


def _require_held_scope(cfg, calibrated_objects):
    if not {obj+1 for obj in cfg.object_order}.issubset(set(calibrated_objects)):
        raise ValueError("RGB held calibration does not cover the requested objects")


class RGBPosePerception:
    """New checkpoint contract; old JEPA progress/distance heads are NOT pose heads.

    MLP: LN(16*384+24), Linear(256), SiLU, Linear(24): three [xyz,wxyz,confidence].
    Requires calibrated camera metadata and training-only same-frame pose labels.
    Confidence describes held-out pose accuracy, not a visibility/contact sensor.
    """
    def __init__(self, checkpoint, cfg):
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError("Set ATEC_THROW_PERCEPTION to a calibrated RGB pose checkpoint; old JEPA/ACT weights cannot replace it")
        import torch
        self.torch = torch
        self.confidence_min = cfg.confidence_min
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.spatial = payload.get("format") in SPATIAL_FORMATS
        self.position_thresholds = np.asarray(payload.get("position_confidence_thresholds", payload["confidence_thresholds"]), float)
        if (self.position_thresholds.shape != (3,) or not np.isfinite(self.position_thresholds).all()
                or np.any(self.position_thresholds < .7) or np.any(self.position_thresholds > 1)):
            raise ValueError("Invalid root/COM confidence calibration")
        if (payload.get("format") not in ("atec_throw_rgb_pose_v1", *SPATIAL_FORMATS)
                or not payload.get("calibrated", False)):
            raise ValueError("RGB throwing perception has not been calibrated")
        if payload.get("format")=="atec_throw_rgb_pose_spatial_v6" and not payload.get("position_quality_head",False):
            raise ValueError("Spatial v6 requires its separate root/COM confidence head")
        if payload.get("format")=="atec_throw_rgb_pose_spatial_v6":
            held = payload.get("held_calibrated_object_ids",[])
            if "position_confidence_thresholds" not in payload or not held or any(i not in (1,2,3) for i in held):
                raise ValueError("Spatial v6 requires phase-specific usage calibration")
            _require_held_scope(cfg,held)
        expected = camera_contract(cfg)
        try:
            camera_matches = all(np.allclose(payload["camera"][key], value, rtol=0, atol=1e-6)
                                 for key, value in expected.items())
        except (KeyError, TypeError, ValueError):
            camera_matches = False
        if not camera_matches or payload.get("input_fields") != ["rgb", "proprio"]:
            raise ValueError("Perception camera/input contract does not match Task E")
        self.mean = torch.tensor(payload["proprio_mean"], device=self.device)
        self.std = torch.tensor(payload["proprio_std"], device=self.device)
        if self.mean.shape != (24,) or self.std.shape != (24,) or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or not (self.std > 0).all():
            raise ValueError("Invalid pose normalization")
        if self.spatial:
            from .pose_model import load_encoder
            from .pose_spatial import SpatialPoseHead
            table_z = torch.tensor(payload["table_root_z"])
            quat_mean = torch.tensor(payload["quat_mean"])
            self.thresholds = np.asarray(payload["confidence_thresholds"], float)
            if (payload.get("original_image_size") != [480, 640] or table_z.shape != (3,)
                    or quat_mean.shape != (3, 4) or not torch.isfinite(table_z).all()
                    or not torch.isfinite(quat_mean).all() or self.thresholds.shape != (3,)
                    or not np.isfinite(self.thresholds).all() or np.any(self.thresholds < .7)
                    or np.any(self.thresholds > 1)):
                raise ValueError("Invalid spatial pose checkpoint contract")
            self.encoder = load_encoder(self.device)
            self.head = SpatialPoseHead(payload["camera"], payload["proprio_mean"], payload["proprio_std"],
                                        payload["table_root_z"], payload["quat_mean"],
                                        kinematic_prior=payload.get("kinematic_prior",False),
                                        position_quality=payload.get("position_quality_head",False)).to(self.device).eval()
            self.head.load_state_dict(payload["model"], strict=True)
            return
        from .pose_model import load_encoder, make_head
        self.position_mean = torch.tensor(payload["position_mean"], device=self.device)
        self.position_std = torch.tensor(payload["position_std"], device=self.device)
        self.thresholds = np.asarray(payload["confidence_thresholds"], float)
        if (self.position_mean.shape != (3, 3) or self.position_std.shape != (3, 3)
                or not torch.isfinite(self.position_mean).all() or not torch.isfinite(self.position_std).all()
                or not (self.position_std > 0).all() or self.thresholds.shape != (3,)
                or not np.isfinite(self.thresholds).all() or np.any(self.thresholds < .7) or np.any(self.thresholds > 1)):
            raise ValueError("Invalid calibrated pose output contract")
        self.encoder = load_encoder(self.device)
        self.head = make_head().to(self.device).eval()
        self.head.load_state_dict(payload["model"], strict=True)

    def observe(self, p, rgb):
        torch = self.torch
        from .pose_model import image_features, patch_features, decode
        with torch.inference_mode():
            image = torch.from_numpy(canonical_rgb(rgb)).to(self.device)[None]
            prop = torch.as_tensor(p, device=self.device).float()
            if self.spatial:
                # Match the frozen AMP/half feature cache used for calibration.
                with torch.autocast(device_type=self.device.type, enabled=self.device.type=="cuda", dtype=torch.float16):
                    feature = patch_features(self.encoder, image).half()
                xyz, quat, logits, _, _ = self.head(feature, image, prop[None])
            else:
                feature = image_features(self.encoder, image)
                raw = self.head(torch.cat((feature, ((prop-self.mean)/self.std)[None]), -1))
                xyz, quat, logits = decode(raw, self.position_mean, self.position_std)
            xyz, quat = xyz[0].cpu().numpy(), quat[0].cpu().numpy()
            # Monotone quality SCORE, not a calibrated posterior probability.
            confidence = np.minimum(1., logits[0].sigmoid().cpu().numpy()/self.thresholds*self.confidence_min)
            position_logits = (self.head.position_quality_logits if self.spatial and self.head.position_quality_logits is not None else logits)
            position_confidence = np.minimum(1., position_logits[0].sigmoid().cpu().numpy()/self.position_thresholds*self.confidence_min)
        if not all(np.isfinite(x).all() for x in (xyz, quat, confidence, position_confidence)) or np.any(np.linalg.norm(quat, axis=-1) < .99):
            raise ValueError("Non-finite RGB pose prediction")
        return VisualPose(xyz, quat, confidence, position_confidence)
