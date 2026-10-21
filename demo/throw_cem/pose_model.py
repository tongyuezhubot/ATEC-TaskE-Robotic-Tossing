"""Shared frozen image features and pose head for training/inference (Torch only).

No downloading, hub code, ACT, or old JEPA predictor initialization.
"""
import os
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F


def load_encoder(device):
    bundle = Path(os.environ.get("ATEC_JEPA_BUNDLE", Path(__file__).resolve().parents[2]/"work/jepa_prepare_2026-10-09"))
    if not (bundle/"weights/dinov2_vits14_pretrain.pth").is_file():
        raise FileNotFoundError("Set ATEC_JEPA_BUNDLE to the existing vendored JEPA bundle with DINOv2 weights")
    vendor = (bundle/"vendor/dinov2").resolve()
    module = sys.modules.get("dinov2")
    if module is not None:
        locations = list(getattr(module, "__path__", []))
        if not locations or any(not Path(p).resolve().is_relative_to(vendor) for p in locations):
            raise RuntimeError("Unexpected DINOv2 module source; do not mix encoders")
    sys.path.insert(0, str(vendor))
    # Import ONLY DINOv2. JEPA's upstream_imports checks/imports src/app too,
    # which can collide with Isaac's namespaces in the simulation process.
    from dinov2.hub.backbones import dinov2_vits14
    encoder = dinov2_vits14(pretrained=False).to(device).eval()
    encoder.load_state_dict(torch.load(bundle/"weights/dinov2_vits14_pretrain.pth", map_location=device, weights_only=True))
    encoder.requires_grad_(False)
    return encoder


def patch_features(encoder, rgb):
    # NHWC uint8 throughout. Exactly the same resizing at collection/inference.
    image = rgb.permute(0, 3, 1, 2).float()/255
    image = F.interpolate(image, (224, 224), mode="bilinear", align_corners=False, antialias=True)
    mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
    std = image.new_tensor([.229, .224, .225])[None, :, None, None]
    tokens = encoder.forward_features((image-mean)/std)["x_norm_patchtokens"]
    return tokens.transpose(1, 2).reshape(-1, 384, 16, 16)


def image_features(encoder, rgb):
    return F.adaptive_avg_pool2d(patch_features(encoder, rgb), (4, 4)).flatten(1)


def make_head():
    return nn.Sequential(nn.LayerNorm(384*16+24), nn.Linear(384*16+24, 256), nn.SiLU(), nn.Linear(256, 24))


def decode(raw, position_mean, position_std):
    values = raw.reshape(-1, 3, 8)
    xyz = values[..., :3]*position_std+position_mean
    quat = F.normalize(values[..., 3:7], dim=-1, eps=1e-6)
    return xyz, quat, values[..., 7]
