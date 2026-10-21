"""Spatial RGB pose head with fixed-camera ray/depth reconstruction.

DINO's 16x16 grid and an RGB 56x56 branch jointly localize each actor origin.
Only the depth along that image ray is regressed, using legal proprioception.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .kinematics import quaternion_rotation
from .kinematics import A, ALPHA, OFFSET, D
from .config import ThrowConfig, DEFAULT_Q

POSE_SPATIAL_FORMAT = "atec_throw_rgb_pose_spatial_v2"


class SpatialPoseHead(nn.Module):
    def __init__(self, camera, proprio_mean, proprio_std, table_root_z, quat_mean, kinematic_prior=False, position_quality=False):
        super().__init__()
        self.register_buffer("camera_position", torch.tensor(camera["position"], dtype=torch.float32))
        self.register_buffer("camera_rotation", torch.tensor(quaternion_rotation(camera["rotation_wxyz"]), dtype=torch.float32))
        # Original 640x480 frame is resized to 224x224, preserving these ratios.
        self.register_buffer("focal_uv", torch.tensor([camera["focal_length"]/camera["horizontal_aperture"],
                              camera["focal_length"]/(camera["horizontal_aperture"]*.75)], dtype=torch.float32))
        self.register_buffer("proprio_mean", torch.as_tensor(proprio_mean, dtype=torch.float32))
        self.register_buffer("proprio_std", torch.as_tensor(proprio_std, dtype=torch.float32))
        self.register_buffer("table_root_z", torch.as_tensor(table_root_z, dtype=torch.float32))
        self.register_buffer("quat_mean", torch.as_tensor(quat_mean, dtype=torch.float32))
        self.kinematic_prior = kinematic_prior
        axis = (torch.arange(56)+.5)/56
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinates", torch.stack((x, y), 0)[None])
        self.dino_projection = nn.Sequential(nn.Conv2d(384, 32, 1), nn.GroupNorm(4, 32), nn.SiLU())
        self.rgb_branch = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.SiLU(),
                                        nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU())
        self.fusion = nn.Sequential(nn.Conv2d(66, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
                                    nn.Conv2d(64, 64, 3, padding=1), nn.SiLU())
        self.heatmap = nn.Conv2d(64, 3, 1)
        self.object_embedding = nn.Embedding(3, 8)
        self.pose = nn.Sequential(nn.Linear(64+2+24+8, 128), nn.SiLU(), nn.Linear(128, 6))
        nn.init.zeros_(self.pose[-1].weight); nn.init.zeros_(self.pose[-1].bias)
        self.has_position_quality = position_quality
        if position_quality:
            self.position_quality_head = nn.Linear(128,1)
            nn.init.zeros_(self.position_quality_head.weight); nn.init.zeros_(self.position_quality_head.bias)
        if kinematic_prior:
            cfg = ThrowConfig()
            self.register_buffer("object_com_offsets", torch.tensor(cfg.object_com_offsets, dtype=torch.float32))
            self.register_buffer("pad_distance", torch.tensor([.14, .165, .12]))
            self.attachment_gate = nn.Linear(128, 1)
            nn.init.zeros_(self.attachment_gate.weight); nn.init.constant_(self.attachment_gate.bias, -2.)
            self.attachment_residual = nn.Linear(128, 1)
            nn.init.zeros_(self.attachment_residual.weight); nn.init.zeros_(self.attachment_residual.bias)

    def tool_pose(self, proprio):
        """The same static modified-DH model as CEM, from six measured joints."""
        with torch.autocast(device_type=proprio.device.type, enabled=False):
            q = proprio[:, :6].float()+proprio.new_tensor(DEFAULT_Q[:6])
            root = proprio.new_tensor([[ -1., 0., 0., 1.404253903],
                                       [0., -1., 0., 0.], [0., 0., 1., .826642648], [0., 0., 0., 1.]])
            transform = root[None].expand(len(q), -1, -1)
            for j in range(6):
                c, s = torch.cos(q[:, j]+OFFSET[j]), torch.sin(q[:, j]+OFFSET[j])
                ca, sa = math.cos(ALPHA[j]), math.sin(ALPHA[j])
                zero, one = torch.zeros_like(c), torch.ones_like(c)
                mat = torch.stack((c, -s, zero, one*A[j],
                                   s*ca, c*ca, -one*sa, -one*sa*D[j],
                                   s*sa, c*sa, one*ca, one*ca*D[j],
                                   zero, zero, zero, one), -1).reshape(-1, 4, 4)
                transform = transform@mat
        return transform

    def project(self, xyz):
        local = (xyz-self.camera_position) @ self.camera_rotation
        forward = local[..., 0].clamp_min(.05)
        uv = .5-torch.stack((local[..., 1], local[..., 2]), -1)/forward[..., None]*self.focal_uv
        return uv, forward

    def reconstruct(self, uv, depth):
        yz = -((uv-.5)/self.focal_uv)
        local_ray = torch.cat((torch.ones_like(yz[..., :1]), yz), -1)
        ray = local_ray @ self.camera_rotation.T
        return self.camera_position+ray*depth[..., None], ray

    def forward(self, patches, rgb, proprio):
        dino = F.interpolate(self.dino_projection(patches.float()), (56, 56), mode="bilinear", align_corners=False)
        image = rgb.permute(0, 3, 1, 2).float()/255
        coarse = self.rgb_branch(image)
        coordinates = self.coordinates.expand(len(rgb), -1, -1, -1)
        fused = self.fusion(torch.cat((dino, coarse, coordinates), 1))
        logits = self.heatmap(fused).flatten(2)
        weights = logits.softmax(-1)
        uv = weights @ self.coordinates.flatten(2)[0].T
        context = weights @ fused.flatten(2).transpose(1, 2)
        prop = ((proprio-self.proprio_mean)/self.proprio_std)[:, None].expand(-1, 3, -1)
        embedding = self.object_embedding.weight[None].expand(len(rgb), -1, -1)
        hidden = self.pose[:2](torch.cat((context, uv, prop, embedding), -1))
        raw = self.pose[2](hidden)
        self.position_quality_logits = self.position_quality_head(hidden).squeeze(-1) if self.has_position_quality else None
        quat = F.normalize(self.quat_mean+.30*raw[..., 1:5], dim=-1, eps=1e-6)
        _, ray = self.reconstruct(uv, torch.ones_like(raw[..., 0]))
        down = ray[..., 2].clamp(max=-.08)
        base_depth = ((self.table_root_z-self.camera_position[2])/down).clamp(.25, 2.8)
        fraction = ((base_depth-.20)/2.8).clamp(.01, .99)
        depth = .20+2.8*torch.sigmoid(torch.logit(fraction)+raw[..., 0])
        if self.kinematic_prior:
            tool = self.tool_pose(proprio)
            normal, ee = tool[:, :3, 2], tool[:, :3, 3]
            w,x,y,z = quat.unbind(-1)
            rotation = torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                                    2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                                    2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1,3,3,3)
            com_offset = (rotation@self.object_com_offsets[None, :, :, None]).squeeze(-1)
            pad_root = ee[:, None]+normal[:, None]*self.pad_distance[None, :, None]-com_offset
            # Closest point on the visual ray to the known pad center is stable
            # even when the camera ray grazes the tool's local-Z plane.
            attached_depth = ((pad_root-self.camera_position)*ray).sum(-1)/ray.square().sum(-1).clamp_min(.1)
            pad_uv, _ = self.project(pad_root)
            joints = proprio[:, :8]+proprio.new_tensor(DEFAULT_Q)
            gap = (joints[:,6]-joints[:,7]).abs()
            commanded = proprio[:,16:24]*.5+proprio.new_tensor(DEFAULT_Q)
            closed = (commanded[:,6]<0) & (commanded[:,7]>0)
            eligible = (closed[:,None] & (gap[:,None]>.01) & (gap[:,None]<.068)
                        & (attached_depth>.2) & (attached_depth<3.)
                        & ((uv-pad_uv).norm(dim=-1)<.04))
            self.attachment_logits = self.attachment_gate(hidden).squeeze(-1)
            mixture = self.attachment_logits.sigmoid()*eligible
            residual = .02*self.attachment_residual(hidden).squeeze(-1)
            depth = (1-mixture)*depth+mixture*(attached_depth+residual)
        xyz, _ = self.reconstruct(uv, depth)
        return xyz, quat, raw[..., 5], logits, uv

    def attachment_loss(self, xyz, proprio):
        tool = self.tool_pose(proprio)
        joints = proprio[:, :8]+proprio.new_tensor(DEFAULT_Q)
        gap = (joints[:,6]-joints[:,7]).abs()
        commanded = proprio[:,16:24]*.5+proprio.new_tensor(DEFAULT_Q)
        closed = (commanded[:,6]<0) & (commanded[:,7]>0)
        target = ((xyz[...,2]-self.table_root_z>.06)
                  & ((xyz-tool[:,None,:3,3]).norm(dim=-1)<.20)
                  & closed[:,None] & (gap[:,None]>.01) & (gap[:,None]<.068))
        return F.binary_cross_entropy_with_logits(self.attachment_logits, target.float())

    def localization_loss(self, logits, xyz):
        uv, _ = self.project(xyz)
        coordinates = self.coordinates.flatten(2)[0].T
        distance = ((coordinates[None, None]-uv[:, :, None])*56).square().sum(-1)
        target = (-distance/(2*1.2**2)).softmax(-1)
        visible = ((uv >= 0) & (uv <= 1)).all(-1)
        loss = -(target*logits.log_softmax(-1)).sum(-1)
        return (loss*visible).sum()/visible.sum().clamp_min(1)
