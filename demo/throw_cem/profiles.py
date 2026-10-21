"""Per-object calibrated components. Missing RGB authorization fails closed."""
from dataclasses import asdict, replace
import json
from pathlib import Path

from .config import MOTION_FIELDS
from .perception import RGBPosePerception
from .release_world import ReleaseWorldModel
from .physics import physics_contract


def load_profiles(path, cfg, joint_model, provided_perception=None):
    project = Path(__file__).resolve().parents[2]
    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else project/p
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != "atec_throw_profiles_v1":
        raise ValueError("Invalid per-object throw profile format")
    entries = payload["objects"]
    if set(entries)-{"1","2","3"} or any(str(i+1) not in entries for i in cfg.object_order):
        raise ValueError("Profiles do not cover the requested objects")
    if provided_perception is None:
        missing = [i+1 for i in cfg.object_order if not entries[str(i+1)].get("perception_model")]
        if missing:
            raise ValueError(f"No calibrated RGB profile declared for objects {missing}; Oracle is not an automatic fallback")
    profiles, metadata = {}, {}
    for obj in cfg.object_order:
        entry = entries[str(obj+1)]
        if set(entry)-{"planner_config","world_model","perception_model"}:
            raise ValueError("Unsupported per-object profile fields")
        motion = json.loads(resolve(entry["planner_config"]).read_text())
        if not isinstance(motion,dict) or set(motion)-MOTION_FIELDS:
            raise ValueError("Profile may only override documented motion fields")
        config = replace(cfg,**motion)
        residual = ReleaseWorldModel(resolve(entry["world_model"]),config)
        if obj+1 not in residual.object_ids:
            raise ValueError("World model does not support its assigned object")
        perception = provided_perception
        if perception is None:
            # Calibration scope is the active object, not the global sequence.
            perception = RGBPosePerception(resolve(entry["perception_model"]),
                                           replace(config,object_order=(obj,)))
        profiles[obj] = dict(config=config,perception=perception,residual=residual,joint_model=joint_model)
        metadata[str(obj+1)] = dict(throw_config=asdict(config),physics=physics_contract(config),
                                    world_model=str(resolve(entry["world_model"])),
                                    perception_model=entry.get("perception_model"),
                                    provided_perception=provided_perception is not None)
    return profiles, metadata
