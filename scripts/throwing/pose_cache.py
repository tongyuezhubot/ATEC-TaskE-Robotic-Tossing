"""Shared offline pose sampling/cache identity; no labels select the frames."""
import hashlib
import json
from pathlib import Path

HELD_STAGES = frozenset(("check_hold", "calibration_wrist", "plan", "pre_throw"))


def dense_episode(path, object_id):
    if object_id is None:
        return False
    meta = json.loads((Path(path).parent/"run_config.json").read_text())
    return meta["throw_config"]["object_order"][0]+1 == object_id


def cache_key(path, stride, dense_held_object=None):
    path = Path(path).resolve(); stat = path.stat()
    profile = (f"spatial_patch_rgb_denseheld_v1:object{dense_held_object}"
               if dense_episode(path,dense_held_object) else "spatial_patch_rgb_v1")
    return hashlib.sha256(f"{profile}:{path}:{stat.st_size}:{stat.st_mtime_ns}:stride{stride}".encode()).hexdigest()


def sample_indices(stages, stride, dense_held=False):
    if stride < 1:
        raise ValueError("Positive pose sampling stride required")
    selected = set(range(0,len(stages),stride))
    if dense_held:
        selected.update(i for i,stage in enumerate(stages) if stage in HELD_STAGES)
    return sorted(selected)
