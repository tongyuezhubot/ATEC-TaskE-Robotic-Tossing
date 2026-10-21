"""Create a non-destructive corrected label view; original RGB/data are external links."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def task_geometry(cfg_path):
    # Read the task's scalar arithmetic without importing/starting Isaac Sim.
    text = Path(cfg_path).read_text()
    namespace = {"__builtins__": {}, "tuple": tuple}
    for node in ast.parse(text).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith(("TABLE_", "BASKET_"))):
            namespace[node.targets[0].id] = eval(
                compile(ast.Expression(node.value), "<task geometry>", "eval"), namespace
            )
    return dict(center=namespace["BASKET_SUCCESS_CENTER"],
                half_x=namespace["BASKET_SUCCESS_HALF_X"],
                half_y=namespace["BASKET_SUCCESS_HALF_Y"],
                table_top_z=namespace["TABLE_TOP_Z"]), hashlib.sha256(text.encode()).hexdigest()


def basket_success_from_positions(positions, params):
    # Same float32 inclusive geometric predicate as ObjectsInBasketDone.__call__.
    center = np.asarray(params["center"][:2], dtype=np.float32)
    z = params["table_top_z"]
    inside = ((np.abs(positions[..., 0]-center[0]) <= params["half_x"])
              & (np.abs(positions[..., 1]-center[1]) <= params["half_y"])
              & (positions[..., 2] >= z) & (positions[..., 2] <= z+0.15))
    return inside.all(axis=-1)


def repair(source, output, cfg_path):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source == output:
        raise FileExistsError("Choose a new corrected output; originals are preserved")
    params, cfg_hash = task_geometry(cfg_path)
    corrected, summaries = {}, []
    with h5py.File(source, "r") as f:
        for key in f:
            g = f[key]
            positions = g["training_labels/object_pos"][...]
            if not np.isfinite(positions).all():
                raise ValueError(f"{key}: nonfinite object positions")
            success = basket_success_from_positions(positions, params)
            terminated = g["terminated"][...]
            if not np.array_equal(success, terminated):
                raise ValueError(f"{key}: geometry vs terminated mismatch; manual diagnosis required")
            if bool(success[-1]) != bool(g.attrs["success"]):
                raise ValueError(f"{key}: final success attribute mismatch")
            old = g["training_labels/basket_success"][...]
            corrected[key] = success
            summaries.append(dict(episode=key, steps=len(success), success=bool(success[-1]),
                                  corrected_steps=int((old != success).sum())))
        # Write only corrected labels/boundaries; keep original observations as links.
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, "x") as out:
            out.attrs["original_dataset"] = str(source)
            out.attrs["task_cfg_sha256"] = cfg_hash
            out.attrs["success_geometry"] = json.dumps(params)
            for key in f:
                src = f[key]
                g = out.create_group(key)
                for name, value in src.attrs.items():
                    g.attrs[name] = value
                success = corrected[key]
                g.attrs["actor_supervision"] = bool(success[-1])
                g.attrs["episode_end_reason"] = str(src.attrs["termination_reason"])
                for name in src:
                    if name not in ("training_labels", "truncated"):
                        g[name] = h5py.ExternalLink(str(source), f"/{key}/{name}")
                truncated = src["truncated"][...]
                if not success[-1]:
                    truncated[-1] = True  # collector ended; not a task terminal failure
                g.create_dataset("truncated", data=truncated)
                labels = g.create_group("training_labels")
                labels.attrs["deployment_forbidden"] = True
                labels.attrs["basket_success_source"] = "recomputed_object_pos"
                for name in src["training_labels"]:
                    if name != "basket_success":
                        labels[name] = h5py.ExternalLink(str(source), f"/{key}/training_labels/{name}")
                labels.create_dataset("basket_success", data=success)
    print(json.dumps(dict(output=str(output), episodes=summaries), ensure_ascii=False), flush=True)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task_cfg", default="source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py")
    a = p.parse_args()
    repair(a.source, a.output, a.task_cfg)


if __name__ == "__main__":
    main()
