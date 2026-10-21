"""Capture training transitions before Isaac Lab's automatic episode reset."""
from __future__ import annotations

import numpy as np
import torch

from demo.world_model import resize_rgb_uint8


def policy_snapshot(obs, image_size):
    proprio = obs["proprio"].detach().float().cpu().clone()
    if proprio.shape != (1, 24):
        raise ValueError("Aligned recording currently requires one env and 24-D proprio")
    image = resize_rgb_uint8(obs["image"]["video_rgb"], image_size).clone()
    return {"proprio": proprio[0].numpy(), "rgb": image[0].permute(1, 2, 0).numpy()}


def truth_snapshot(base):
    result = {}
    # These are the exact one-time reward counters; no extra reward call occurs.
    for name, key in (("grasped_objects_once", "grasped"), ("objects_in_basket", "inside")):
        term = base.reward_manager.get_term_cfg(name).func
        result[key] = term._counted[0].detach().cpu().numpy().copy()
    result["object_pos"] = np.stack([
        (base.scene[f"object_{i}"].data.root_pos_w[0] - base.scene.env_origins[0])
        .detach().cpu().numpy().copy() for i in (1, 2, 3)
    ])
    # Isaac 5.0's iterable/get_term accessors retain the last episode's done
    # reason. Evaluate the configured pure geometry predicate for THIS state.
    term = base.termination_manager.get_term_cfg("basket_success")
    result["basket_success"] = bool(term.func(base, **term.params)[0].item())
    return result


def aligned_step(env, action, image_size=96, expert_stage=""):
    """Return the regular env.step result plus a separate training-only record.

    Single-env only. The instance reset method is restored even if step fails.
    The terminal next observation is captured after physics/rewards and before
    reset, instead of treating the returned reset observation as its target.
    """
    base = env.unwrapped
    if base.num_envs != 1:
        raise ValueError("Use one environment for aligned transition recording")
    current = policy_snapshot(base.observation_manager.compute(update_history=False), image_size)
    before = truth_snapshot(base)
    original_reset = base._reset_idx
    had_override = "_reset_idx" in base.__dict__
    previous_override = base.__dict__.get("_reset_idx")
    terminal = {}

    def capture_reset(env_ids):
        if len(env_ids):
            terminal["policy"] = policy_snapshot(
                base.observation_manager.compute(update_history=False), image_size
            )
            terminal["truth"] = truth_snapshot(base)
        return original_reset(env_ids)

    base._reset_idx = capture_reset
    try:
        result = env.step(action)
    finally:
        if had_override:
            base._reset_idx = previous_override
        else:
            del base.__dict__["_reset_idx"]
    obs, reward, terminated, truncated, info = result
    ended = bool(terminated.any() or truncated.any())
    next_policy = terminal.get("policy")
    after = terminal.get("truth")
    if next_policy is None:
        next_policy = policy_snapshot(obs, image_size)
        after = truth_snapshot(base)
    record = {
        "policy_input": current,
        "next_policy_input": next_policy,
        "action": action[0].detach().cpu().numpy().copy(),
        "reward": float(reward.reshape(-1)[0]),
        "terminated": bool(terminated.reshape(-1)[0]),
        "truncated": bool(truncated.reshape(-1)[0]),
        "next_obs_valid": not ended or bool(terminal),
        "training_labels": {
            "grasped_event": after["grasped"] & ~before["grasped"],
            "in_basket_event": after["inside"] & ~before["inside"],
            "basket_success": after["basket_success"],
            "object_pos": after["object_pos"],
            "expert_stage": expert_stage,
        },
    }
    return result, record


def save_aligned_group(group, records):
    """Keep deployable observations separate from privileged supervision."""
    import h5py

    group.attrs["schema_version"] = 2
    for name in ("policy_input", "next_policy_input"):
        target = group.create_group(name)
        for field in ("proprio", "rgb"):
            data = np.stack([r[name][field] for r in records])
            kwargs = {"chunks": (1, *data.shape[1:])} if field == "rgb" else {}
            target.create_dataset(field, data=data, compression="gzip", **kwargs)
    for name in ("reward", "terminated", "truncated", "next_obs_valid"):
        group.create_dataset(name, data=[r[name] for r in records])
    group.create_dataset("episode_start", data=[True] + [False] * (len(records) - 1))
    labels = group.create_group("training_labels")
    labels.attrs["deployment_forbidden"] = True
    labels.attrs["basket_success_source"] = "current_term"
    for name in ("grasped_event", "in_basket_event", "basket_success", "object_pos"):
        labels.create_dataset(name, data=np.stack([r["training_labels"][name] for r in records]))
    labels.create_dataset("expert_stage",
        data=[r["training_labels"]["expert_stage"] for r in records],
        dtype=h5py.string_dtype("utf-8"))

    # Recovery labels remain training-only; none are deployment observations.
    for name in ("active_object","recovery_attempt","fault_applied","actor_valid"):
        if name in records[0]["training_labels"]:
            labels.create_dataset(name,data=[r["training_labels"][name] for r in records])
    if "fault_request" in records[0]["training_labels"]:
        labels.create_dataset("fault_request",
            data=[r["training_labels"]["fault_request"] for r in records],
            dtype=h5py.string_dtype("utf-8"))
