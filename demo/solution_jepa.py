"""Task E ACT proposals + visual progress + optional guarded JEPA scoring."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


PROJECT = Path(__file__).resolve().parents[1]


def preparation_modules():
    bundle = Path(os.environ.get("ATEC_JEPA_BUNDLE", PROJECT / "work/jepa_prepare_2026-10-09")).resolve()
    if not (bundle / "jepa_model.py").is_file():
        raise FileNotFoundError("Install the prepared JEPA bundle; do not install the full upstream environment")
    # Shared prepared modules are resolved from one explicit, verified location.
    sys.path.insert(0, str(bundle))
    from jepa_model import TaskEJEPA
    from progress_model import read_progress_checkpoint
    from hybrid_control import HybridController
    return TaskEJEPA, read_progress_checkpoint, HybridController


class ACTProposer:
    def __init__(self, paths, device):
        from .solution_act import Agent, Args
        if len(paths) not in (1, 3):
            raise ValueError("Use one whole-task ACT or three banana/mustard/sugar ACT checkpoints")
        self.device, self.policies = device, []
        for path in paths:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            norm = ckpt["norm_stats"]
            state_dim, act_dim = norm["state_mean"].shape[-1], norm["action_mean"].shape[-1]
            if state_dim not in (8, 11) or act_dim != 8:
                raise ValueError("ACT proposal checkpoint input/action dimensions are incompatible")
            weights = ckpt["ema_agent"]
            args = Args()
            args.num_queries = int(weights["model.query_embed.weight"].shape[0])
            args.include_rgb = any("backbone" in key for key in weights)
            if args.num_queries < 5:
                raise ValueError("ACT proposal must span a complete five-action macro interval")
            agent = Agent(state_dim, act_dim, args).to(device).eval()
            agent.load_state_dict(weights, strict=True)
            stats = {key: norm[key].to(device) for key in ("state_mean", "state_std", "action_mean", "action_std")}
            if not all(torch.isfinite(value).all() for value in stats.values()):
                raise ValueError("Non-finite ACT normalization")
            if torch.any(stats["state_std"] <= 0) or torch.any(stats["action_std"] <= 0):
                raise ValueError("Invalid ACT standard deviations")
            self.policies.append((agent, stats, state_dim))
        self.reset()

    def reset(self):
        # Retain multi-step intent within a skill, not a forbidden expert phase.
        self.sequence, self.cursor, self.sequence_phase = None, 0, None

    @torch.inference_mode()
    def propose(self, proprio, rgb, phase):
        agent, stats, state_dim = self.policies[phase if len(self.policies) == 3 else 0]
        cache_valid = self.sequence is not None and self.sequence_phase == phase and self.cursor < len(self.sequence)
        if cache_valid and self.cursor >= 50 and self.cursor % 50 == 0:
            # Preserve intentional static waits (e.g. expert INIT). Blindly
            # querying every 50 steps can restart the same wait indefinitely.
            target_rel = .5*self.sequence[self.cursor-1, :6]
            stalled = bool((proprio[0, :6]-target_rel).abs().max() > .15
                           and proprio[0, 8:14].abs().max() < .1)
            cache_valid = not stalled
        if cache_valid:
            result = self.sequence[self.cursor:self.cursor+5]
            self.cursor += len(result)
            if len(result) < 5:
                result = torch.cat((result, result[-1:].expand(5-len(result), -1)))
            return result
        qpos = proprio[:, :8] + proprio.new_tensor([0., 1.2, -1.5, 0., 1.2, 0., .035, -.035])
        if state_dim == 11:
            token = F.one_hot(torch.tensor([phase], device=self.device), num_classes=3).float()
            qpos = torch.cat((qpos, token), dim=-1)
        observation = {"state": (qpos-stats["state_mean"])/stats["state_std"]}
        if agent.include_rgb:
            image = rgb.permute(0, 3, 1, 2).float()
            # Agent handles /255 and ImageNet normalization, exactly once.
            image = F.interpolate(image, (224, 224), mode="bilinear", align_corners=False, antialias=True)
            observation["rgb"] = image[:, None]
        sequence = agent.get_action(observation)[0]
        self.sequence = sequence*stats["action_std"].reshape(1, 8)+stats["action_mean"].reshape(1, 8)
        self.cursor, self.sequence_phase = 5, phase
        return self.sequence[:5]


class AlgSolution:
    def __init__(self):
        TaskEJEPA, read_progress, HybridController = preparation_modules()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TaskEJEPA().to(self.device).eval()
        progress_path = os.environ.get("ATEC_JEPA_PROGRESS")
        if not progress_path:
            raise ValueError("Set ATEC_JEPA_PROGRESS to a trained, calibrated visual-progress checkpoint")
        progress, payload = read_progress(progress_path, self.device)
        if not payload.get("calibrated", False):
            raise RuntimeError("Visual progress has not passed its heldout calibration gate")
        scoring = os.environ.get("ATEC_JEPA_USE_SCORING", "0") == "1"
        contract = os.environ.get("ATEC_JEPA_SCORE_CONTRACT", "full_v1")
        if contract not in ("full_v1", "decision_outputs_v1"):
            raise ValueError("Unknown scoring contract")
        if scoring:
            checkpoint = os.environ.get("ATEC_JEPA_CHECKPOINT")
            if not checkpoint:
                raise ValueError("JEPA scoring requires an adapted checkpoint")
            wm = model.load_adapted(checkpoint)
            metrics = wm.get("validation", {})
            moving, baseline = metrics.get("moving_mse"), metrics.get("moving_persistence_mse")
            if moving is None or baseline is None or not 0 <= moving < baseline:
                raise RuntimeError("JEPA has not beaten the heldout moving persistence baseline")
            if not model.proprio_dynamics_loaded or not wm.get("dynamics_calibrated", False):
                raise RuntimeError("Scoring requires calibrated learned proprio and two-block rollout")
            if contract == "decision_outputs_v1":
                from decision_contract import decision_outputs_ready
                if not decision_outputs_ready(payload):
                    raise RuntimeError("Decision-output contract failed: real feedback and forecast scoring outputs must pass")
            elif payload.get("format") not in ("atec_jepa_progress_v2", "atec_jepa_progress_v3") or not payload.get("scoring_ready", False):
                raise RuntimeError("Task value must pass both real and predicted feature calibration")
            from prepare import sha256
            if payload.get("dynamics_sha256") != sha256(checkpoint):
                raise RuntimeError("Task head was fitted against a different dynamics checkpoint")
        paths = os.environ.get("ATEC_JEPA_ACT_POLICIES", "").strip()
        if paths:
            paths = [Path(path.strip()).resolve() for path in paths.split(",") if path.strip()]
        else:
            paths = [PROJECT / "runs" / name / "checkpoints/best_loss.pt" for name in (
                "act_q250_combined20_20k", "act_q250_mustard20_10k", "act_q250_sugar20_10k")]
        proposer = ACTProposer(paths, self.device)
        self.controller = HybridController(model, progress, proposer, scoring=scoring,
                                           elevation_feedback=payload.get("elevated_calibrated", False),
                                           scoring_contract=contract)
        self._reported_scoring_state = None

    def reset(self):
        self.controller.reset()
        self._reported_scoring_state = None

    def get_action_spec(self):
        return {}

    def predicts(self, obs, current_score):
        del current_score
        if not isinstance(obs, dict) or "proprio" not in obs or "video_rgb" not in obs.get("image", {}):
            raise ValueError("Only standard proprio and video_rgb are accepted")
        p = obs["proprio"].to(self.device).float()
        if p.shape != (1, 24):
            raise ValueError("JEPA hybrid currently supports exactly one Task E environment")
        image = obs["image"]["video_rgb"].to(self.device)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("One RGB image batch required")
        if image.shape[1] in (3, 4):
            image = image.permute(0, 2, 3, 1)
        image = image[..., :3]
        if image.shape[-1] != 3:
            raise ValueError("RGB camera layout is invalid")
        if image.dtype != torch.uint8:
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("Float RGB must be in [0,1]")
            image = (255*image).round().to(torch.uint8)
        small = F.interpolate(image.permute(0, 3, 1, 2).float(), (96, 96), mode="bilinear",
                              align_corners=False, antialias=True).round().clamp(0, 255).to(torch.uint8)
        small = small.permute(0, 2, 3, 1)[:, None]
        action, giveup = self.controller.action(p[0], small, image)
        if self.controller.scoring_decisions > 0 and self._reported_scoring_state != self.controller.scoring_active:
            print(f"[WM_SCORING] active={self.controller.scoring_active} contract={self.controller.scoring_contract} "
                  f"decisions={self.controller.scoring_decisions} reason={self.controller.scoring_reason}", flush=True)
            self._reported_scoring_state = self.controller.scoring_active
        return {"action": [action.cpu().tolist()], "giveup": giveup}

    def get_control_diagnostics(self, proprio):
        return self.controller.diagnostics()
