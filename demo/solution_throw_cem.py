"""New local default: actor-free CEM throwing, no simulator truth in inference."""
import os
from dataclasses import replace
from .throw_cem.config import ThrowConfig
from .throw_cem.controller import ThrowController
from .throw_cem.perception import RGBPosePerception, numpy_observation
from .throw_cem.physics import ReleaseResidual
from .throw_cem.joint_model import JointResponse
from .throw_cem.release_world import ReleaseWorldModel


class AlgSolution:
    def __init__(self, perception=None, config=None):
        cfg = config or ThrowConfig()
        joint_model = JointResponse(cfg, os.environ.get("ATEC_THROW_JOINT_MODEL"))
        cfg = replace(cfg, joint_response_alpha=tuple(joint_model.alpha.tolist()))
        self.profile_metadata = {}
        if os.environ.get("ATEC_THROW_PROFILES"):
            from .throw_cem.profiles import load_profiles
            profiles,self.profile_metadata = load_profiles(os.environ["ATEC_THROW_PROFILES"],cfg,joint_model,perception)
            initial = profiles[cfg.object_order[0]]
            self.controller = ThrowController(initial["config"],initial["perception"],initial["residual"],
                                              joint_model,profiles=profiles)
            return
        if perception is None:
            perception = RGBPosePerception(os.environ.get("ATEC_THROW_PERCEPTION"), cfg)
        world_path = os.environ.get("ATEC_THROW_WORLD_MODEL")
        residual = (ReleaseWorldModel(world_path, cfg) if world_path else
                    ReleaseResidual(os.environ.get("ATEC_THROW_RESIDUAL"), cfg))
        self.controller = ThrowController(cfg, perception, residual, joint_model)

    def get_action_spec(self):
        return {}  # original 8-D position contract, no effort/physics override

    def reset(self):
        self.controller.reset()

    def predicts(self, obs, current_score):
        del current_score
        p, rgb = numpy_observation(obs)
        action, giveup = self.controller.action(p, rgb)
        return {"action": [action.tolist()], "giveup": bool(giveup)}

    def get_control_diagnostics(self, proprio=None):
        return self.controller.diagnostics()
