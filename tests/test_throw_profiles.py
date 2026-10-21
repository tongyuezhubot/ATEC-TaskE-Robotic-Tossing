"""Focused profile handoff proofs, not simulator task success."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from demo.throw_cem.config import ThrowConfig,HOME_Q,DEFAULT_Q
from demo.throw_cem.controller import ThrowController
from demo.throw_cem.joint_model import JointResponse
from demo.throw_cem.physics import ReleaseResidual
from demo.throw_cem.perception import VisualPose
from demo.throw_cem.profiles import load_profiles


class Poses:
    def __init__(self,cfg):self.cfg=cfg
    def observe(self,p,rgb):
        xyz=np.tile([*self.cfg.basket_xyz[:2],self.cfg.table_z+.05],(3,1))
        return VisualPose(xyz,np.tile([1.,0.,0.,0.],(3,1)),np.ones(3))


class ProfileTests(unittest.TestCase):
    def test_handoff_preserves_episode_and_reset_restores_first_profile(self):
        base=ThrowConfig(object_order=(2,1,0),throw_primitive="joint_swing")
        profiles={}
        for obj,lift,duration in ((2,.20,.60),(1,.12,.57),(0,.10,.60)):
            cfg=replace(base,lift_clearance=lift,joint_swing_duration=duration)
            profiles[obj]=dict(config=cfg,perception=Poses(cfg),residual=ReleaseResidual(),joint_model=JointResponse(cfg))
        first=profiles[2];c=ThrowController(first["config"],first["perception"],profiles=profiles)
        p=np.zeros(24);p[:8]=HOME_Q-DEFAULT_Q
        for phase,obj in enumerate((2,1,0)):
            c.enter("wait_landing");c.target=HOME_Q.copy();c.stable=c.cfg.settle_steps-1
            c.release_visual_xyz=np.array([*base.basket_xyz[:2],base.table_z+.05])+[.1,0.,0.]
            c.upward_seen=c.separated_seen=True;c.plans+=1;c.candidates+=240
            c.planning_q=HOME_Q.copy();c.stance_alignment_attempted=True
            _,done=c.action(p,np.zeros((224,224,3),np.uint8))
            self.assertTrue(c.visual_success[obj])
            self.assertIn(str(obj+1),c.completed_profiles)
            self.assertEqual(c.step,phase+1)
            if phase<2:
                self.assertFalse(done);self.assertEqual(c.stage,"home")
                next_obj=(2,1,0)[phase+1]
                self.assertIs(c.planner.residual,profiles[next_obj]["residual"])
                self.assertIs(c.perception,profiles[next_obj]["perception"])
                self.assertIsNone(c.poses);self.assertFalse(c.stance_alignment_attempted)
                self.assertFalse(hasattr(c,"planning_q"))
            else:self.assertTrue(done)
        self.assertEqual(c.visual_success,[True,True,True])
        c.reset();self.assertEqual(c.phase,0);self.assertEqual(c.stage,"startup")
        self.assertEqual(c.cfg.lift_clearance,.20);self.assertEqual(c.completed_profiles,{})

    def test_sugar_can_enter_domain_without_reaching_distant_training_pose(self):
        cfg=ThrowConfig(object_order=(0,),sugar_centered_grasp=True,normalize_throw_stance=True)
        c=ThrowController(cfg,None);q=HOME_Q.copy();original=q.copy()
        training=q[:6].copy();training[0]+=.5
        bounds=np.c_[q[:6]-1.,q[:6]+1.];bounds[0]=[q[0]+.1,q[0]+.9]
        def check(obj,state,offset):
            if state[0]<bounds[0,0]:raise ValueError("outside")
            return np.zeros(4),np.ones(4)
        c.planner.residual=SimpleNamespace(world_model=True,bounds=bounds,
            candidate_stances=lambda obj,state:np.array([training]),candidate_bounds=check)
        goal=c.select_training_stance(q,0,np.array([0.,0.,-.12]))
        self.assertIsNotNone(goal);self.assertLessEqual(np.max(np.abs(goal-q[:6])),.35)
        np.testing.assert_array_equal(q,original)
        self.assertGreaterEqual(goal[0],bounds[0,0]+.03)
        self.assertFalse(np.array_equal(goal,training))
        target=q.copy();target[:6]=goal
        self.assertLessEqual(np.linalg.norm(c.kin.forward(target)[:3,3]-c.kin.forward(q)[:3,3]),.05)
        c.cfg=replace(cfg,normalize_throw_stance=False)
        self.assertIsNone(c.select_training_stance(q,0,np.array([0.,0.,-.12])))

    def test_missing_rgb_profile_never_falls_back_to_oracle(self):
        cfg=ThrowConfig(object_order=(0,))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"profiles.json"
            p.write_text(json.dumps({"format":"atec_throw_profiles_v1","objects":{"1":{
                "planner_config":"unused","world_model":"unused","perception_model":None}}}))
            with self.assertRaisesRegex(ValueError,"No calibrated RGB profile"):
                load_profiles(p,cfg,JointResponse(cfg))

    def test_wrong_object_world_model_is_rejected(self):
        cfg=ThrowConfig(object_order=(0,))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);motion=root/"motion.json";motion.write_text('{}')
            p=root/"profiles.json"
            p.write_text(json.dumps({"format":"atec_throw_profiles_v1","objects":{"1":{
                "planner_config":str(motion),"world_model":"unused","perception_model":None}}}))
            with patch("demo.throw_cem.profiles.ReleaseWorldModel",return_value=SimpleNamespace(object_ids=[3])):
                with self.assertRaisesRegex(ValueError,"assigned object"):
                    load_profiles(p,cfg,JointResponse(cfg),Poses(cfg))


if __name__=="__main__":unittest.main()
