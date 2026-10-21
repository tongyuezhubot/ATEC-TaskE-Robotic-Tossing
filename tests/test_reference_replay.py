"""Same-command reference replay checks; no simulator or RGB model needed."""
from dataclasses import asdict,replace
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from demo.throw_cem.config import ThrowConfig,DEFAULT_Q
from demo.throw_cem.kinematics import PiperKinematics
from demo.throw_cem.planner import ThrowPlanner
from demo.throw_cem.joint_model import JointResponse
from demo.throw_cem.physics import physics_contract,predict_driven_release,ReleaseResidual
from scripts.throwing.analyze_toss_batch import replay_release_reference,reference_contract,release_dynamics_contract


class ReferenceReplayTests(unittest.TestCase):
    def fixture(self,root):
        cfg = ThrowConfig(throw_primitive="joint_swing",object_order=(0,),
                          joint_swing_duration=.60,joint_response_alpha=(.18,)*6,
                          planning_min_remaining_apex=0.,lift_clearance=.10,
                          joint_swing_lower=(-.45,-.85,-.85,.4))
        q=np.array([-.558150649,2.139956427,-2.004052639,-.546047688,
                    1.141624880,1.297532916,.019801039,-.021412298])
        theta=np.array([.9,-.716997783,-.85,.524251937]);offset=np.array([-.022911739,.031668684,-.133770182])
        kin=PiperKinematics(cfg);trajectory,nominal=ThrowPlanner(cfg,kin).joint_swing_trajectory(q,theta,0)
        pred,_=predict_driven_release(cfg,0,q,offset,theta,nominal,trajectory.targets,
                                     kin,JointResponse(cfg),ReleaseResidual())
        old=np.r_[pred.release_position,pred.release_velocity]
        newcfg=replace(cfg,gripper_delay_prior=0.)
        _,newnom=ThrowPlanner(newcfg,kin).joint_swing_trajectory(q,theta,0)
        newpred,_=predict_driven_release(newcfg,0,q,offset,theta,newnom,trajectory.targets,
                                        kin,JointResponse(newcfg),ReleaseResidual())
        new=np.r_[newpred.release_position,newpred.release_velocity]
        actual=new+np.array([.001,.002,-.003,.01,.02,-.03])
        sample={"source":str(root),"q_start":q.tolist(),"theta":theta.tolist(),"offset":offset.tolist(),
                "object":1,"plan":1,"nominal_release_state":old.tolist(),"delta":(actual-old).tolist()}
        (root/"run_config.json").write_text(json.dumps({"throw_config":asdict(cfg),"physics":physics_contract(cfg)}))
        rows=[{"action_stage":"swing","controller":{"cem_plans":1,"object":1},
               "action":((target-DEFAULT_Q)/cfg.action_scale).tolist()} for target in trajectory.targets]
        (root/"trace.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        return sample,new,actual

    def test_preserves_observed_label_source_files_and_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sample,new,actual=self.fixture(root);before=copy.deepcopy(sample)
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            result=replay_release_reference(sample,0.)
            np.testing.assert_allclose(result["nominal_release_state"],new,atol=1e-8)
            np.testing.assert_allclose(np.array(result["nominal_release_state"])+result["delta"],actual)
            self.assertEqual(sample,before)
            self.assertEqual(hashes,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})

    def test_rejects_changed_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sample,_,_=self.fixture(root)
            rows=[json.loads(x) for x in (root/"trace.jsonl").read_text().splitlines()]
            rows[0]["action"][0]+=.01
            (root/"trace.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaisesRegex(ValueError,"executed commands"):
                replay_release_reference(sample,0.)

    def test_contract_normalizes_only_explicit_delay(self):
        a=physics_contract(ThrowConfig());b=copy.deepcopy(a);b["gripper_delay_prior"]=0.
        self.assertNotEqual(reference_contract(a,None),reference_contract(b,None))
        self.assertEqual(reference_contract(a,0.),reference_contract(b,0.))
        b["lift_clearance"]+=.01
        self.assertNotEqual(release_dynamics_contract(reference_contract(a,0.)),
                            release_dynamics_contract(reference_contract(b,0.)))


if __name__=="__main__":
    unittest.main()
