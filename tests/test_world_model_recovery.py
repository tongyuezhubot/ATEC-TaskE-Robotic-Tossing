"""Focused fault bounds, retry lifecycle, and actor-mask/data-loader regression."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from demo.world_model_sequence import ExpertSequenceDataset
from scripts.world_model.aligned_transitions import save_aligned_group

spec=importlib.util.spec_from_file_location(
    "training_recovery", Path(__file__).resolve().parents[1]/"scripts/act/task_e/recovery.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RecoveryController=module.RecoveryController


class RecoveryTests(unittest.TestCase):
    def controller(self,kind="miss_grasp"):
        return RecoveryController((kind,3,1),[3,2,1],
            dict(center=(1.08,-.3,.98),half_x=.20,half_y=.11,table_top_z=.82664),
            (1.,0.),(.404,.568))

    def positions(self):
        return np.array([[1.,.27,.88],[1.,.17,.90],[1.,.06,.87]],dtype=np.float32)

    def test_miss_disabled_after_one_bounded_retry(self):
        c=self.controller();pos=self.positions();p=np.array([1.,.06,.98])
        out,g,bad=c.adjust(3,"DESCEND",0,p,"open",pos)
        self.assertTrue(bad)
        self.assertAlmostEqual(float(out[1]),.15,places=5)
        self.assertEqual(c.check(3,"LIFT",True,pos,[1.,.06,1.23]),"retry")
        self.assertEqual(c.retries[3],1)
        out,g,bad=c.adjust(3,"DESCEND",0,p,"open",pos)
        self.assertFalse(bad)
        self.assertTrue(np.allclose(out,p))
        self.assertEqual(c.check(3,"LIFT",True,pos,[1.,.06,1.23]),"skip")
        self.assertEqual(c.retries[3],1)

    def test_slip_requires_low_height_over_support(self):
        c=self.controller("slip");pos=self.positions();target=np.array([1.,.06,1.22])
        self.assertEqual(c.adjust(3,"LIFT",70,target,"close",pos)[1],"close")
        pos[2,2]=1.10
        self.assertEqual(c.adjust(3,"LIFT",70,target,"close",pos)[1],"open")
        self.assertTrue(c.applied)
        other=self.controller("slip");pos[2,0]=.2
        self.assertEqual(other.adjust(3,"LIFT",70,target,"close",pos)[1],"close")

    def test_missed_place_is_supported_clear_and_outside_goal(self):
        c=self.controller("place_miss");pos=self.positions()
        out,g,bad=c.adjust(3,"PLACE",0,[1.08,-.3,.94],"close",pos)
        self.assertTrue(bad)
        self.assertTrue(c.in_support(out[:2]))
        self.assertFalse(c.inside(out))
        self.assertTrue(all(np.linalg.norm(out[:2]-x[:2])>=.22 for x in pos[:2]))

    def test_verified_retry_mask_survives_hdf5_and_loader(self):
        c=self.controller()
        def record(obj,attempt):
            r=dict(policy_input=dict(proprio=np.zeros(24,np.float32),rgb=np.zeros((32,32,3),np.uint8)),
                   next_policy_input=dict(proprio=np.zeros(24,np.float32),rgb=np.zeros((32,32,3),np.uint8)),
                   reward=0.,terminated=False,truncated=False,next_obs_valid=True,
                   training_labels=dict(grasped_event=np.zeros(3,bool),in_basket_event=np.zeros(3,bool),
                    basket_success=False,object_pos=self.positions(),expert_stage="CLOSE"))
            c.annotate(r,obj,attempt,False)
            return r
        records=[record(3,0),record(3,1),record(2,0)]
        c.retries[3]=1
        c.successful_attempts[3]=1
        report=c.finalize(records)
        self.assertEqual(report["actor_valid_steps"],1)
        self.assertEqual([r["training_labels"]["actor_valid"] for r in records],[False,True,False])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"data.h5"
            with h5py.File(path,"w") as f:
                g=f.create_group("traj_0")
                g.attrs["actor_supervision"]=True
                g.attrs["requires_actor_valid_mask"]=True
                save_aligned_group(g,records)
                g.create_dataset("obs",data=np.zeros((3,8)))
                g.create_dataset("qvel",data=np.zeros((3,8)))
                g.create_dataset("actions",data=np.zeros((3,8)))
                g.create_group("images")["rgb"]=g["policy_input/rgb"]
                self.assertEqual(set(g["policy_input"]),{"proprio","rgb"})
                self.assertTrue(g["training_labels"].attrs["deployment_forbidden"])
            ds=ExpertSequenceDataset(path,image_size=32,sequence_length=3,burn_in=0)
            sample=ds[0];ds.close()
            self.assertEqual(sample["expert"].tolist(),[False,True,False])
            self.assertNotIn("active_object",sample)
            self.assertNotIn("fault_request",sample)


if __name__=="__main__":
    unittest.main()
