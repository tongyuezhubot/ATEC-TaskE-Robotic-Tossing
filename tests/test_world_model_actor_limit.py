"""Candidate bound conversion must survive save/load; never mutates old files."""
from pathlib import Path
import tempfile
import unittest
import torch
from demo.world_model import WorldModelConfig,save_world_model,load_world_model
from demo.world_model_recurrent import RecurrentWorldModel
from scripts.world_model.pretrain_recurrent import configure_action_limit

class BoundTests(unittest.TestCase):
    def test_actor_and_inverse_candidate_limits_roundtrip(self):
        m=RecurrentWorldModel(WorldModelConfig(image_size=32,image_feature_dim=24,latent_dim=16,hidden_dim=32))
        configure_action_limit(m,3.5)
        self.assertEqual(m.config.action_limit,3.5)
        self.assertEqual(m.actor[-1].limit,3.5)
        self.assertEqual(m.inverse_head[-1].limit,3.5)
        h=torch.randn(2,16);pair=torch.cat((h,h),-1)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"candidate.pt";save_world_model(p,m)
            restored,_=load_world_model(p)
            self.assertTrue(torch.equal(m.actor(h),restored.actor(h)))
            self.assertTrue(torch.equal(m.inverse_head(pair),restored.inverse_head(pair)))

if __name__=="__main__":unittest.main()
