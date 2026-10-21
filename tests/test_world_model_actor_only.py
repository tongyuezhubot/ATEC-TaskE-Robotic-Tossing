"""Targeted action-head-only update and movement weighting contract."""
import unittest
import torch
from demo.world_model import WorldModelConfig
from demo.world_model_recurrent import RecurrentWorldModel
from scripts.world_model.pretrain_recurrent import freeze_actor_only, action_change_weights


class ActorOnlyTests(unittest.TestCase):
    def test_only_actor_changes_on_weighted_sequence_update(self):
        torch.manual_seed(7)
        m=RecurrentWorldModel(WorldModelConfig(image_size=32,image_feature_dim=24,latent_dim=16,hidden_dim=32))
        freeze_actor_only(m)
        before={name:p.detach().clone() for name,p in m.named_parameters()}
        p=torch.zeros(2,3,24);a=torch.ones(2,3,8)*.2
        image=torch.zeros(2,3,3,32,32,dtype=torch.uint8)
        batch=dict(proprio=p,next_proprio=p,image=image,next_image=image,action=a,
                   reward=torch.zeros(2,3),done=torch.zeros(2,3,dtype=torch.bool),
                   reward_valid=torch.zeros(2,3,dtype=torch.bool))
        batch=action_change_weights(batch)
        self.assertTrue(torch.all(batch["actor_weight"]==5))
        opt=torch.optim.AdamW([v for v in m.parameters() if v.requires_grad],lr=.001)
        loss=m.compute_loss(batch)["actor"]
        loss.backward();opt.step();m.update_target()
        changed=[name for name,v in m.named_parameters() if not torch.equal(v,before[name])]
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith("actor.") for name in changed),changed)
        p[:,:,16:24]=a
        self.assertTrue(torch.all(action_change_weights(batch)["actor_weight"]==1))


if __name__=="__main__":unittest.main()
