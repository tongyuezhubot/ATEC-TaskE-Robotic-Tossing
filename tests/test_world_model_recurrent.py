"""One focused suite for alignment, reset, and sequence train/inference contracts."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from demo.world_model import WorldModelConfig, save_world_model, load_world_model
from demo.world_model_recurrent import RecurrentWorldModel
from demo.world_model_sequence import ExpertSequenceDataset
from demo.world_model_sequence_replay import SequenceReplay
from scripts.world_model.aligned_transitions import aligned_step, save_aligned_group, truth_snapshot
from scripts.world_model.pretrain_recurrent import EarlyStopping, freeze_action_prior
from scripts.world_model.repair_aligned_labels import basket_success_from_positions


class RecurrentTests(unittest.TestCase):
    def model(self):
        return RecurrentWorldModel(WorldModelConfig(
            image_size=32,image_feature_dim=24,latent_dim=16,hidden_dim=32))

    def test_memory_reset_sequence_update_checkpoint(self):
        model = self.model()
        proprio = torch.randn(2,24)
        image = torch.randint(0,256,(2,3,32,32),dtype=torch.uint8)
        first = model.observe(proprio,image)
        remembered = model.observe(proprio,image,first)
        self.assertFalse(torch.allclose(first,remembered))
        reset = model.observe(proprio,image,remembered,torch.ones(2,dtype=torch.bool))
        self.assertTrue(torch.allclose(first,reset))
        batch = {
            "proprio": proprio[:,None].expand(-1,3,-1),
            "next_proprio": proprio[:,None].expand(-1,3,-1),
            "image": image[:,None].expand(-1,3,-1,-1,-1),
            "next_image": image[:,None].expand(-1,3,-1,-1,-1),
            "action": torch.ones(2,3,8)*0.2,
            "reward": torch.zeros(2,3),
            "done": torch.zeros(2,3,dtype=torch.bool),
            "reward_valid": torch.zeros(2,3,dtype=torch.bool),
        }
        losses=model.compute_loss(batch)
        self.assertEqual(float(losses["reward"]),0)
        self.assertEqual(float(losses["value"]),0)
        losses["total"].backward()
        self.assertIsNotNone(model.belief_cell.weight_ih.grad)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        optimizer.step()
        model.update_target()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"policy.pt"
            save_world_model(path,model)
            restored,payload=load_world_model(path)
            self.assertEqual(payload["format_version"],2)
            self.assertTrue(torch.allclose(model.observe(proprio,image),restored.observe(proprio,image)))


    def test_outcome_adaptation_preserves_all_prior_parameters(self):
        model=self.model()
        freeze_action_prior(model)
        before={name:p.detach().clone() for name,p in model.named_parameters()
                if not name.startswith(("reward_head.","value_head.","event_head.","target_"))}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.001)
        p=torch.randn(2,24); image=torch.zeros(2,3,32,32,dtype=torch.uint8)
        belief=model.observe(p,image)
        actions=model.actor(belief)
        loss=model.predict_reward(belief,actions).square().mean()
        loss=loss+model.predict_value(belief).square().mean()
        loss=loss+model.event_head(torch.cat((belief,actions),-1)).square().mean()
        loss.backward(); optimizer.step(); model.update_target()
        self.assertIsNotNone(model.reward_head[0].weight.grad)
        self.assertIsNotNone(model.event_head.weight.grad)
        for name,value in before.items():
            self.assertTrue(torch.equal(value,dict(model.named_parameters())[name]),name)
        self.assertTrue(torch.equal(actions,model.actor(model.observe(p,image))))

    def test_success_geometry_does_not_reuse_stale_cache(self):
        params=dict(center=(1.08,-0.3,0.97),half_x=0.20,half_y=0.11,table_top_z=0.8266)
        positions=np.tile(np.array([1.08,-0.3,0.9],dtype=np.float32),(3,3,1))
        positions[0,0,0]=0.5
        positions[1,1,2]=0.7
        self.assertEqual(basket_success_from_positions(positions,params).tolist(),[False,False,True])

    def test_early_stopping_significant_improvement_and_warmup(self):
        stop=EarlyStopping(patience=3,min_delta=0.001,min_steps=300)
        self.assertFalse(stop.update(0.2,1))
        self.assertFalse(stop.update(0.1998,100))
        self.assertFalse(stop.update(0.1996,200))
        self.assertTrue(stop.update(0.1995,300))
        stop=EarlyStopping(patience=3,min_delta=0.001,min_steps=0)
        stop.update(0.2,1)
        stop.update(0.201,100)
        self.assertFalse(stop.update(0.198,200))
        self.assertEqual(stop.bad_checks,0)

    def test_sequence_replay_wrap_and_episode_boundaries(self):
        model=self.model()
        replay=SequenceReplay(5,model.config)
        for step in range(9):
            p=torch.ones(1,24)*step
            i=torch.zeros(1,3,32,32,dtype=torch.uint8)
            replay.add(p,i,torch.zeros(1,8),torch.zeros(1),p+1,i,
                torch.tensor([step%3==2]),terminated=torch.tensor([step%3==2]))
        batch=replay.sample_sequences(4,3,"cpu")
        self.assertTrue(torch.all(batch["proprio"][:,:,0] == torch.tensor([6.,7.,8.])))
        self.assertFalse(bool(batch["done"][:,:-1].any()))
        self.assertFalse(bool(batch["expert"].any()))

    def test_pre_reset_capture_and_label_isolation(self):
        class FakeEnv:
            def __init__(self):
                self.unwrapped=self
                self.num_envs=1
                self.p=torch.zeros(1,24)
                self.i=torch.zeros(1,24,32,3,dtype=torch.uint8)
                self.grasp=SimpleNamespace(_counted=torch.zeros(1,3,dtype=torch.bool))
                self.inside=SimpleNamespace(_counted=torch.zeros(1,3,dtype=torch.bool))
                self.reward_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(
                    func=self.grasp if name=="grasped_objects_once" else self.inside))
                self.observation_manager=SimpleNamespace(compute=lambda **kwargs:self.obs())
                self.success=False
                self.termination_manager=SimpleNamespace(
                    get_active_iterable_terms=lambda i:[("basket_success",[True])],
                    get_term_cfg=lambda name: SimpleNamespace(
                        func=lambda env: torch.tensor([env.success]), params={}))
                self.scene={f"object_{i}":SimpleNamespace(data=SimpleNamespace(
                    root_pos_w=torch.ones(1,3)*i)) for i in (1,2,3)}
                class Scene(dict): pass
                self.scene=Scene(self.scene)
                self.scene.env_origins=torch.zeros(1,3)
            def obs(self):
                return {"proprio":self.p,"image":{"video_rgb":self.i}}
            def _reset_idx(self,ids):
                self.p.fill_(0); self.i.fill_(0)
                self.grasp._counted.zero_(); self.inside._counted.zero_()
                self.success=False
            def step(self,action):
                self.p.fill_(7); self.p[:,16:24]=action
                self.i.fill_(100)
                self.grasp._counted.fill_(True); self.inside._counted.fill_(True)
                self.success=True
                self._reset_idx(torch.tensor([0]))
                return self.obs(),torch.tensor([0.06]),torch.tensor([True]),torch.tensor([False]),{}
        env=FakeEnv()
        self.assertFalse(truth_snapshot(env)["basket_success"])  # stale done cache is True
        result,record=aligned_step(env,torch.ones(1,8),32,"CLOSE")
        self.assertEqual(float(record["next_policy_input"]["proprio"][0]),7)
        self.assertEqual(float(result[0]["proprio"][0,0]),0)
        self.assertTrue(record["next_obs_valid"])
        self.assertTrue(record["training_labels"]["basket_success"])
        self.assertNotIn("_reset_idx",env.__dict__)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"data.h5"
            with h5py.File(path,"w") as f:
                g=f.create_group("traj_0")
                save_aligned_group(g,[record]*3)
                g.create_dataset("obs",data=np.zeros((3,8)))
                g.create_dataset("qvel",data=np.zeros((3,8)))
                g.create_dataset("actions",data=np.ones((3,8)))
                g.create_group("images")["rgb"]=g["policy_input/rgb"]
                self.assertEqual(set(g["policy_input"]),{"proprio","rgb"})
            # One terminal transition per episode is required by real collectors;
            # use this synthetic file only to check schema reading.
            ds=ExpertSequenceDataset(path,image_size=32,sequence_length=2,burn_in=0)
            sample=ds[0]; ds.close()
            self.assertEqual(sample["events"].shape,(2,7))
            self.assertTrue(bool(sample["events_valid"].all()))


if __name__=="__main__":
    unittest.main()
