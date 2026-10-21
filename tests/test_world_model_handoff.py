"""Only the deployment handoff/reset/rate contracts; no Isaac launch."""
from types import SimpleNamespace
import unittest
import torch
from demo.solution_world_model import AlgSolution


class HandoffTests(unittest.TestCase):
    def solution(self):
        s=AlgSolution.__new__(AlgSolution)
        s.device=torch.device("cpu")
        s.default_joint_pos=torch.tensor([[0.,1.2,-1.5,0.,1.2,0.,.035,-.035]])
        s.teleop_home_joint_pos=torch.tensor([[0.,.924525,-1.514983,0.,1.2199,0.,.035,-.035]])
        s.teleop_home_action=torch.tensor([[0.,-.55093,-.03000,0.,.44,0.,0.,0.]])
        s.belief_burn_in=4; s.belief_warmup_step=0
        s.home_steps=0; s.home_timeout=300
        s.max_action_delta=torch.tensor([[.05]*6+[.02]*2])
        s.startup_zero_steps=25; s.startup_step=25
        s.home_done=False; s.last_action=None; s.belief=None
        s.is_recurrent=True; s.outcome_supervised=False; s.control_mode="actor"
        s.control_phase="home"; s.planner=SimpleNamespace(previous_action=None)
        s.observe_calls=0; s.actor_calls=0
        def observe(p,image,belief):
            s.observe_calls+=1
            return torch.ones(1,4)*s.observe_calls
        def actor(h):
            s.actor_calls+=1
            return torch.ones(1,8)*2.0
        s.model=SimpleNamespace(config=SimpleNamespace(proprio_dim=24,action_dim=8,action_limit=3.0),
                                observe=observe,actor=actor)
        return s

    def observation(self,s):
        p=torch.zeros(1,24)
        p[:,:8]=s.teleop_home_joint_pos-s.default_joint_pos
        p[:,16:24]=s.teleop_home_action
        return {"proprio":p,"image":{"video_rgb":torch.zeros(1,32,32,3,dtype=torch.uint8)}}

    def test_home_requires_position_and_velocity_settling(self):
        s=self.solution();p=self.observation(s)["proprio"]
        self.assertTrue(s._home_action(p)[1])
        p[:,1]+=.05
        self.assertFalse(s._home_action(p)[1])
        p[:,1]-=.05;p[:,9]=.4
        self.assertFalse(s._home_action(p)[1])

    def test_four_real_observations_before_first_actor_and_action_rate_cap(self):
        s=self.solution();obs=self.observation(s)
        home=s.predicts(obs,999)
        self.assertEqual(s.control_phase,"home")
        self.assertEqual(s.actor_calls,0)
        for _ in range(4):
            result=s.predicts(obs,999)
            self.assertEqual(s.control_phase,"belief_warmup")
            self.assertTrue(torch.allclose(torch.tensor(result["action"]),s.teleop_home_action))
            self.assertEqual(s.actor_calls,0)
        result=s.predicts(obs,999)
        self.assertEqual(s.control_phase,"actor")
        self.assertEqual(s.observe_calls,5)
        self.assertEqual(s.actor_calls,1)
        delta=torch.tensor(result["action"])-obs["proprio"][:,16:24]
        self.assertTrue(torch.all(delta.abs()<=s.max_action_delta+1e-6))
        s.reset()
        self.assertIsNone(s.belief)
        self.assertEqual(s.belief_warmup_step,0)
        self.assertEqual(s.home_steps,0)
        self.assertFalse(s.home_done)

    def test_failed_home_stops_at_bounded_budget(self):
        s=self.solution();obs=self.observation(s)
        obs["proprio"][:,1]+=.2
        s.home_steps=s.home_timeout-1
        self.assertTrue(s.predicts(obs,0)["giveup"])
        self.assertEqual(s.control_phase,"home_timeout")
        self.assertEqual(s.actor_calls,0)

if __name__=="__main__":
    unittest.main()
