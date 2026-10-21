"""Read-only action-prior diagnosis on fixed held-out recorded windows."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from demo.world_model import load_world_model
from demo.world_model_sequence import ExpertSequenceDataset
from demo.world_model_prior_planner import PriorCEMPlanner

def main():
    torch.set_num_threads(1)
    torch.manual_seed(43)
    path="datasets/atec_task_e/aligned_003_corrected/trajectory.hdf5"
    ds=ExpertSequenceDataset(path,keys=["traj_2"],sequence_length=32,burn_in=4)
    starts=[0,128,512,1024,1536,2048,2560,3072,3395]
    lookup={start:i for i,(key,start,aligned) in enumerate(ds.windows)}
    samples=[ds[lookup[start]] for start in starts]
    ds.close()
    p=torch.stack([s["proprio"] for s in samples]).cuda()
    image=torch.stack([s["image"] for s in samples]).cuda()
    target=torch.stack([s["action"] for s in samples]).cuda()
    previous=p[:,:,16:24]
    valid=torch.ones(target.shape[:2],dtype=torch.bool,device=p.device)
    valid[:,:4]=False
    moving=valid & ((target-previous).norm(dim=-1)>0.01)
    def metrics(actions):
        error=(actions-target).square().mean(-1)
        return {"action_rmse":float(error[valid].mean().sqrt()),
                "moving_rmse":float(error[moving].mean().sqrt()) if moving.any() else None,
                "mean_delta_from_previous":float((actions-previous).norm(dim=-1)[valid].mean())}
    result={"heldout_episode":"traj_2","window_starts":starts,
            "moving_frames":int(moving.sum()),"total_frames":int(valid.sum()),
            "repeat_previous_action":metrics(previous),"closed_loop_result":"0/4000 steps, no basket_success"}
    for name,checkpoint in (
        ("expert_prior","atec_robot_model/baseline/world_model/recurrent_expert_1k.pt"),
        ("labeled_prior","atec_robot_model/baseline/world_model/recurrent_labeled_300.pt")):
        model,_=load_world_model(checkpoint,"cuda")
        belief=None
        actions=[]
        with torch.inference_mode():
            for t in range(32):
                belief=model.observe(p[:,t],image[:,t],belief)
                actions.append(model.actor(belief))
            result[name]=metrics(torch.stack(actions,dim=1))
            if name=="labeled_prior":
                planner=PriorCEMPlanner(model)
                comparisons=[]
                for i in (0,3,6,8):
                    h=belief[i:i+1]
                    planner.previous_action=previous[i:i+1,-1]
                    actor=model.actor(h)
                    planned=planner.plan(h)
                    expert=target[i:i+1,-1]
                    comparisons.append({"window_start":starts[i],
                        "actor_expert_l2":float((actor-expert).norm()),
                        "cem_expert_l2":float((planned-expert).norm()),
                        "cem_actor_l2":float((planned-actor).norm()),
                        "reward_actor":float(model.predict_reward(h,actor)),
                        "reward_cem":float(model.predict_reward(h,planned)),
                        "reward_expert":float(model.predict_reward(h,expert))})
                result["cem_comparisons"]=comparisons
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    main()
