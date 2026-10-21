"""Compare fixed held-out legal-input windows; no simulator and no training."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from demo.world_model import load_world_model
from demo.world_model_sequence import ExpertSequenceDataset
from scripts.world_model.pretrain_recurrent import event_windows


def evaluate(checkpoint,output):
    candidate,payload=load_world_model(checkpoint,"cuda")
    baseline,_=load_world_model(payload["metadata"]["init"],"cuda")
    split=payload["metadata"]["episode_splits"]
    items=[];info=[]
    for path,keys in split.items():
        ds=ExpertSequenceDataset(path,sequence_length=12,keys=keys["validation"])
        indices=set(torch.linspace(0,len(ds)-1,min(8,len(ds))).long().tolist())
        indices.update(event_windows(ds,actor_only=True))
        for index in sorted(indices):
            sample=ds[index];items.append(sample)
            info.append(dict(dataset=path,episode=ds.windows[index][0],start=ds.windows[index][1]))
        ds.close()
    batch={k:torch.stack([s[k] for s in items]).cuda() for k in items[0]}
    valid=batch["expert"].bool()
    moving=valid & ((batch["action"]-batch["proprio"][:,:,16:24]).norm(dim=-1)>.02)
    weights=1.+4.*moving.float()
    def metrics(actions):
        error=(actions-batch["action"]).square().mean(-1)
        return dict(rmse=float(error[valid].mean().sqrt()),
                    moving_rmse=float(error[moving].mean().sqrt()),
                    weighted_rmse=float((error[valid]*weights[valid]).sum().div(weights[valid].sum()).sqrt()))
    result=dict(windows=info,valid_frames=int(valid.sum()),moving_frames=int(moving.sum()),
                checkpoint=checkpoint,checkpoint_step=payload["step"],
                repeat_previous=metrics(batch["proprio"][:,:,16:24]))
    with torch.inference_mode():
        for name,model in (("old_prior",baseline),("candidate",candidate)):
            belief=None;actions=[]
            for t in range(batch["action"].shape[1]):
                belief=model.observe(batch["proprio"][:,t],batch["image"][:,t],belief)
                actions.append(model.actor(belief))
            result[name]=metrics(torch.stack(actions,1))
    before=baseline.state_dict();after=candidate.state_dict()
    frozen=[k for k in before if not k.startswith(("actor.","target_"))]
    changed=[k for k in frozen if not torch.equal(before[k],after[k])]
    assert not changed,changed
    result["unchanged_nonactor_tensors"]=len(frozen)
    result["candidate_action_limit"]=candidate.config.action_limit
    result["closed_loop_verified"]=False
    with open(output,"x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="windows"},indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();torch.set_num_threads(1)
    evaluate(args.checkpoint,args.output)
