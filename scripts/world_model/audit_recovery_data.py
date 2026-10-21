"""One-pass alignment and recovery-mask audit; no simulator or model evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
from scripts.world_model.repair_aligned_labels import task_geometry, basket_success_from_positions


def audit(path, expected=6):
    params,_=task_geometry("source/atec_rl_lab/atec_rl_lab/tasks/task_e/env_cfg.py")
    result={"dataset":str(Path(path).resolve()),"episodes":[],"total_steps":0,
            "successful_episodes":0,"actor_valid_steps":0,"injected_steps_excluded":0,
            "target_failures":0,"target_recoveries":0,"fault_applied_episodes":0}
    with h5py.File(path,"r") as f:
        if len(f)!=expected:
            raise ValueError(f"Expected {expected} saved episodes, found {len(f)}")
        for key in sorted(f):
            g=f[key]; report=json.loads(g.attrs["recovery_report"]);labels=g["training_labels"]
            n=len(g["actions"])
            p=g["policy_input/proprio"][...];next_p=g["next_policy_input/proprio"][...];a=g["actions"][...]
            assert p.shape==next_p.shape==(n,24)
            assert a.shape==(n,8)
            assert np.isfinite(p).all() and np.isfinite(next_p).all() and np.isfinite(a).all()
            assert np.allclose(next_p[:,16:24],a,atol=1e-6,rtol=0),"previous-action mismatch"
            assert np.allclose(next_p[:-1],p[1:],atol=1e-6,rtol=0),"transition adjacency mismatch"
            assert np.all(g["next_obs_valid"][...]),"missing terminal next observation"
            done=g["terminated"][...] | g["truncated"][...]
            assert done[-1] and not done[:-1].any(),"episode boundary mismatch"
            success=labels["basket_success"][...]
            positions=labels["object_pos"][...]
            expected_success=basket_success_from_positions(positions,params)
            assert np.array_equal(success,expected_success),"success geometry mismatch"
            assert bool(g.attrs["success"])==bool(success[-1])
            assert set(g["policy_input"])=={"proprio","rgb"}
            assert set(g["next_policy_input"])=={"proprio","rgb"}
            assert labels.attrs["deployment_forbidden"]
            assert labels.attrs["basket_success_source"]=="current_term"
            for name in ("policy_input/rgb","next_policy_input/rgb"):
                assert g[name].shape==(n,96,96,3) and g[name].dtype==np.uint8
            # Fixed sparse image adjacency checks, not reading all RGB twice.
            for t in np.linspace(0,n-2,min(12,n-1)).astype(int):
                assert np.array_equal(g["next_policy_input/rgb"][t],g["policy_input/rgb"][t+1])
            active=labels["active_object"][...];attempt=labels["recovery_attempt"][...]
            valid=labels["actor_valid"][...]
            deliberate=(active==report["target_object"]) & (attempt==0)
            assert not valid[deliberate].any(),"fault attempt leaked into BC"
            assert not valid[labels["fault_applied"][...]].any(),"injected action leaked into BC"
            for obj,number in zip(active[valid],attempt[valid]):
                assert report["successful_attempts"].get(str(int(obj)))==int(number)
            failures=[e for e in report["events"] if e["event"]=="failure_observed"]
            target_failures=[e for e in failures if e["object"]==report["target_object"]]
            recovered=report["successful_attempts"].get(str(report["target_object"]))==1
            item={"name":key,"steps":n,"success":bool(success[-1]),"kind":report["requested_kind"],
                  "target_object":report["target_object"],"fault_applied":report["fault_applied"],
                  "target_failures":len(target_failures),"target_recovered":recovered,
                  "retries":report["retries"],"actor_valid_steps":int(valid.sum()),
                  "excluded_target_attempt_steps":int(deliberate.sum())}
            result["episodes"].append(item);result["total_steps"]+=n
            result["successful_episodes"]+=int(success[-1]);result["actor_valid_steps"]+=int(valid.sum())
            result["injected_steps_excluded"]+=int(deliberate.sum())
            result["target_failures"]+=len(target_failures);result["target_recoveries"]+=int(recovered)
            result["fault_applied_episodes"]+=int(report["fault_applied"])
    result["alignment_and_mask_verified"]=True
    result["learned_policy_success_verified"]=False
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True);p.add_argument("--expected",type=int,default=6)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError("Preserve the previous audit; choose a new output")
    result=audit(a.dataset,a.expected)
    with open(a.output,"x") as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
