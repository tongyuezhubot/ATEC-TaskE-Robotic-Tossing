#!/usr/bin/env bash
# One fixed warm-start + mustard-only usage calibration; never auto-deploy.
# Banana keeps its previously validated checkpoint, not this specialist.
set -euo pipefail
cd /root/Desktop/ATEC_TaskE_Standalone
task_run=${1:?new run directory required}
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_run/queue_exit_code.txt"' EXIT
/opt/conda/envs/isaaclab/bin/python - "$task_run" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
sources=json.loads(Path('work/toss_pose_padpoint_2026-10-14_o1eJfhcA/sources.json').read_text())
for name in ['toss_mustard_timing_eval_2026-10-15_pzZAEDgI','toss_mustard_timing_world_2026-10-15_mhlU2wdI']:
    root=Path('work')/name
    manifest=json.loads((root/'batch_manifest.json').read_text())
    for job in manifest['jobs']:
        group={'train':'train','validation':'validation'}[job['split']]
        sources[group].append(str((root/f"object2_seed{job['seed']}"/'episode/pose_data.hdf5').resolve()))
all_paths=sources['train']+sources['validation']
assert len(all_paths)==len(set(all_paths))
(out/'sources.json').write_text(json.dumps(sources,indent=2))
print({'training_episodes':len(sources['train']),'validation_episodes':len(sources['validation'])},flush=True)
PY
task_train_exit=0
/opt/conda/envs/isaaclab/bin/python scripts/throwing/train_pose_spatial.py \
  --manifest "$task_run/sources.json" --output-dir "$task_run/model" \
  --cache-dir work/toss_pose_spatial_features_v2 --steps 2000 --stride 4 --batch-size 32 \
  --kinematic-prior --held-focus --held-object 2 --dense-held-object 2 --position-quality \
  --warm-start work/toss_pose_padpoint_2026-10-14_o1eJfhcA/usage_calibration/rgb_pose_tracking.pt || task_train_exit=$?
# v6 deliberately exits 2 with an uncalibrated candidate. Continue ONLY to
# the strict usage gate, never treat that status as deployment approval.
if [[ "$task_train_exit" != 0 && "$task_train_exit" != 2 ]]; then
  exit "$task_train_exit"
fi
test -s "$task_run/model/spatial_pose_candidate.pt"
/opt/conda/envs/isaaclab/bin/python scripts/throwing/calibrate_pose_usage.py \
  --checkpoint "$task_run/model/spatial_pose_candidate.pt" \
  --manifest "$task_run/sources.json" --cache-dir work/toss_pose_spatial_features_v2 \
  --require-held-objects 2 --output-dir "$task_run/mustard_usage_calibration"
