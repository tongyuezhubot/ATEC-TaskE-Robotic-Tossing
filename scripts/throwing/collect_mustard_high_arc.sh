#!/usr/bin/env bash
# Finite Oracle TRAINING collection, not a pure-RGB evaluation or deployment.
set -euo pipefail
cd /root/Desktop/ATEC_TaskE_Standalone
task_run=${1:?new empty run directory required}
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_run/queue_exit_code.txt"' EXIT
/opt/conda/envs/isaaclab/bin/python scripts/throwing/run_toss_collection_batch.py \
  --run-dir "$task_run" --object 2 --seed-start 310 --count 12 \
  --planner-config work/throw_motion_mustard_high_arc_2026-10-15.json \
  --joint-model work/toss_joint_fit_2026-10-14_jZpLDlry/model/joint_response.json
/opt/conda/envs/isaaclab/bin/python scripts/throwing/analyze_toss_batch.py \
  --batch-dir "$task_run" --output-dir "$task_run/analysis"
# Analyzer keeps insufficient/failed fits uncalibrated. Nothing is deployed.
