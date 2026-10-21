#!/usr/bin/env bash
# Complete only the three predeclared jobs after disk-protected stop at 494.
set -euo pipefail
cd /root/Desktop/ATEC_TaskE_Standalone
task_train=$1
task_validation=$2
task_python=/opt/conda/envs/isaaclab/bin/python
task_config=work/throw_motion_sugar_timing_2026-10-15.json
task_joint=work/toss_joint_fit_2026-10-14_jZpLDlry/model/joint_response.json
export OPENBLAS_NUM_THREADS=1
"$task_python" scripts/throwing/run_toss_collection_batch.py --run-dir "$task_train" \
  --planner-config "$task_config" --joint-model "$task_joint" --object 1 --seed-start 495 --count 1
"$task_python" scripts/throwing/run_toss_collection_batch.py --run-dir "$task_validation" \
  --planner-config "$task_config" --joint-model "$task_joint" --object 1 --seed-start 496 --count 2 --validation-only
