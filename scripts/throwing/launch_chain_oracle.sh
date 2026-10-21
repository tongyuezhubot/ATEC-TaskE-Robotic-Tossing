#!/usr/bin/env bash
# Development chain ONLY: explicitly privileged pose labels, never RGB proof.
set -euo pipefail
cd /root/Desktop/ATEC_TaskE_Standalone
task_run=${1:?provide a NEW output directory}
task_seed=${2:-220}
if [[ -e "$task_run" ]]; then
  echo "Choose a new run directory; existing data is preserved." >&2
  exit 2
fi
mkdir -p "$task_run"
unset ATEC_THROW_WORLD_MODEL ATEC_THROW_PERCEPTION ATEC_THROW_RESIDUAL
export ATEC_THROW_PROFILES=/root/Desktop/ATEC_TaskE_Standalone/work/throw_profiles_chain_2026-10-15.json
export ATEC_THROW_JOINT_MODEL=/root/Desktop/ATEC_TaskE_Standalone/work/toss_joint_fit_2026-10-14_jZpLDlry/model/joint_response.json
export OPENBLAS_NUM_THREADS=1
bash scripts/throwing/launch_collection.sh "$task_run" "3 2 1" "$task_seed" 2400 300 joint_swing "" 2 ""
