#!/usr/bin/env bash
set -euo pipefail
cd /root/Desktop/ATEC_TaskE_Standalone
source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab
export OMNI_KIT_ALLOW_ROOT=1
export ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1
export PYTHONPATH="/root/Desktop/ATEC_TaskE_Standalone/source/atec_rl_lab:/root/Desktop/ATEC_TaskE_Standalone:${PYTHONPATH:-}"
task_site=$(/opt/conda/envs/isaaclab/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH="$task_site/torch/lib:$task_site/nvidia/cublas/lib:$task_site/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
task_run=$1
task_object=$2
read -r -a task_objects <<< "$task_object"
task_seed=$3
task_steps=${4:-2000}
task_wall_seconds=${5:-900}
task_primitive=${6:-arm_swing}
task_data_stride=${8:-5}
task_perception=${9:-}
task_pose_args=(--oracle-training)
if [[ -n "$task_perception" ]]; then
  export ATEC_THROW_PERCEPTION="$task_perception"
  task_pose_args=()
fi
task_motion_args=()
if [[ -n "${7:-}" ]]; then
  task_motion_args=(--planner-config "$7")
fi
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_run/exit_code.txt"' EXIT
timeout --signal=TERM --kill-after=20s "$task_wall_seconds" /root/IsaacLab/isaaclab.sh -p \
  scripts/throwing/run_throw_task.py --headless "${task_pose_args[@]}" --save-dataset --data-stride "$task_data_stride" \
  --objects "${task_objects[@]}" --seed "$task_seed" --max-steps "$task_steps" --throw-primitive "$task_primitive" "${task_motion_args[@]}" \
  --output-dir "$task_run/episode"
