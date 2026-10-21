#!/usr/bin/env bash
set -euo pipefail
cd /root/Desktop/ATEC_TaskE_Standalone
source /opt/conda/etc/profile.d/conda.sh
conda activate isaaclab
export PYTHONUNBUFFERED=1
export ATEC_JEPA_BUNDLE=/root/Desktop/ATEC_TaskE_Standalone/work/jepa_prepare_2026-10-09
task_site=$(/opt/conda/envs/isaaclab/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH="$task_site/torch/lib:$task_site/nvidia/cublas/lib:$task_site/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
task_run=$1
trap 'task_exit=$?; printf "%s\n" "$task_exit" > "$task_run/exit_code.txt"' EXIT
# Bootstrap perception ONLY. No free-flight residual training or deployment.
timeout --signal=TERM --kill-after=20s 900 /opt/conda/envs/isaaclab/bin/python \
  scripts/throwing/train_pose.py --steps 1500 --batch-size 64 --feature-batch-size 16 \
  --train-files \
  work/toss_banana_2026-10-13_IDkgm5cE/episode/pose_data.hdf5 \
  work/toss_banana_approach_2026-10-13_IiYRMvrE/episode/pose_data.hdf5 \
  work/toss_banana_settle_2026-10-13_mLYIgyLu/episode/pose_data.hdf5 \
  work/toss_banana_jointbrake_2026-10-13_I5fdkUEQ/episode/pose_data.hdf5 \
  work/toss_banana_tilt_2026-10-13_6SH6kQoR/episode/pose_data.hdf5 \
  --val-files work/toss_banana_release_2026-10-13_w7UJcOBy/episode/pose_data.hdf5 \
  --cache-dir work/toss_pose_frozen_features \
  --output-dir "$task_run/model"
