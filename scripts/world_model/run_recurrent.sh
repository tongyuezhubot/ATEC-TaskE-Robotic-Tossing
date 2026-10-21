#!/usr/bin/env bash
set -euo pipefail
cd /root/Desktop/ATEC_TaskE_Standalone
export PYTHONPATH="$PWD/source/atec_rl_lab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
atec_env=/opt/conda/envs/isaaclab
atec_site=$atec_env/lib/python3.11/site-packages
export LD_LIBRARY_PATH="$atec_site/torch/lib:$atec_site/nvidia/cublas/lib:$atec_site/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
atec_python=$atec_env/bin/python
export PATH="$atec_env/bin:$PATH"
export CONDA_PREFIX="$atec_env"
atec_expert=datasets/atec_task_e/combined_20/trajectory.hdf5
atec_prior=atec_robot_model/baseline/world_model/recurrent_expert_1k.pt
atec_labeled=atec_robot_model/baseline/world_model/recurrent_labeled_heads_300.pt
atec_actor_candidate=atec_robot_model/baseline/world_model/recurrent_actor_recovery_400.pt
case "${1:-}" in
  pretrain)
    "$atec_python" scripts/world_model/pretrain_recurrent.py \
      --dataset "$atec_expert" --iterations 1000 --batch_size 4 --sequence_length 12 \
      --init atec_robot_model/baseline/world_model/expert_batch001_5epoch.pt \
      --checkpoint "$atec_prior"
    ;;
  collect-small)
    /root/IsaacLab/isaaclab.sh -p scripts/act/collect_demos_task_e.py \
      --headless --aligned --save_images --image_size 96 \
      --pick_objects 3 2 1 --num_demos 3 --max_attempts 6 --seed 42 \
      --output_dir datasets/atec_task_e/aligned_003
    ;;
  collect-recovery)
    /root/IsaacLab/isaaclab.sh -p scripts/act/collect_demos_task_e.py \
      --headless --aligned --recovery_mix --save_images --image_size 96 \
      --pick_objects 3 2 1 --num_demos 6 --max_attempts 6 --max_record_steps 6000 --seed 104 \
      --output_dir datasets/atec_task_e/aligned_recovery_006
    ;;
  audit-recovery)
    "$atec_python" scripts/world_model/audit_recovery_data.py \
      --dataset datasets/atec_task_e/aligned_recovery_006/trajectory.hdf5 \
      --output datasets/atec_task_e/aligned_recovery_006/audit.json
    ;;
  refine-actor)
    "$atec_python" scripts/world_model/pretrain_recurrent.py \
      --dataset "$atec_expert" --aligned_dataset datasets/atec_task_e/aligned_recovery_006/trajectory.hdf5 \
      --init "$atec_labeled" --checkpoint "$atec_actor_candidate" --actor_only --actor_action_limit 3.5 \
      --iterations 400 --batch_size 4 --sequence_length 12 \
      --val_every 100 --patience 3 --min_delta 0.00001 --min_steps 200
    ;;
  train-labeled)
    "$atec_python" scripts/world_model/pretrain_recurrent.py \
      --dataset "$atec_expert" --aligned_dataset datasets/atec_task_e/aligned_003_corrected/trajectory.hdf5 \
      --init "$atec_prior" --checkpoint "$atec_labeled" --freeze_prior \
      --iterations 300 --batch_size 4 --sequence_length 12 \
      --val_every 50 --patience 3 --min_delta 0.001 --min_steps 100
    ;;
  control-check)
    atec_trace=$(mktemp -d runs/recurrent_control_check_XXXXXXXX)
    echo "Control trace output: $atec_trace"
    ATEC_TASK_E_BACKEND=world_model ATEC_TASK_E_WORLD_MODEL="${ATEC_TASK_E_WORLD_MODEL:-$atec_prior}" \
      ATEC_WM_CONTROL="${ATEC_WM_CONTROL:-actor}" /root/IsaacLab/isaaclab.sh -p scripts/play_atec_task.py \
      --headless --enable_cameras --max_steps 300 --seed 43 \
      --control_trace "$atec_trace/control_trace.json" --result_file "$atec_trace/result.json" \
      2>&1 | tee "$atec_trace/play.log"
    ;;
  prior-eval)
    ATEC_TASK_E_BACKEND=world_model ATEC_TASK_E_WORLD_MODEL="$atec_prior" \
      ATEC_WM_CONTROL=actor /root/IsaacLab/isaaclab.sh -p scripts/play_atec_task.py \
      --headless --enable_cameras --max_steps 1500 --seed 43
    ;;
  online)
    /root/IsaacLab/isaaclab.sh -p scripts/world_model/train_recurrent.py \
      --headless --resume "$atec_labeled" --expert_dataset "$atec_expert" \
      --output_dir runs/recurrent_online_1k --total_steps 1000 --seed 43
    ;;
  eval)
    ATEC_TASK_E_BACKEND=world_model ATEC_TASK_E_WORLD_MODEL=runs/recurrent_online_1k/policy.pt \
      ATEC_WM_CONTROL=cem /root/IsaacLab/isaaclab.sh -p scripts/play_atec_task.py \
      --headless --enable_cameras --max_steps 4000 --seed 43
    ;;
  *)
    echo "usage: bash $0 {pretrain|collect-small|collect-recovery|audit-recovery|refine-actor|train-labeled|control-check|prior-eval|online|eval}" >&2
    exit 2
    ;;
esac
