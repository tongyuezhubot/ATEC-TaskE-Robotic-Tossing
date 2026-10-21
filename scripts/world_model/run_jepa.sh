#!/usr/bin/env bash
# Each invocation performs one finite job; no implicit training/evaluation chain.
set -euo pipefail
atec_project=$(cd "$(dirname "$0")/../.." && pwd)
atec_bundle=${ATEC_JEPA_BUNDLE:-"$atec_project/work/jepa_prepare_2026-10-09"}
atec_python=${ATEC_PYTHON:-/opt/conda/envs/isaaclab/bin/python}
atec_site=$("$atec_python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH="$atec_site/torch/lib:$atec_site/nvidia/cublas/lib:$atec_site/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
atec_first="$atec_project/datasets/atec_task_e/aligned_003_corrected/trajectory.hdf5"
atec_recovery="$atec_project/datasets/atec_task_e/aligned_recovery_006/trajectory.hdf5"
case "${1:-}" in
  dynamics)
    : "${ATEC_JEPA_INIT:?Set the previous adapted JEPA checkpoint}"
    atec_run=$(mktemp -d "$atec_project/work/jepa_optimized_dynamics_XXXXXX")
    echo "Output: $atec_run"
    "$atec_python" "$atec_bundle/fit_pilot.py" --dataset "$atec_first" "$atec_recovery" \
      --init "$ATEC_JEPA_INIT" --output-dir "$atec_run/fit" --device cuda \
      --steps 500 --batch-size 2 --last-blocks 1 --val-every 100 --wall-seconds 900
    ;;
  progress)
    : "${ATEC_JEPA_CHECKPOINT:?Set the NEW dynamics checkpoint with learned proprio}"
    atec_run=$(mktemp -d "$atec_project/work/jepa_optimized_task_XXXXXX")
    echo "Output: $atec_run"
    "$atec_python" "$atec_bundle/fit_progress.py" --dataset "$atec_first" "$atec_recovery" \
      --project "$atec_project" --world-model "$ATEC_JEPA_CHECKPOINT" \
      --output-dir "$atec_run/fit" --steps 500 --batch-size 8 --device cuda --wall-seconds 900
    ;;
  control)
    : "${ATEC_JEPA_PROGRESS:?Set the NEW v2 task-state checkpoint}"
    : "${ATEC_JEPA_CHECKPOINT:?Set the corresponding NEW dynamics checkpoint}"
    atec_run=$(mktemp -d "$atec_project/runs/jepa_optimized_control_XXXXXX")
    echo "Output: $atec_run"
    cd "$atec_project"
    export ATEC_JEPA_BUNDLE="$atec_bundle"
    export ATEC_JEPA_USE_SCORING=${ATEC_JEPA_USE_SCORING:-1}
    ATEC_TASK_E_BACKEND=jepa /root/IsaacLab/isaaclab.sh -p scripts/play_atec_task.py \
      --headless --enable_cameras --seed 43 --max_steps 6000 \
      --control_trace "$atec_run/control_trace.json" --result_file "$atec_run/result.json"
    ;;
  *) echo 'usage: bash scripts/world_model/run_jepa.sh dynamics|progress|control' >&2; exit 2 ;;
esac
