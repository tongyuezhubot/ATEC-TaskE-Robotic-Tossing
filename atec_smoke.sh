#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/root/IsaacLab}"
PROJECT_ROOT="${ATEC_PROJECT_ROOT:-$SCRIPT_DIR}"
CACHE_ROOT="/tmp/atec-smoke-cache"
PYTHON_BIN="/opt/conda/envs/isaaclab/bin/python"
SITE_PACKAGES="/opt/conda/envs/isaaclab/lib/python3.11/site-packages"
TORCH_LIB="$SITE_PACKAGES/torch/lib"

export TERM=xterm
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export WARP_CACHE_PATH="$CACHE_ROOT/warp"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export PYTHONPATH="$PROJECT_ROOT/source/atec_rl_lab:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$WARP_CACHE_PATH" "$XDG_CACHE_HOME"

# Isaac Kit loads some PyTorch CUDA components lazily.  Put the matching
# PyTorch and pip-installed CUDA 12.8 libraries ahead of the system toolkit so
# dlopen resolves the same libraries before and after Kit starts.
configure_cuda_library_path() {
    local nvidia_lib_dirs=""
    local lib_dir
    while IFS= read -r lib_dir; do
        nvidia_lib_dirs="${nvidia_lib_dirs:+$nvidia_lib_dirs:}$lib_dir"
    done < <(find "$SITE_PACKAGES/nvidia" -mindepth 2 -maxdepth 2 \
        -type d -name lib -print | sort)

    export LD_LIBRARY_PATH="$TORCH_LIB${nvidia_lib_dirs:+:$nvidia_lib_dirs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

preflight() {
    echo "[preflight] architecture: $(uname -m)"
    echo "[preflight] Python: $(python3 --version 2>&1)"
    df -h "$PROJECT_ROOT"

    local free_kib
    free_kib=$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if (( free_kib < 8 * 1024 * 1024 )); then
        echo "[error] Less than 8 GiB free; refusing simulation smoke tests." >&2
        return 1
    fi

    if ! nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
        echo "[error] NVIDIA GPU/driver is unavailable in this environment." >&2
        return 1
    fi

    test -x "$ISAACLAB_ROOT/isaaclab.sh"
    test -f "$PROJECT_ROOT/scripts/play_atec_task.py"
    echo "[preflight] PASS"
}

run_cpu_act() {
    cd "$PROJECT_ROOT"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 - <<'PY'
import torch
from demo.solution_act import Agent, Args

checkpoint = torch.load(
    "atec_robot_model/baseline/act/policy.pt",
    map_location="cpu",
    weights_only=True,
)
stats = checkpoint["norm_stats"]
state_dim = int(stats["state_mean"].shape[-1])
action_dim = int(stats["action_mean"].shape[-1])
weight_key = "ema_agent" if "ema_agent" in checkpoint else "agent"

args = Args(cuda=False)
args.num_queries = 30
args.include_rgb = any("backbone" in name for name in checkpoint[weight_key])
agent = Agent(state_dim, action_dim, args).cpu()
agent.load_state_dict(checkpoint[weight_key], strict=True)
agent.eval()

observation = {"state": torch.zeros(1, state_dim)}
if args.include_rgb:
    observation["rgb"] = torch.zeros(1, 1, 3, 224, 224, dtype=torch.uint8)

with torch.inference_mode():
    action_sequence = agent.get_action(observation)

assert action_sequence.shape == (1, 30, action_dim)
assert torch.isfinite(action_sequence).all()
print(
    {
        "checkpoint_key": weight_key,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "include_rgb": args.include_rgb,
        "output_shape": tuple(action_sequence.shape),
        "finite": True,
    }
)
PY
}

run_torch_cuda() {
    preflight
    configure_cuda_library_path
    "$PYTHON_BIN" - <<'PY'
import os
import torch

print({
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
})
assert torch.cuda.is_available()
x = torch.eye(4, device="cuda")
y = torch.linalg.solve(x, x)
torch.cuda.synchronize()
print({
    "device": torch.cuda.get_device_name(0),
    "linalg_device": str(y.device),
    "linalg_finite": bool(torch.isfinite(y).all()),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
})
PY
}

run_cartpole() {
    preflight
    configure_cuda_library_path
    cd "$ISAACLAB_ROOT"
    ./isaaclab.sh -p scripts/environments/zero_agent.py \
        --task Isaac-Cartpole-v0 --num_envs 16 --headless
}

run_task_scene() {
    preflight
    configure_cuda_library_path
    cd "$PROJECT_ROOT"
    "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/view_task_e.py \
        --num_envs 1 --headless --enable_cameras
}

run_task_scene_gui() {
    preflight
    configure_cuda_library_path
    cd "$PROJECT_ROOT"
    "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/view_task_e.py \
        --num_envs 1 --enable_cameras
}

run_act() {
    preflight
    configure_cuda_library_path
    cd "$PROJECT_ROOT"
    unset ATEC_TASK_E_POLICY
    "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/play_atec_task.py \
        --headless --enable_cameras --num_envs 1 --debug
}

run_act_gui() {
    preflight
    configure_cuda_library_path
    cd "$PROJECT_ROOT"
    unset ATEC_TASK_E_POLICY
    "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/play_atec_task.py \
        --enable_cameras --num_envs 1
}

run_act_video() {
    preflight
    configure_cuda_library_path
    cd "$PROJECT_ROOT"
    unset ATEC_TASK_E_POLICY
    "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/play_atec_task.py \
        --headless --enable_cameras --video --video_length 500 --num_envs 1
}

run_world_gui() {
    preflight
    configure_cuda_library_path
    test -f "$PROJECT_ROOT/atec_robot_model/baseline/world_model/policy.pt" || {
        echo "[error] Missing world-model policy.pt; run world-smoke first." >&2
        return 1
    }
    cd "$PROJECT_ROOT"
    export ATEC_TASK_E_BACKEND=world_model
    "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/play_atec_task.py \
        --enable_cameras --num_envs 1
}

run_demo_smoke() {
    local object_id="${1:-3}"
    local max_attempts="${2:-3}"
    if [[ ! "$object_id" =~ ^[123]$ ]]; then
        echo "[error] demo-smoke object id must be 1, 2, or 3." >&2
        return 2
    fi
    preflight
    configure_cuda_library_path
    cd "$PROJECT_ROOT"
    "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/act/collect_demos_task_e.py \
        --pick_objects "$object_id" --num_demos 1 \
        --output_dir "datasets/atec_task_e/smoke_object${object_id}" \
        --headless --enable_cameras --save_images --image_size 224 \
        --only_success --max_attempts "$max_attempts"
}

run_demo_diagnose() {
    local object_id="${1:?object id is required}"
    local log_dir="$PROJECT_ROOT/logs/diagnostics"
    local log_path="$log_dir/object${object_id}_latest.log"
    mkdir -p "$log_dir"
    echo "[INFO] Full diagnostic log: $log_path"
    run_demo_smoke "$object_id" 1 2>&1 | tee "$log_path"
}

run_pretrain_smoke() {
    preflight
    configure_cuda_library_path
    local dataset="$PROJECT_ROOT/datasets/atec_task_e/smoke_object3/trajectory.hdf5"
    test -f "$dataset" || {
        echo "[error] Missing demo-smoke dataset; run demo-smoke first." >&2
        return 1
    }
    cd "$PROJECT_ROOT"
    "$PYTHON_BIN" scripts/world_model/pretrain_task_e.py \
        --dataset "$dataset" --total_iters 50 --batch_size 16 \
        --log_every 10 --save_every 50 \
        --checkpoint atec_robot_model/baseline/world_model/expert_smoke.pt
}

run_world_tests() {
    test -f "$PROJECT_ROOT/tests/test_world_model.py" || {
        echo "[error] Missing tests/test_world_model.py; upload the updated project first." >&2
        return 1
    }
    cd "$PROJECT_ROOT"
    KMP_USE_SHM=0 OMP_NUM_THREADS=1 \
        "$ISAACLAB_ROOT/isaaclab.sh" -p -m unittest -v tests.test_world_model
}

run_world_smoke() {
    preflight
    configure_cuda_library_path
    test -f "$PROJECT_ROOT/scripts/world_model/train_task_e.py" || {
        echo "[error] Missing scripts/world_model/train_task_e.py; upload the updated project first." >&2
        return 1
    }
    cd "$PROJECT_ROOT"
    "$ISAACLAB_ROOT/isaaclab.sh" -p scripts/world_model/train_task_e.py \
        --headless \
        --num_envs 1 \
        --total_steps 1000 \
        --random_steps 1000 \
        --buffer_size 1000 \
        --batch_size 32 \
        --save_every 500
}

case "${1:-preflight}" in
    preflight) preflight ;;
    cpu-act) run_cpu_act ;;
    torch-cuda) run_torch_cuda ;;
    cartpole) run_cartpole ;;
    task) run_task_scene ;;
    task-gui) run_task_scene_gui ;;
    act) run_act ;;
    act-gui) run_act_gui ;;
    act-video) run_act_video ;;
    world-gui) run_world_gui ;;
    demo-smoke|demo-smoke-3) run_demo_smoke 3 ;;
    demo-smoke-1) run_demo_smoke 1 ;;
    demo-smoke-2) run_demo_smoke 2 ;;
    demo-diagnose-1) run_demo_diagnose 1 ;;
    demo-diagnose-2) run_demo_diagnose 2 ;;
    pretrain-smoke) run_pretrain_smoke ;;
    world-tests) run_world_tests ;;
    world-smoke) run_world_smoke ;;
    *)
        echo "Usage: $0 {preflight|cpu-act|torch-cuda|cartpole|task|task-gui|act|act-gui|act-video|world-tests|world-smoke|world-gui|demo-smoke-{1,2,3}|demo-diagnose-{1,2}|pretrain-smoke}" >&2
        exit 2
        ;;
esac
