#!/bin/bash

set -euo pipefail

REPO_ROOT="$(pwd -P)"
JOB_COPY_BASE="$REPO_ROOT/sbatch-tmp"
JOB_COPY_ROOT="$JOB_COPY_BASE/${SLURM_JOB_ID:-local}"
JOB_WORKSPACE_ROOT="$JOB_COPY_ROOT/$(basename "$REPO_ROOT")"
SCRATCH_BASE="/scratch/$USER"
LOCAL_TMP_BASE="/tmp/$USER"
APPTAINER_IMAGE="docker://quay.io/pypa/manylinux_2_28_x86_64"

if [ -z "${TUNE_MATRIX_LABEL:-}" ]; then
    echo "TUNE_MATRIX_LABEL is required" >&2
    exit 1
fi
if [ -z "${TUNE_RL_ALG:-}" ]; then
    echo "TUNE_RL_ALG is required" >&2
    exit 1
fi
if [ -z "${TUNE_KL_MODE:-}" ]; then
    echo "TUNE_KL_MODE is required" >&2
    exit 1
fi

export APPTAINER_CACHEDIR="${TRAIN_AUTOMODE_APPTAINER_CACHEDIR:-$SCRATCH_BASE/.cache/apptainer}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${TRAIN_AUTOMODE_APPTAINER_TMPDIR:-$LOCAL_TMP_BASE/apptainer-tmp-${SLURM_JOB_ID:-local}}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_BASE/.cache/uv}"
export TMPDIR="${TMPDIR:-$SCRATCH_BASE/tmp}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SCRATCH_BASE/.venvs/puffer-soccer-${SLURM_JOB_ID:-local}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PERSISTENT_REPO_ROOT="${PERSISTENT_REPO_ROOT:-$REPO_ROOT}"
mkdir -p \
    "$APPTAINER_CACHEDIR" \
    "$APPTAINER_TMPDIR" \
    "$UV_CACHE_DIR" \
    "$TMPDIR" \
    "$(dirname "$UV_PROJECT_ENVIRONMENT")" \
    "$REPO_ROOT/sbatch/logs" \
    "$JOB_COPY_BASE"

rm -rf "$JOB_COPY_ROOT"
mkdir -p "$JOB_COPY_ROOT"
rsync -a --delete --exclude 'sbatch-tmp/' "$REPO_ROOT/" "$JOB_WORKSPACE_ROOT/"

printf 'REPO_ROOT=%s\n' "$REPO_ROOT"
printf 'JOB_WORKSPACE_ROOT=%s\n' "$JOB_WORKSPACE_ROOT"
printf 'TUNE_MATRIX_LABEL=%s\n' "$TUNE_MATRIX_LABEL"
printf 'TUNE_RL_ALG=%s\n' "$TUNE_RL_ALG"
printf 'TUNE_KL_MODE=%s\n' "$TUNE_KL_MODE"

module purge

apptainer exec --nv \
    --bind "$JOB_WORKSPACE_ROOT:/workspace" \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    bash -lc '
set -euo pipefail

cd /workspace

export APPTAINER_CACHEDIR="'"$APPTAINER_CACHEDIR"'"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="'"$APPTAINER_TMPDIR"'"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export UV_CACHE_DIR="'"$UV_CACHE_DIR"'"
export TMPDIR="'"$TMPDIR"'"
export UV_PROJECT_ENVIRONMENT="'"$UV_PROJECT_ENVIRONMENT"'"
export PYTHONUNBUFFERED="'"$PYTHONUNBUFFERED"'"
export PERSISTENT_REPO_ROOT="'"$PERSISTENT_REPO_ROOT"'"
export MAX_JOBS="${MAX_JOBS:-1}"

export GPU_HEARTBEAT_TARGET_UTILIZATION="${GPU_HEARTBEAT_TARGET_UTILIZATION:-70}"
export GPU_HEARTBEAT_CHECK_INTERVAL="${GPU_HEARTBEAT_CHECK_INTERVAL:-0.2}"
export GPU_HEARTBEAT_MATRIX_SIZE="${GPU_HEARTBEAT_MATRIX_SIZE:-6144}"
export GPU_HEARTBEAT_UTILIZATION_TOLERANCE="${GPU_HEARTBEAT_UTILIZATION_TOLERANCE:-3}"
export GPU_HEARTBEAT_MIN_COMPUTE_SECONDS="${GPU_HEARTBEAT_MIN_COMPUTE_SECONDS:-0.10}"
export GPU_HEARTBEAT_MAX_COMPUTE_SECONDS="${GPU_HEARTBEAT_MAX_COMPUTE_SECONDS:-1.20}"
export GPU_HEARTBEAT_COMPUTE_GAIN_SECONDS="${GPU_HEARTBEAT_COMPUTE_GAIN_SECONDS:-0.03}"
export GPU_HEARTBEAT_MATMULS_PER_CHUNK="${GPU_HEARTBEAT_MATMULS_PER_CHUNK:-8}"
export GPU_HEARTBEAT_DTYPE="${GPU_HEARTBEAT_DTYPE:-bfloat16}"

export TUNE_OUTPUT_ROOT="${TUNE_OUTPUT_ROOT:-$PERSISTENT_REPO_ROOT/experiments/self_play_rl_matrix}"
export TUNE_MAX_RUNS="${TUNE_MAX_RUNS:-8}"
export TUNE_CONFIRM_CANDIDATES="${TUNE_CONFIRM_CANDIDATES:-3}"
export TUNE_CANDIDATE_TOTAL_SEEDS="${TUNE_CANDIDATE_TOTAL_SEEDS:-3}"
export TUNE_TOTAL_TIMESTEPS="${TUNE_TOTAL_TIMESTEPS:-30000000}"
export TUNE_FINAL_EVAL_GAMES="${TUNE_FINAL_EVAL_GAMES:-128}"
export TUNE_VEC_BACKEND="${TUNE_VEC_BACKEND:-auto}"
export TUNE_AUTOTUNE_SECONDS="${TUNE_AUTOTUNE_SECONDS:-1.5}"
export TUNE_DEVICE="${TUNE_DEVICE:-cuda}"
export TUNE_METHOD="${TUNE_METHOD:-Protein}"
export TUNE_EXTRA_ARGS="${TUNE_EXTRA_ARGS:-}"
export TUNE_RUNTIME_CONFIG_PATH="${TUNE_RUNTIME_CONFIG_PATH:-$PERSISTENT_REPO_ROOT/experiments/rl_tuning_runtime_config.json}"
export TUNE_CACHED_WARM_START_PATH="${TUNE_CACHED_WARM_START_PATH:-$PERSISTENT_REPO_ROOT/experiments/cached_warm_start_policy.pt}"

uv sync --extra dev --python /opt/python/cp312-cp312/bin/python

cleanup() {
    if [ -n "${HEARTBEAT_PID:-}" ] && kill -0 "$HEARTBEAT_PID" 2>/dev/null; then
        kill "$HEARTBEAT_PID"
        wait "$HEARTBEAT_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

nice -n 19 uv run python -u sbatch/gpu_heartbeat.py &
HEARTBEAT_PID=$!
echo "Started GPU heartbeat with PID: $HEARTBEAT_PID"

OUTPUT_DIR="$TUNE_OUTPUT_ROOT/$TUNE_MATRIX_LABEL"
mkdir -p "$OUTPUT_DIR"

uv run python -u scripts/tune_best_checkpoint_rl.py \
    --rl-alg "$TUNE_RL_ALG" \
    --kl-regularization-mode "$TUNE_KL_MODE" \
    --device "$TUNE_DEVICE" \
    --vec-backend "$TUNE_VEC_BACKEND" \
    --autotune-seconds "$TUNE_AUTOTUNE_SECONDS" \
    --total-timesteps "$TUNE_TOTAL_TIMESTEPS" \
    --final-eval-games "$TUNE_FINAL_EVAL_GAMES" \
    --max-runs "$TUNE_MAX_RUNS" \
    --confirm-candidates "$TUNE_CONFIRM_CANDIDATES" \
    --candidate-total-seeds "$TUNE_CANDIDATE_TOTAL_SEEDS" \
    --method "$TUNE_METHOD" \
    --output-dir "$OUTPUT_DIR" \
    --runtime-config-path "$TUNE_RUNTIME_CONFIG_PATH" \
    --cached-warm-start-path "$TUNE_CACHED_WARM_START_PATH" \
    --reuse-runtime-config \
    --save-runtime-config \
    --reuse-cached-warm-start \
    --save-cached-warm-start \
    $TUNE_EXTRA_ARGS
'
