#!/bin/bash

set -euo pipefail

REPO_ROOT="$(pwd -P)"
JOB_COPY_BASE="$REPO_ROOT/sbatch-tmp"
JOB_COPY_ROOT="$JOB_COPY_BASE/${SLURM_JOB_ID:-local}"
JOB_WORKSPACE_ROOT="$JOB_COPY_ROOT/$(basename "$REPO_ROOT")"
SCRATCH_BASE="/scratch/$USER"
LOCAL_TMP_BASE="/tmp/$USER"
APPTAINER_IMAGE="docker://quay.io/pypa/manylinux_2_28_x86_64"

if [ -z "${TRAIN_RL_ALG:-}" ]; then
    echo "TRAIN_RL_ALG is required" >&2
    exit 1
fi
if [ -z "${TRAIN_KL_MODE:-}" ]; then
    echo "TRAIN_KL_MODE is required" >&2
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
printf 'TRAIN_RL_ALG=%s\n' "$TRAIN_RL_ALG"
printf 'TRAIN_KL_MODE=%s\n' "$TRAIN_KL_MODE"

module purge

apptainer exec --nv \
    --bind "$JOB_WORKSPACE_ROOT:/workspace" \
    --bind "$REPO_ROOT/experiments:/persistent-experiments" \
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
export TRAIN_HYPERPARAMETERS_PATH="${TRAIN_HYPERPARAMETERS_PATH:-}"
export TRAIN_PPO_ITERATIONS="${TRAIN_PPO_ITERATIONS:-100000}"
export TRAIN_NO_OPPONENT_MIN_ITERATIONS="${TRAIN_NO_OPPONENT_MIN_ITERATIONS:-8}"
export TRAIN_NO_OPPONENT_MAX_ITERATIONS="${TRAIN_NO_OPPONENT_MAX_ITERATIONS:-$TRAIN_PPO_ITERATIONS}"
export TRAIN_NO_OPPONENT_EVAL_INTERVAL="${TRAIN_NO_OPPONENT_EVAL_INTERVAL:-4}"
export TRAIN_NO_OPPONENT_GOAL_RATE_THRESHOLD="${TRAIN_NO_OPPONENT_GOAL_RATE_THRESHOLD:-0.70}"
export TRAIN_NO_OPPONENT_STAGE_ADVANCEMENT_THRESHOLD="${TRAIN_NO_OPPONENT_STAGE_ADVANCEMENT_THRESHOLD:-0.50}"
export TRAIN_NO_OPPONENT_MULTI_GOAL_RATE_THRESHOLD="${TRAIN_NO_OPPONENT_MULTI_GOAL_RATE_THRESHOLD:-0.0}"
export TRAIN_NO_OPPONENT_EVAL_GAMES="${TRAIN_NO_OPPONENT_EVAL_GAMES:-100}"
export TRAIN_NO_OPPONENT_MAP_SCALE_LADDER="${TRAIN_NO_OPPONENT_MAP_SCALE_LADDER:-0.2,0.4,0.6,0.8,1.0}"
export TRAIN_PRETUNE_VECENV="${TRAIN_PRETUNE_VECENV:-1}"
export TRAIN_PRETUNE_VECENV_BACKEND="${TRAIN_PRETUNE_VECENV_BACKEND:-auto}"
export TRAIN_PRETUNE_AUTOTUNE_SECONDS="${TRAIN_PRETUNE_AUTOTUNE_SECONDS:-1.0}"
export TRAIN_PRETUNE_MAX_NUM_ENVS="${TRAIN_PRETUNE_MAX_NUM_ENVS:-}"
export TRAIN_PRETUNE_MAX_NUM_SHARDS="${TRAIN_PRETUNE_MAX_NUM_SHARDS:-}"
export TRAIN_PRETUNE_SELECTION_MODE="${TRAIN_PRETUNE_SELECTION_MODE:-training_like}"
export TRAIN_PRETUNE_DEVICE="${TRAIN_PRETUNE_DEVICE:-auto}"
export TRAIN_PRETUNE_TRAINING_SECONDS="${TRAIN_PRETUNE_TRAINING_SECONDS:-6.0}"
export TRAIN_PRETUNE_SHORTLIST_PER_BACKEND="${TRAIN_PRETUNE_SHORTLIST_PER_BACKEND:-3}"
export TRAIN_FINAL_BEST_EVAL_GAMES="${TRAIN_FINAL_BEST_EVAL_GAMES:-256}"
export TRAIN_BEST_PROMOTION_CONFIDENCE="${TRAIN_BEST_PROMOTION_CONFIDENCE:-0.95}"
export TRAIN_BEST_PROMOTION_MIN_BATCHES="${TRAIN_BEST_PROMOTION_MIN_BATCHES:-4}"
export TRAIN_BEST_PROMOTION_MAX_BATCHES="${TRAIN_BEST_PROMOTION_MAX_BATCHES:-64}"
export TRAIN_CACHED_WARM_START_PATH="${TRAIN_CACHED_WARM_START_PATH:-}"
export TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

# Force-rebuild the C extension from the current source to pick up env changes.
# The uv cache may contain a stale wheel with the old obs_size.
uv cache clean puffer-soccer 2>/dev/null || true
uv sync --python /opt/python/cp312-cp312/bin/python --reinstall-package puffer-soccer

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

if [ "$TRAIN_PRETUNE_VECENV" = "1" ]; then
    PRETUNE_ARGS=(
        uv run python -u scripts/pretune_train_automode_sps.py
        --output-path "${TRAIN_HYPERPARAMETERS_PATH:-experiments/autoload_hyperparameters.json}"
        --players-per-team 5
        --vec-backend "$TRAIN_PRETUNE_VECENV_BACKEND"
        --autotune-seconds "$TRAIN_PRETUNE_AUTOTUNE_SECONDS"
        --selection-mode "$TRAIN_PRETUNE_SELECTION_MODE"
        --device "$TRAIN_PRETUNE_DEVICE"
        --training-benchmark-seconds "$TRAIN_PRETUNE_TRAINING_SECONDS"
        --training-shortlist-per-backend "$TRAIN_PRETUNE_SHORTLIST_PER_BACKEND"
    )
    if [ -n "$TRAIN_PRETUNE_MAX_NUM_ENVS" ]; then
        PRETUNE_ARGS+=(
            --autotune-max-num-envs "$TRAIN_PRETUNE_MAX_NUM_ENVS"
        )
    fi
    if [ -n "$TRAIN_PRETUNE_MAX_NUM_SHARDS" ]; then
        PRETUNE_ARGS+=(
            --autotune-max-num-shards "$TRAIN_PRETUNE_MAX_NUM_SHARDS"
        )
    fi

    echo "Pretuning vecenv layout before training"
    "${PRETUNE_ARGS[@]}"
fi

TRAIN_ARGS=(
    uv run python -u scripts/train_pufferl.py
    --rl-alg "'"$TRAIN_RL_ALG"'"
    --kl-regularization-mode "'"$TRAIN_KL_MODE"'"
    --device cuda
    --wandb
    --wandb-project robot-soccer-discrete
    --wandb-group "'"$TRAIN_RL_ALG"'__kl_'"$TRAIN_KL_MODE"'"
    --ppo-iterations "$TRAIN_PPO_ITERATIONS"
    --no-opponent-phase-min-iterations "$TRAIN_NO_OPPONENT_MIN_ITERATIONS"
    --no-opponent-phase-max-iterations "$TRAIN_NO_OPPONENT_MAX_ITERATIONS"
    --no-opponent-phase-eval-interval "$TRAIN_NO_OPPONENT_EVAL_INTERVAL"
    --no-opponent-phase-goal-rate-threshold "$TRAIN_NO_OPPONENT_GOAL_RATE_THRESHOLD"
    --no-opponent-phase-stage-advancement-threshold "$TRAIN_NO_OPPONENT_STAGE_ADVANCEMENT_THRESHOLD"
    --no-opponent-phase-multi-goal-rate-threshold "$TRAIN_NO_OPPONENT_MULTI_GOAL_RATE_THRESHOLD"
    --no-opponent-eval-games "$TRAIN_NO_OPPONENT_EVAL_GAMES"
    --no-opponent-map-scale-ladder "$TRAIN_NO_OPPONENT_MAP_SCALE_LADDER"
    --final-best-eval-games "$TRAIN_FINAL_BEST_EVAL_GAMES"
    --best-checkpoint-promotion-confidence "$TRAIN_BEST_PROMOTION_CONFIDENCE"
    --best-checkpoint-promotion-min-batches "$TRAIN_BEST_PROMOTION_MIN_BATCHES"
    --best-checkpoint-promotion-max-batches "$TRAIN_BEST_PROMOTION_MAX_BATCHES"
)

if [ -n "$TRAIN_HYPERPARAMETERS_PATH" ]; then
    TRAIN_ARGS+=(--hyperparameters-path "$TRAIN_HYPERPARAMETERS_PATH")
fi

if [ "${TRAIN_SKIP_CACHED_WARM_START:-0}" = "1" ]; then
    TRAIN_ARGS+=(--no-reuse-cached-warm-start)
elif [ -n "$TRAIN_CACHED_WARM_START_PATH" ]; then
    TRAIN_ARGS+=(--cached-warm-start-path "$TRAIN_CACHED_WARM_START_PATH")
else
    TRAIN_ARGS+=(--cached-warm-start-path "/persistent-experiments/cached_warm_start.pt")
fi

if [ -n "$TRAIN_EXTRA_ARGS" ]; then
    TRAIN_ARGS+=($TRAIN_EXTRA_ARGS)
fi

"${TRAIN_ARGS[@]}"
'
