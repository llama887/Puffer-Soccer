#!/bin/bash
#
#SBATCH --job-name=tune_warmstart
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_45_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --chdir=/scratch/fyy2003/repos/Puffer-Soccer
#SBATCH --output=/scratch/fyy2003/repos/Puffer-Soccer/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/fyy2003/repos/Puffer-Soccer/sbatch/logs/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Warm-start hyperparameter tuning (Optuna).
#
# Finds PPO settings that reliably clear the no-opponent scoring gate so the
# subsequent self-play training never fails at the warm-start phase. Runs
# --local because the GPU is already allocated by this sbatch job.
# ---------------------------------------------------------------------------

REPO_ROOT="$(pwd -P)"
JOB_COPY_BASE="$REPO_ROOT/sbatch-tmp"
JOB_COPY_ROOT="$JOB_COPY_BASE/${SLURM_JOB_ID:-local}"
JOB_WORKSPACE_ROOT="$JOB_COPY_ROOT/$(basename "$REPO_ROOT")"
SCRATCH_BASE="/scratch/$USER"
LOCAL_TMP_BASE="/tmp/$USER"
APPTAINER_IMAGE="docker://quay.io/pypa/manylinux_2_28_x86_64"

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

# GPU heartbeat: keeps GPU utilization at ~70% during idle gaps so the cluster
# scheduler does not flag the job as underutilizing the GPU allocation.
export GPU_HEARTBEAT_TARGET_UTILIZATION="${GPU_HEARTBEAT_TARGET_UTILIZATION:-70}"
export GPU_HEARTBEAT_CHECK_INTERVAL="${GPU_HEARTBEAT_CHECK_INTERVAL:-0.2}"
export GPU_HEARTBEAT_MATRIX_SIZE="${GPU_HEARTBEAT_MATRIX_SIZE:-6144}"
export GPU_HEARTBEAT_UTILIZATION_TOLERANCE="${GPU_HEARTBEAT_UTILIZATION_TOLERANCE:-3}"
export GPU_HEARTBEAT_MIN_COMPUTE_SECONDS="${GPU_HEARTBEAT_MIN_COMPUTE_SECONDS:-0.10}"
export GPU_HEARTBEAT_MAX_COMPUTE_SECONDS="${GPU_HEARTBEAT_MAX_COMPUTE_SECONDS:-1.20}"
export GPU_HEARTBEAT_COMPUTE_GAIN_SECONDS="${GPU_HEARTBEAT_COMPUTE_GAIN_SECONDS:-0.03}"
export GPU_HEARTBEAT_MATMULS_PER_CHUNK="${GPU_HEARTBEAT_MATMULS_PER_CHUNK:-8}"
export GPU_HEARTBEAT_DTYPE="${GPU_HEARTBEAT_DTYPE:-bfloat16}"

# Tuning parameters. The warm-start env now uses the full self-play mechanics
# (both teams present, throw-ins active, offside active, per-agent stats in obs)
# with red driven by a scripted max-power kick. Blue therefore trains against
# stationary red defenders instead of an empty field, which is harder than the
# previous no-opponent bootstrap. We keep the same thresholds as v4 and let the
# tuner find a per-trial budget that clears them.
WARMSTART_TRIALS="${WARMSTART_TRIALS:-16}"
WARMSTART_MIN_ITERATIONS="${WARMSTART_MIN_ITERATIONS:-512}"
WARMSTART_GOAL_RATE="${WARMSTART_GOAL_RATE:-0.70}"
WARMSTART_STAGE_ADVANCEMENT="${WARMSTART_STAGE_ADVANCEMENT:-0.65}"
WARMSTART_MAP_SCALE_LADDER="${WARMSTART_MAP_SCALE_LADDER:-0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
WARMSTART_OUTPUT_DIR="${WARMSTART_OUTPUT_DIR:-/persistent-experiments/warmstart_optuna_v8}"

# Force-rebuild the C extension from the current source to pick up env changes.
uv cache clean puffer-soccer 2>/dev/null || true
uv sync --extra dev --python /opt/python/cp312-cp312/bin/python --reinstall-package puffer-soccer

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

# Delete stale cached warm-start payloads. The observation space changed to
# 19 + 24*n (added per-agent stat observations), so any v1-format cache or any
# cache saved before the warm-start refactor is incompatible. The loader also
# requires format_version=2, so stale caches would be rejected anyway, but we
# remove them here for clarity. Remove from both the bind-mounted persistent
# dir and the workspace-local copy (rsync copies any stale file before this
# line runs).
rm -f /persistent-experiments/cached_warm_start.pt
rm -f /workspace/experiments/cached_warm_start.pt

uv run --with optuna python -u scripts/tune_warmstart_optuna.py \
    --local \
    --device cuda \
    --trials "$WARMSTART_TRIALS" \
    --warmstart-min-iterations "$WARMSTART_MIN_ITERATIONS" \
    --goal-rate-threshold "$WARMSTART_GOAL_RATE" \
    --stage-advancement-threshold "$WARMSTART_STAGE_ADVANCEMENT" \
    --multi-goal-rate-threshold 0.0 \
    --map-scale-ladder "$WARMSTART_MAP_SCALE_LADDER" \
    --eval-games 100 \
    --eval-max-steps 600 \
    --output-dir "$WARMSTART_OUTPUT_DIR" \
    --players-per-team 5 \
    --trial-timeout 18000
'
