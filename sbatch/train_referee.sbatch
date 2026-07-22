#!/bin/bash
#
#SBATCH --job-name=puffer_soccer_referee
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wz2445@nyu.edu
#SBATCH --chdir=/scratch/wz2445/Puffer-Soccer
#SBATCH --output=/scratch/wz2445/Puffer-Soccer/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/wz2445/Puffer-Soccer/sbatch/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(pwd -P)"
JOB_COPY_BASE="$REPO_ROOT/sbatch-tmp"
JOB_COPY_ROOT="$JOB_COPY_BASE/${SLURM_JOB_ID:-local}"
JOB_WORKSPACE_ROOT="$JOB_COPY_ROOT/$(basename "$REPO_ROOT")"
SCRATCH_BASE="/scratch/$USER"
LOCAL_TMP_BASE="/tmp/$USER"
APPTAINER_IMAGE="docker://quay.io/pypa/manylinux_2_28_x86_64"

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH_BASE/apptainer-cache/cache}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$LOCAL_TMP_BASE/apptainer-tmp-${SLURM_JOB_ID:-local}}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_BASE/.cache/uv}"
export TMPDIR="${TMPDIR:-$SCRATCH_BASE/tmp}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SCRATCH_BASE/.venvs/puffer-soccer-shared}"
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
printf 'APPTAINER_CACHEDIR=%s\n' "$APPTAINER_CACHEDIR"
printf 'APPTAINER_TMPDIR=%s\n' "$APPTAINER_TMPDIR"

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

uv sync --python /opt/python/cp312-cp312/bin/python

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

uv run python -u scripts/train_pufferl.py \
    --device cuda \
    --wandb \
    --wandb-project robot-soccer \
    --wandb-group referee \
    --enable-referee \
    --export-videos \
    --video-max-steps 1800 \
    --ppo-iterations 100000
'
