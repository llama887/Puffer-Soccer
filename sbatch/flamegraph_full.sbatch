#!/bin/bash
#
#SBATCH --job-name=puffer_soccer_flamegraph_full
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --account=torch_pr_355_tandon_advanced
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

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH_BASE/.cache/apptainer}"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$LOCAL_TMP_BASE/apptainer-tmp-${SLURM_JOB_ID:-local}}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_BASE/.cache/uv}"
export TMPDIR="${TMPDIR:-$SCRATCH_BASE/tmp}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SCRATCH_BASE/.venvs/puffer-soccer-flamegraph-${SLURM_JOB_ID:-local}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

mkdir -p \
    "$APPTAINER_CACHEDIR" \
    "$APPTAINER_TMPDIR" \
    "$UV_CACHE_DIR" \
    "$TMPDIR" \
    "$(dirname "$UV_PROJECT_ENVIRONMENT")" \
    "$REPO_ROOT/sbatch/logs" \
    "$REPO_ROOT/experiments" \
    "$JOB_COPY_BASE"

rm -rf "$JOB_COPY_ROOT"
mkdir -p "$JOB_COPY_ROOT"
rsync -a --delete --exclude 'sbatch-tmp/' "$REPO_ROOT/" "$JOB_WORKSPACE_ROOT/"

printf 'REPO_ROOT=%s\n' "$REPO_ROOT"
printf 'JOB_WORKSPACE_ROOT=%s\n' "$JOB_WORKSPACE_ROOT"

module purge

FLAME_OUTPUT="/flame-out/flame-full-${SLURM_JOB_ID:-local}.svg"

apptainer exec --nv \
    --bind "$JOB_WORKSPACE_ROOT:/workspace" \
    --bind "$REPO_ROOT/experiments:/flame-out" \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    bash -c '
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

uv sync --python /opt/python/cp312-cp312/bin/python --group dev

# Convert SLURM signal to a clean stop: SIGINT tells py-spy to write the
# flamegraph and exit gracefully instead of being killed mid-write.
trap "echo Caught TERM, signaling py-spy to flush flamegraph; kill -INT \$PYSPY_PID 2>/dev/null" TERM

uv run py-spy record -o "'"$FLAME_OUTPUT"'" --rate 20 --subprocesses -- \
    python -u scripts/train_pufferl.py \
        --players-per-team 5 \
        --vec-backend auto \
        --ppo-iterations 1000 &
PYSPY_PID=$!
wait "$PYSPY_PID"
'

echo "Flame graph written to $REPO_ROOT/experiments/flame-full-${SLURM_JOB_ID:-local}.svg"
