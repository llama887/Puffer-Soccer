#!/bin/bash
#
#SBATCH --job-name=puffer_soccer_flamegraph
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --account=torch_pr_355_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wesleyz2004@gmail.com
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
printf 'APPTAINER_CACHEDIR=%s\n' "$APPTAINER_CACHEDIR"
printf 'APPTAINER_TMPDIR=%s\n' "$APPTAINER_TMPDIR"

module purge

# Write the SVG directly to the real experiments/ dir (not the tmp copy) by
# binding it into the container separately as /flame-out.
FLAME_OUTPUT="/flame-out/flame-${SLURM_JOB_ID:-local}.svg"

apptainer exec --nv \
    --bind "$JOB_WORKSPACE_ROOT:/workspace" \
    --bind "$REPO_ROOT/experiments:/flame-out" \
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

uv sync --python /opt/python/cp312-cp312/bin/python --group dev

OUTPUT="'"$FLAME_OUTPUT"'" \
PLAYERS="${FLAMEGRAPH_PLAYERS:-5}" \
NUM_ENVS="${FLAMEGRAPH_NUM_ENVS:-8}" \
PPO_ITERS="${FLAMEGRAPH_PPO_ITERS:-30}" \
RATE="${FLAMEGRAPH_RATE:-100}" \
NATIVE="${FLAMEGRAPH_NATIVE:-0}" \
bash scripts/flamegraph.sh
'

echo "Flame graph written to $REPO_ROOT/experiments/flame-${SLURM_JOB_ID:-local}.svg"
