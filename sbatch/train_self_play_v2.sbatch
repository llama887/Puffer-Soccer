#!/bin/bash
#
#SBATCH --job-name=train_self_play_v2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_45_tandon_advanced
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fyy2003@nyu.edu
#SBATCH --chdir=/scratch/fyy2003/repos/Puffer-Soccer
#SBATCH --output=/scratch/fyy2003/repos/Puffer-Soccer/sbatch/logs/%x-%j.out
#SBATCH --error=/scratch/fyy2003/repos/Puffer-Soccer/sbatch/logs/%x-%j.err

set -u
cd /scratch/fyy2003/repos/Puffer-Soccer

CACHE=/scratch/fyy2003/repos/Puffer-Soccer/experiments/warmstart_optuna_v8/trial_005_seed_86939547/cached_warm_start.pt
RUN_DIR=/scratch/fyy2003/repos/Puffer-Soccer/experiments/self_play_v2_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

echo "=== START at $(date -Iseconds) job=${SLURM_JOB_ID} ===" | tee -a "$RUN_DIR/train.log"

# Pre-sync the venv so parallel `uv run` invocations below don't deadlock on .venv/.lock.
uv sync --python /opt/python/cp312-cp312/bin/python 2>&1 | tee -a "$RUN_DIR/train.log"

cleanup() {
    if [ -n "${HEARTBEAT_PID:-}" ] && kill -0 "$HEARTBEAT_PID" 2>/dev/null; then
        kill "$HEARTBEAT_PID"
        wait "$HEARTBEAT_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

nice -n 19 uv run python -u sbatch/gpu_heartbeat.py &
HEARTBEAT_PID=$!
echo "Started GPU heartbeat with PID: $HEARTBEAT_PID" | tee -a "$RUN_DIR/train.log"

# Target 6h of training (~30000 iters at ~83 epochs/min from v1 run).
# Hard cap at 6h0m so the trainer gets a chance to save on SIGINT before the 12h slurm wall.
timeout --signal=INT 21600s uv run python -u scripts/train_pufferl.py \
  --rl-alg self_play \
  --kl-regularization-mode on \
  --device cuda \
  --vec-backend multiprocessing \
  --num-envs 128 \
  --vec-num-shards 16 \
  --vec-batch-size 64 \
  --cached-warm-start-path "$CACHE" \
  --ppo-iterations 30000 \
  --no-wandb \
  --learning-rate 0.0003387386059114101 \
  --gamma 0.9913280905225446 \
  --gae-lambda 0.9645855166980009 \
  --update-epochs 6 \
  --clip-coef 0.21107749458389993 \
  --vf-coef 0.5484168968185966 \
  --vf-clip-coef 0.3592286319433691 \
  --max-grad-norm 0.762834146142455 \
  --ent-coef 0.002287515187112888 \
  --prio-alpha 0.8465529303714516 \
  --prio-beta0 0.6902086020011543 \
  --minibatch-size 10240 \
  --bptt-horizon 64 \
  --past-kl-coef 0.1 \
  --uniform-kl-base-coef 0.05 \
  --uniform-kl-power 0.3 \
  --checkpoint-interval 500 \
  --run-summary-path "$RUN_DIR/summary.json" \
  --video-output "$RUN_DIR/self_play.mp4" \
  2>&1 | tee -a "$RUN_DIR/train.log"

echo "=== END at $(date -Iseconds) (exit $?) ===" | tee -a "$RUN_DIR/train.log"
