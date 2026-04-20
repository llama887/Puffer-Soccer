#!/bin/bash
#
#SBATCH --job-name=train_sp_kl_on
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

# KL-regularized self-play training. Steps:
#   1) Pretune the vector-env layout to maximize SPS on this node
#      (run_train_variant.sh handles this when TRAIN_PRETUNE_VECENV=1).
#   2) Run the warm-start (no-opponent) phase to bootstrap a scoring policy.
#   3) Hand off to KL-regularized self-play for the rest of the budget.
#
# All three phases share one process so the pretuned hyperparams flow
# directly into training without an intermediate file copy.

export TRAIN_RL_ALG="self_play"
export TRAIN_KL_MODE="on"

# Pretune SPS first; the result lands in autoload_hyperparameters.json and
# auto-flows into the training run that follows in the same process.
export TRAIN_PRETUNE_VECENV="1"
export TRAIN_PRETUNE_VECENV_BACKEND="auto"
export TRAIN_PRETUNE_AUTOTUNE_SECONDS="2.0"
export TRAIN_PRETUNE_TRAINING_SECONDS="8.0"
export TRAIN_PRETUNE_SHORTLIST_PER_BACKEND="3"
export TRAIN_PRETUNE_SELECTION_MODE="training_like"
export TRAIN_PRETUNE_DEVICE="auto"
# Use the autoload file pretune writes to so the train phase reads it back.
export TRAIN_HYPERPARAMETERS_PATH="experiments/autoload_hyperparameters.json"

# Warm-start robustness: generous budget for the new env features (bicycle,
# stats, throw-in, offside-as-free-kick). Lower the goal-rate gate slightly
# since the new dynamics are harder than the old.
export TRAIN_NO_OPPONENT_MAX_ITERATIONS="256"
export TRAIN_NO_OPPONENT_GOAL_RATE_THRESHOLD="0.70"
export TRAIN_NO_OPPONENT_EVAL_INTERVAL="4"
export TRAIN_NO_OPPONENT_MAP_SCALE_LADDER="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"

# Skip any stale cached warm-start (obs space and rules have changed).
export TRAIN_SKIP_CACHED_WARM_START="1"

bash sbatch/run_train_variant.sh
