#!/bin/bash
#
#SBATCH --job-name=train_fp_kl_on
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

export TRAIN_RL_ALG="fictitious_play"
export TRAIN_KL_MODE="on"
export TRAIN_HYPERPARAMETERS_PATH="experiments/tuned_hyperparameters/fictitious_play__kl_on.json"

bash sbatch/run_train_variant.sh
