#!/bin/bash
#
#SBATCH --job-name=tune_sp_kl_on
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

export TUNE_MATRIX_LABEL="self_play__kl_on"
export TUNE_RL_ALG="self_play"
export TUNE_KL_MODE="on"

bash sbatch/run_tune_self_play_variant.sh
