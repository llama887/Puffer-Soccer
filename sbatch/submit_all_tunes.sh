#!/bin/bash
#
# Submit all 6 hyperparameter tuning jobs (3 RL algorithms x 2 KL modes).
#
# Each job tunes against the current best checkpoint and exports its winning
# hyperparameters to experiments/tuned_hyperparameters/{variant}.json in the
# standardized format that train_pufferl.py --hyperparameters-path can load.
#
# Usage:
#   bash sbatch/submit_all_tunes.sh          # submit all 6
#   bash sbatch/submit_all_tunes.sh --dry-run # print commands without submitting
#
# After all tuning jobs finish, launch training with the corresponding
# train_{variant}.sbatch files -- they already point to the tuned paths.

set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

SBATCHES=(
    sbatch/tune_self_play_kl_on.sbatch
    sbatch/tune_self_play_kl_off.sbatch
    sbatch/tune_league_kl_on.sbatch
    sbatch/tune_league_kl_off.sbatch
    sbatch/tune_fictitious_play_kl_on.sbatch
    sbatch/tune_fictitious_play_kl_off.sbatch
)

mkdir -p experiments/tuned_hyperparameters

for SBATCH_FILE in "${SBATCHES[@]}"; do
    if [ "$DRY_RUN" = "1" ]; then
        echo "[dry-run] sbatch $SBATCH_FILE"
    else
        JOB_ID=$(sbatch --parsable "$SBATCH_FILE")
        echo "Submitted $SBATCH_FILE -> job $JOB_ID"
    fi
done
