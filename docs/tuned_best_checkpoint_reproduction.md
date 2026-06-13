# Tuned Best-Checkpoint Reproduction

This branch tuned PPO hyperparameters against the fixed dynamic-team-size best checkpoint and
then retrained with the best confirmed Optuna configuration.

## Reproduce the Optuna tuning run

```bash
sbatch \
  --job-name=tune_best_checkpoint_native \
  --nodes=1 \
  --cpus-per-task=32 \
  --mem=64G \
  --gres=gpu:1 \
  --time=48:00:00 \
  --account=torch_pr_45_tandon_advanced \
  --chdir=/scratch/fyy2003/repos/Puffer-Soccer \
  --output=/scratch/fyy2003/repos/Puffer-Soccer/sbatch/logs/%x-%j.out \
  --error=/scratch/fyy2003/repos/Puffer-Soccer/sbatch/logs/%x-%j.err \
  --export=ALL,TUNE_MATRIX_LABEL=best_checkpoint_optuna_100m_native,TUNE_RL_ALG=self_play,TUNE_KL_MODE=auto,TUNE_TOTAL_TIMESTEPS=100000000,TUNE_MAX_RUNS=8,TUNE_CONFIRM_CANDIDATES=3,TUNE_CANDIDATE_TOTAL_SEEDS=3,TUNE_FINAL_EVAL_GAMES=256,TUNE_VEC_BACKEND=native,TUNE_AUTOTUNE_SECONDS=0.1,TUNE_DEVICE=cuda,TUNE_OUTPUT_ROOT=/scratch/fyy2003/repos/Puffer-Soccer/experiments/best_checkpoint_rl_tuning,TUNE_RUNTIME_CONFIG_PATH=/scratch/fyy2003/repos/Puffer-Soccer/experiments/rl_tuning_runtime_config_native.json,TUNE_EXTRA_ARGS='--study-name best_checkpoint_optuna_100m_native --num-envs 64' \
  sbatch/run_tune_self_play_variant.sh
```

The best confirmed tuning output was written to:

```text
experiments/best_checkpoint_rl_tuning/best_checkpoint_optuna_100m_native/best_hyperparameters.json
```

## Reproduce the tuned retrain

```bash
sbatch \
  --job-name=puffer_tuned_retrain \
  --nodes=1 \
  --cpus-per-task=32 \
  --mem=64G \
  --gres=gpu:1 \
  --time=48:00:00 \
  --account=torch_pr_45_tandon_advanced \
  --chdir=/scratch/fyy2003/repos/Puffer-Soccer \
  --output=/scratch/fyy2003/repos/Puffer-Soccer/sbatch/logs/%x-%j.out \
  --error=/scratch/fyy2003/repos/Puffer-Soccer/sbatch/logs/%x-%j.err \
  --export=ALL,TRAIN_AUTOMODE_HYPERPARAMETERS_PATH=experiments/autoload_hyperparameters_best_checkpoint_optuna_100m_native.json,TRAIN_AUTOMODE_PRETUNE_VECENV=0,TRAIN_AUTOMODE_PPO_ITERATIONS=100000,TRAIN_AUTOMODE_NO_OPPONENT_MIN_ITERATIONS=0,TRAIN_AUTOMODE_NO_OPPONENT_MAX_ITERATIONS=0,TRAIN_AUTOMODE_FINAL_BEST_EVAL_GAMES=512,TRAIN_AUTOMODE_EXTRA_ARGS='--rl-alg self_play --kl-regularization-mode auto --past-kl-coef 0.001793480721178979 --uniform-kl-base-coef 0.019292020778203717 --uniform-kl-power 0.8763891522960383 --best-checkpoint-eval --wandb-group tuned-best-checkpoint-retrain' \
  sbatch/train_automode.sbatch
```

The promoted checkpoint in this branch is the periodic peak from that retrain:

```text
run_id=zawz33hx
epoch=65000
global_step=5324800000
artifact_ref=emerge_/robot-soccer/best-checkpoint-zawz33hx-epoch-065000:best
```

The periodic 64-game best-checkpoint eval at that checkpoint was:

```text
win_rate=0.914
score_diff=1.109
```
