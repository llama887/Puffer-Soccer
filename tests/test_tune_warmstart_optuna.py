"""Tests for the warm-start Optuna tuning helper."""

from __future__ import annotations
# pylint: disable=wrong-import-position

from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import tune_warmstart_optuna as tuner


def test_extract_metrics_from_log_prefers_final_eval_line() -> None:
    """Parse the final printed warm-start metrics when the run exits at the gate.

    Failed warm-start trials still emit a useful final no-opponent evaluation line just before
    they raise. The tuner relies on that line so Optuna can learn from unsuccessful trials
    instead of treating them as completely empty failures.
    """

    log_text = """
    No-opponent eval (epoch=56, games=16): train_scale=1.000, goal_rate=0.125,
    multi_goal_rate=0.000, mean_goals_scored=0.12, own_goal_rate=0.000,
    mean_first_goal_step=541.00
    Final no-opponent eval (epoch=64, games=16): goal_rate=0.188,
    multi_goal_rate=0.125, mean_goals_scored=0.31, own_goal_rate=0.000,
    mean_first_goal_step=523.12
    """

    normalized_log = " ".join(line.strip() for line in log_text.splitlines())
    metrics = tuner.extract_metrics_from_log(normalized_log)

    assert metrics is not None
    assert metrics.epoch == 64
    assert metrics.goal_rate == 0.188
    assert metrics.multi_goal_rate == 0.125


def test_build_training_command_keeps_run_warmstart_only() -> None:
    """Set the PPO budget equal to the warm-start cap so self-play never begins.

    The tuner is supposed to optimize the warm-start gate in isolation. This regression test
    checks that the generated command allocates the entire PPO budget to the no-opponent
    phase, disables unrelated evaluation features, and writes a summary file for parsing.
    """

    args = SimpleNamespace(
        hyperparameters_path=REPO_ROOT / "experiments" / "autoload_hyperparameters.json",
        players_per_team=5,
        num_envs=8,
        device="cuda",
        warmstart_min_iterations=32,
        goal_rate_threshold=0.90,
        multi_goal_rate_threshold=0.0,
        eval_games=100,
        eval_max_steps=600,
        map_scale_ladder="0.2,0.4,0.6,0.8,1.0",
    )
    hyperparameters = {
        "ppo_iterations": 128,
        "train_batch_size": 20480,
        "bptt_horizon": 64,
        "minibatch_size": 5120,
        "learning_rate": 0.007,
        "gamma": 0.998,
        "gae_lambda": 0.97,
        "update_epochs": 4,
        "clip_coef": 0.25,
        "vf_coef": 0.9,
        "vf_clip_coef": 0.2,
        "max_grad_norm": 0.8,
        "ent_coef": 1e-7,
        "prio_alpha": 0.9,
        "prio_beta0": 0.7,
        "no_opponent_map_scale_ladder": "0.2,0.4,0.6,0.8,1.0",
    }

    command = tuner.build_training_command(
        args=args,
        hyperparameters=hyperparameters,
        seed=123,
        summary_path=Path("tmp/summary.json"),
    )

    assert "--ppo-iterations" in command
    assert command[command.index("--ppo-iterations") + 1] == "128"
    assert command[command.index("--no-opponent-phase-max-iterations") + 1] == "128"
    assert "--no-past-iterate-eval" in command
    assert "--no-export-videos" in command
    assert "--no-wandb" in command
    assert command[command.index("--no-opponent-map-scale-ladder") + 1] == "0.2,0.4,0.6,0.8,1.0"
    assert command[command.index("--no-opponent-eval-games") + 1] == "100"
    assert command[command.index("--run-summary-path") + 1] == "tmp/summary.json"
