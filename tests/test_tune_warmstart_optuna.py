"""Tests for the warm-start Optuna tuning helper."""

from __future__ import annotations
# pylint: disable=wrong-import-position

from pathlib import Path
import json
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


def test_extract_metrics_from_log_uses_latest_periodic_eval_when_no_final_line() -> None:
    """Parse the decisive warm-start metrics from the new single-eval handoff flow.

    Warm-start no longer runs a second confirmation evaluation after the policy clears the
    gate. This regression test keeps the tuner aligned with that simpler control flow by
    checking that the latest ordinary no-opponent evaluation is treated as the trial result
    when no explicit final line is present.
    """

    log_text = """
    No-opponent eval (epoch=60, games=16): train_scale=1.000, goal_rate=0.125,
    multi_goal_rate=0.000, mean_goals_scored=0.12, own_goal_rate=0.000,
    mean_first_goal_step=541.00
    No-opponent eval (epoch=64, games=16): train_scale=1.000, goal_rate=0.188,
    multi_goal_rate=0.125, mean_goals_scored=0.31, own_goal_rate=0.000,
    mean_first_goal_step=523.12
    No-opponent phase complete: epoch=64, goal_rate=0.188, train_scale=1.000
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


def test_build_srun_command_skips_gpu_request_for_cpu_trials() -> None:
    """Omit the GPU resource request when the warm-start tuner is run on CPU.

    The Optuna helper now supports a CPU-only Slurm fallback so tuning can still be
    exercised when GPU quota is temporarily exhausted. This test keeps that contract
    explicit by checking that the generated `srun` command drops `--gres=gpu:1` while
    preserving the rest of the requested Slurm resources.
    """

    args = SimpleNamespace(
        account="torch_pr_45_tandon_advanced",
        cpus_per_task=8,
        mem="16G",
        job_time="00:05:00",
        partition=None,
        device="cpu",
    )

    command = tuner.build_srun_command(args=args, command=("python", "train.py"))

    assert "--gres=gpu:1" not in command
    assert command[:8] == [
        "srun",
        "--account",
        "torch_pr_45_tandon_advanced",
        "--cpus-per-task",
        "8",
        "--mem",
        "16G",
        "--time",
    ]
    assert command[-2:] == ["python", "train.py"]


def test_read_hyperparameter_defaults_falls_back_to_builtin_defaults(
    tmp_path: Path,
) -> None:
    """Use parser defaults when no standardized warm-start recipe exists yet.

    The tuner should be usable on a brand-new checkout before any earlier sweep has written
    `experiments/autoload_hyperparameters.json`. This regression test keeps that bootstrap
    path alive by asserting the fallback recipe matches the training parser's own curated
    reusable defaults.
    """

    defaults = tuner.read_hyperparameter_defaults(tmp_path / "missing.json")

    assert (
        defaults
        == tuner.train_pufferl.standardized_hyperparameter_defaults(
            tuner.train_pufferl.base_training_arg_defaults()
        )
    )


def test_persist_best_hyperparameters_preserves_vecenv_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    """Write the winning warm-start recipe without erasing pretuned vecenv metadata.

    Warm-start tuning and vecenv pretuning produce different parts of the shared autoload
    file. This test ensures the warm-start saver keeps the existing vecenv defaults and
    benchmark block intact while replacing the train defaults with the new winning recipe.
    """

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "vecenv_defaults": {
                    "vec_backend": "multiprocessing",
                    "num_envs": 96,
                    "vec_num_shards": 24,
                    "vec_batch_size": 24,
                },
                "vecenv_benchmark": {
                    "backend": "multiprocessing",
                    "num_envs": 96,
                    "sps": 1234,
                },
            }
        ),
        encoding="utf-8",
    )
    best_path = tmp_path / "best_result.json"
    best_path.write_text("{}", encoding="utf-8")
    canonical_path = tmp_path / "canonical_autoload.json"
    monkeypatch.setattr(tuner.train_pufferl, "STANDARD_HYPERPARAMETERS_PATH", canonical_path)

    local_path, written_canonical_path = tuner.persist_best_hyperparameters(
        output_dir=tmp_path / "study",
        baseline_hyperparameters_path=baseline_path,
        source_path=best_path,
        effective_hyperparameters={
            "learning_rate": 0.001,
            "gamma": 0.995,
            "gae_lambda": 0.9,
            "update_epochs": 3,
            "clip_coef": 0.2,
            "vf_coef": 1.0,
            "vf_clip_coef": 0.2,
            "max_grad_norm": 0.8,
            "ent_coef": 1e-6,
            "prio_alpha": 0.8,
            "prio_beta0": 0.2,
            "train_batch_size": 20480,
            "bptt_horizon": 64,
            "minibatch_size": 5120,
            "no_opponent_map_scale_ladder": "0.2,0.4,0.6,0.8,1.0",
            "no_opponent_map_scale_start": 0.45,
            "no_opponent_map_scale_end": 1.0,
            "no_opponent_map_scale_power": 3.0,
            "no_opponent_map_scale_full_progress": 0.6,
        },
    )

    assert written_canonical_path == canonical_path
    for path in (local_path, canonical_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["train_defaults"]["learning_rate"] == 0.001
        assert payload["vecenv_defaults"]["vec_backend"] == "multiprocessing"
        assert payload["vecenv_benchmark"]["sps"] == 1234
        assert payload["source"]["label"] == "best_no_opponent_warmstart_optuna_result"
