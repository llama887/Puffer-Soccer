"""Tune no-opponent RL and field-curriculum hyperparameters with Puffer sweeps."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pufferlib.sweep

import train_pufferl


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT_PATH = Path(__file__).resolve().with_name("train_pufferl.py")


@dataclass(frozen=True)
class TrialResult:
    """Store one no-opponent tuning subprocess outcome in ranking-friendly form.

    The sweep driver launches the normal training script in a fresh subprocess so every trial
    starts from a clean interpreter and its own random seed. This dataclass keeps the full
    objective together with the exact command and summary path so later analysis does not
    depend on terminal logs.
    """

    phase: str
    trial_index: int
    seed: int
    returncode: int
    runtime_seconds: float
    goal_rate: float
    multi_goal_rate: float
    mean_goals_scored: float
    mean_first_goal_step: float
    own_goal_rate: float
    objective_score: float
    actual_global_step: int
    failed: bool
    suggestion: dict[str, object]
    effective_hyperparameters: dict[str, object]
    summary_path: Path
    log_path: Path
    command: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        """Convert the trial result into plain JSON data for history logs."""

        return {
            "phase": self.phase,
            "trial_index": self.trial_index,
            "seed": self.seed,
            "returncode": self.returncode,
            "runtime_seconds": self.runtime_seconds,
            "goal_rate": self.goal_rate,
            "multi_goal_rate": self.multi_goal_rate,
            "mean_goals_scored": self.mean_goals_scored,
            "mean_first_goal_step": self.mean_first_goal_step,
            "own_goal_rate": self.own_goal_rate,
            "objective_score": self.objective_score,
            "actual_global_step": self.actual_global_step,
            "failed": self.failed,
            "suggestion": self.suggestion,
            "effective_hyperparameters": self.effective_hyperparameters,
            "summary_path": str(self.summary_path),
            "log_path": str(self.log_path),
            "command": list(self.command),
        }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI used to tune no-opponent curriculum and RL hyperparameters.

    The sweep freezes the runtime vector layout up front and then searches only over the
    learning settings and the map-scale curriculum. Evaluation always happens on the full
    field so the chosen configuration must genuinely transfer beyond the easier early
    curriculum.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--players-per-team", type=int, default=3)
    parser.add_argument(
        "--vec-backend",
        type=str,
        default="auto",
        choices=["native", "serial", "multiprocessing", "auto"],
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--vec-num-shards", type=int, default=None)
    parser.add_argument("--vec-batch-size", type=int, default=None)
    parser.add_argument("--autotune-max-num-envs", type=int, default=None)
    parser.add_argument("--autotune-max-num-shards", type=int, default=None)
    parser.add_argument("--autotune-seconds", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--total-timesteps", type=int, default=4_194_304)
    parser.add_argument("--no-opponent-eval-games", type=int, default=16)
    parser.add_argument("--no-opponent-eval-max-steps", type=int, default=600)
    parser.add_argument("--max-runs", type=int, default=8)
    parser.add_argument("--confirm-candidates", type=int, default=3)
    parser.add_argument("--candidate-total-seeds", type=int, default=3)
    parser.add_argument("--method", type=str, default="Protein")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "experiments/no_opponent_rl_tuning/"
            + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        ),
    )
    return parser


def build_sweep_config() -> dict[str, object]:
    """Define the Puffer sweep search space for the no-opponent curriculum task.

    The search includes both PPO hyperparameters and curriculum geometry. The curriculum only
    changes the map scale during training; evaluation remains on the full-size field.
    """

    return {
        "metric": "no_opponent_objective",
        "goal": "maximize",
        "rollout": {
            "horizon": {
                "distribution": "uniform_pow2",
                "min": 16,
                "max": 128,
                "mean": 32,
                "scale": "auto",
            },
            "batch_multiple": {
                "distribution": "uniform_pow2",
                "min": 1,
                "max": 8,
                "mean": 2,
                "scale": "auto",
            },
            "minibatch_divisor": {
                "distribution": "uniform_pow2",
                "min": 1,
                "max": 8,
                "mean": 2,
                "scale": "auto",
            },
        },
        "train": {
            "learning_rate": {
                "distribution": "log_normal",
                "min": 1e-4,
                "max": 1e-2,
                "mean": 2e-3,
                "scale": 0.7,
            },
            "update_epochs": {
                "distribution": "int_uniform",
                "min": 1,
                "max": 6,
                "mean": 3,
                "scale": "auto",
            },
            "gamma": {
                "distribution": "logit_normal",
                "min": 0.97,
                "max": 0.999,
                "mean": 0.99,
                "scale": "auto",
            },
            "gae_lambda": {
                "distribution": "logit_normal",
                "min": 0.85,
                "max": 0.99,
                "mean": 0.95,
                "scale": "auto",
            },
            "clip_coef": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.3,
                "mean": 0.2,
                "scale": "auto",
            },
            "vf_coef": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 2.0,
                "mean": 1.0,
                "scale": "auto",
            },
            "vf_clip_coef": {
                "distribution": "uniform",
                "min": 0.1,
                "max": 0.4,
                "mean": 0.2,
                "scale": "auto",
            },
            "max_grad_norm": {
                "distribution": "uniform",
                "min": 0.25,
                "max": 1.0,
                "mean": 0.5,
                "scale": "auto",
            },
            "ent_coef": {
                "distribution": "log_normal",
                "min": 1e-7,
                "max": 1e-3,
                "mean": 1e-5,
                "scale": "auto",
            },
            "prio_alpha": {
                "distribution": "logit_normal",
                "min": 0.05,
                "max": 0.99,
                "mean": 0.8,
                "scale": "auto",
            },
            "prio_beta0": {
                "distribution": "logit_normal",
                "min": 0.05,
                "max": 0.99,
                "mean": 0.2,
                "scale": "auto",
            },
        },
        "curriculum": {
            "start_scale": {
                "distribution": "uniform",
                "min": 0.2,
                "max": 0.8,
                "mean": 0.45,
                "scale": "auto",
            },
            "scale_power": {
                "distribution": "uniform",
                "min": 0.5,
                "max": 8.0,
                "mean": 3.0,
                "scale": "auto",
            },
            "full_progress": {
                "distribution": "uniform",
                "min": 0.2,
                "max": 1.0,
                "mean": 0.6,
                "scale": "auto",
            },
        },
    }


def append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    """Append one JSON record to a history file for later inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def sample_seed(rng: random.Random) -> int:
    """Draw one fresh training seed for a trial or confirmation rerun."""

    return rng.randint(1, 2**31 - 1)


def resolve_runtime_vec_config(
    args,
) -> tuple[train_pufferl.VecEnvConfig, train_pufferl.BenchmarkResult | None]:
    """Pick the one fixed vector runtime reused by every RL sweep trial."""

    runtime_args = SimpleNamespace(
        vec_backend=args.vec_backend,
        players_per_team=args.players_per_team,
        num_envs=args.num_envs,
        vec_num_shards=args.vec_num_shards,
        vec_batch_size=args.vec_batch_size,
        autotune_seconds=args.autotune_seconds,
        autotune_max_num_envs=args.autotune_max_num_envs,
        autotune_max_num_shards=args.autotune_max_num_shards,
    )
    return train_pufferl.resolve_training_vec_config(runtime_args)


def render_vec_cli_args(vec_config: train_pufferl.VecEnvConfig) -> list[str]:
    """Render one frozen vector config back into `train_pufferl.py` CLI flags."""

    if vec_config.backend == "native":
        return [
            "--vec-backend",
            "native",
            "--num-envs",
            str(vec_config.shard_num_envs),
        ]

    args = [
        "--vec-backend",
        vec_config.backend,
        "--num-envs",
        str(train_pufferl.total_sim_envs(vec_config)),
        "--vec-num-shards",
        str(vec_config.num_shards),
    ]
    if vec_config.batch_size is not None:
        args.extend(["--vec-batch-size", str(vec_config.batch_size)])
    return args


def resolve_rollout_hyperparameters(
    suggestion: Mapping[str, object], total_agents: int
) -> dict[str, int]:
    """Convert latent sweep parameters into legal PuffeRL rollout sizes."""

    rollout = cast_mapping(suggestion["rollout"])
    horizon = int(rollout["horizon"])
    batch_multiple = int(rollout["batch_multiple"])
    minibatch_divisor = int(rollout["minibatch_divisor"])
    batch_size = total_agents * horizon * max(1, batch_multiple)
    requested_minibatch = max(horizon, batch_size // max(1, minibatch_divisor))
    minibatch_size = train_pufferl.choose_valid_minibatch_size(
        batch_size=batch_size,
        horizon=horizon,
        requested_size=requested_minibatch,
    )
    return {
        "bptt_horizon": horizon,
        "train_batch_size": batch_size,
        "minibatch_size": minibatch_size,
    }


def build_trial_command(
    *,
    args,
    vec_config: train_pufferl.VecEnvConfig,
    suggestion: Mapping[str, object],
    seed: int,
    summary_path: Path,
) -> list[str]:
    """Assemble the exact subprocess command for one no-opponent tuning trial."""

    rollout = resolve_rollout_hyperparameters(suggestion, args.total_agents)
    train = cast_mapping(suggestion["train"])
    curriculum = cast_mapping(suggestion["curriculum"])
    command = [
        sys.executable,
        str(TRAIN_SCRIPT_PATH),
        "--players-per-team",
        str(args.players_per_team),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--total-timesteps",
        str(args.total_timesteps),
        "--train-batch-size",
        str(rollout["train_batch_size"]),
        "--bptt-horizon",
        str(rollout["bptt_horizon"]),
        "--minibatch-size",
        str(rollout["minibatch_size"]),
        "--update-epochs",
        str(int(train["update_epochs"])),
        "--learning-rate",
        str(float(train["learning_rate"])),
        "--gamma",
        str(float(train["gamma"])),
        "--gae-lambda",
        str(float(train["gae_lambda"])),
        "--clip-coef",
        str(float(train["clip_coef"])),
        "--vf-coef",
        str(float(train["vf_coef"])),
        "--vf-clip-coef",
        str(float(train["vf_clip_coef"])),
        "--max-grad-norm",
        str(float(train["max_grad_norm"])),
        "--ent-coef",
        str(float(train["ent_coef"])),
        "--prio-alpha",
        str(float(train["prio_alpha"])),
        "--prio-beta0",
        str(float(train["prio_beta0"])),
        "--no-opponent-team",
        "--no-opponent-map-scale-start",
        str(float(curriculum["start_scale"])),
        "--no-opponent-map-scale-end",
        "1.0",
        "--no-opponent-map-scale-power",
        str(float(curriculum["scale_power"])),
        "--no-opponent-map-scale-full-progress",
        str(float(curriculum["full_progress"])),
        "--no-opponent-eval-games",
        str(args.no_opponent_eval_games),
        "--no-opponent-eval-max-steps",
        str(args.no_opponent_eval_max_steps),
        "--checkpoint-interval",
        str(10**9),
        "--no-past-iterate-eval",
        "--no-export-videos",
        "--no-wandb",
        "--no-regularization",
        "--run-summary-path",
        str(summary_path),
    ]
    command.extend(render_vec_cli_args(vec_config))
    return command


def objective_score(
    *,
    multi_goal_rate: float,
    mean_goals_scored: float,
    goal_rate: float,
    own_goal_rate: float,
    mean_first_goal_step: float,
) -> float:
    """Compress the repeated-scoring objective into one scalar for the sweep backend.

    Repeated scoring after full-size resets is the real goal, so `multi_goal_rate` dominates.
    Mean goals scored breaks ties between equally repeatable policies, then goal rate and
    first-goal speed refine the ranking. Own goals are penalized explicitly because they make
    a video unsuitable even if the agent still scores often.
    """

    return (
        10.0 * multi_goal_rate
        + 2.0 * mean_goals_scored
        + 1.0 * goal_rate
        - 2.0 * own_goal_rate
        - 0.002 * mean_first_goal_step
    )


def trial_sort_key(result: TrialResult) -> tuple[float, float, float, float, float, float]:
    """Rank trials by repeated scoring quality on the full-size evaluation map."""

    return (
        result.multi_goal_rate,
        result.mean_goals_scored,
        result.goal_rate,
        -result.own_goal_rate,
        -result.mean_first_goal_step,
        -result.runtime_seconds,
    )


def load_trial_result(
    *,
    phase: str,
    trial_index: int,
    seed: int,
    suggestion: Mapping[str, object],
    command: Sequence[str],
    summary_path: Path,
    log_path: Path,
    returncode: int,
) -> TrialResult:
    """Read one completed subprocess summary and normalize it into a `TrialResult`."""

    failed = returncode != 0 or not summary_path.exists()
    runtime_seconds = float("inf") if failed else 0.0
    goal_rate = 0.0
    multi_goal_rate = 0.0
    mean_goals_scored = 0.0
    mean_first_goal_step = 1e9
    own_goal_rate = 1.0
    score = -1e9
    actual_global_step = 0
    effective_hyperparameters: dict[str, object] = {}

    if not failed:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        objective = cast_mapping(payload.get("objective_metrics") or {})
        runtime_seconds = float(payload.get("runtime_seconds", 0.0))
        goal_rate = float(objective.get("goal_rate", 0.0))
        multi_goal_rate = float(objective.get("multi_goal_rate", 0.0))
        mean_goals_scored = float(objective.get("mean_goals_scored", 0.0))
        mean_first_goal_step = float(objective.get("mean_first_goal_step", 10**9))
        own_goal_rate = float(objective.get("own_goal_rate", 1.0))
        score = objective_score(
            multi_goal_rate=multi_goal_rate,
            mean_goals_scored=mean_goals_scored,
            goal_rate=goal_rate,
            own_goal_rate=own_goal_rate,
            mean_first_goal_step=mean_first_goal_step,
        )
        actual_global_step = int(payload.get("global_step", 0))
        effective = payload.get("effective_hyperparameters")
        if isinstance(effective, Mapping):
            effective_hyperparameters = dict(effective)

    return TrialResult(
        phase=phase,
        trial_index=trial_index,
        seed=seed,
        returncode=returncode,
        runtime_seconds=runtime_seconds,
        goal_rate=goal_rate,
        multi_goal_rate=multi_goal_rate,
        mean_goals_scored=mean_goals_scored,
        mean_first_goal_step=mean_first_goal_step,
        own_goal_rate=own_goal_rate,
        objective_score=score,
        actual_global_step=actual_global_step,
        failed=failed,
        suggestion=dict(suggestion),
        effective_hyperparameters=effective_hyperparameters,
        summary_path=summary_path,
        log_path=log_path,
        command=tuple(command),
    )


def run_trial(
    *,
    phase: str,
    trial_index: int,
    args,
    vec_config: train_pufferl.VecEnvConfig,
    suggestion: Mapping[str, object],
    seed: int,
    output_dir: Path,
) -> TrialResult:
    """Execute one training subprocess and return its no-opponent objective metrics."""

    trial_dir = output_dir / phase / f"trial_{trial_index:03d}_seed_{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    summary_path = trial_dir / "summary.json"
    log_path = trial_dir / "output.log"
    command = build_trial_command(
        args=args,
        vec_config=vec_config,
        suggestion=suggestion,
        seed=seed,
        summary_path=summary_path,
    )
    train_section = cast_mapping(suggestion["train"])
    curriculum = cast_mapping(suggestion["curriculum"])
    rollout = resolve_rollout_hyperparameters(suggestion, args.total_agents)
    print(
        f"[{phase} {trial_index:03d}] seed={seed} "
        f"lr={mapping_float(train_section, 'learning_rate'):.3g} "
        f"horizon={rollout['bptt_horizon']} "
        f"start_scale={mapping_float(curriculum, 'start_scale'):.3f} "
        f"power={mapping_float(curriculum, 'scale_power'):.3f}"
    )
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    result = load_trial_result(
        phase=phase,
        trial_index=trial_index,
        seed=seed,
        suggestion=suggestion,
        command=command,
        summary_path=summary_path,
        log_path=log_path,
        returncode=completed.returncode,
    )
    status = "failed" if result.failed else "ok"
    print(
        f"[{phase} {trial_index:03d}] {status} "
        f"multi_goal_rate={result.multi_goal_rate:.3f} "
        f"mean_goals={result.mean_goals_scored:.3f} "
        f"goal_rate={result.goal_rate:.3f} "
        f"first_goal={result.mean_first_goal_step:.1f} "
        f"own_goal={result.own_goal_rate:.3f}"
    )
    return result


def cast_mapping(value: object) -> Mapping[str, object]:
    """Type-check that a JSON-like object is a mapping before subscripting it."""

    if not isinstance(value, Mapping):
        raise TypeError("expected mapping")
    return value


def mapping_float(mapping: Mapping[str, object], key: str) -> float:
    """Read one mapping value as a float with a clear error if the type is wrong."""

    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected numeric value for {key}")
    return float(value)


def summarize_candidate_runs(runs: Sequence[TrialResult]) -> dict[str, object]:
    """Aggregate multiple fresh-seed runs for one candidate configuration."""

    return {
        "mean_goal_rate": float(np.mean([run.goal_rate for run in runs])),
        "mean_multi_goal_rate": float(np.mean([run.multi_goal_rate for run in runs])),
        "mean_goals_scored": float(np.mean([run.mean_goals_scored for run in runs])),
        "mean_first_goal_step": float(np.mean([run.mean_first_goal_step for run in runs])),
        "mean_own_goal_rate": float(np.mean([run.own_goal_rate for run in runs])),
        "mean_runtime_seconds": float(np.mean([run.runtime_seconds for run in runs])),
        "mean_objective_score": float(np.mean([run.objective_score for run in runs])),
        "runs": [run.to_record() for run in runs],
        "effective_hyperparameters": runs[0].effective_hyperparameters,
        "suggestion": runs[0].suggestion,
    }


def print_top_results(results: Sequence[TrialResult], limit: int = 5) -> None:
    """Print a compact leaderboard for the strongest finished trials so far."""

    successful = sorted(
        (result for result in results if not result.failed), key=trial_sort_key
    )
    if not successful:
        print("No successful trials yet.")
        return
    print("Top trials")
    for result in successful[-limit:][::-1]:
        hypers = result.effective_hyperparameters
        print(
            f"  trial={result.phase}:{result.trial_index:03d} "
            f"seed={result.seed} multi_goal_rate={result.multi_goal_rate:.3f} "
            f"mean_goals={result.mean_goals_scored:.3f} goal_rate={result.goal_rate:.3f} "
            f"first_goal={result.mean_first_goal_step:.1f} "
            f"batch={hypers.get('train_batch_size')} horizon={hypers.get('bptt_horizon')} "
            f"lr={hypers.get('learning_rate')} start_scale={hypers.get('no_opponent_map_scale_start')}"
        )


def main() -> None:
    """Tune no-opponent RL and map-curriculum hyperparameters, then confirm the winner."""

    parser = build_parser()
    args = parser.parse_args()
    if args.max_runs < 1:
        raise ValueError("max_runs must be positive")
    if args.confirm_candidates < 1:
        raise ValueError("confirm_candidates must be positive")
    if args.candidate_total_seeds < 1:
        raise ValueError("candidate_total_seeds must be positive")

    train_pufferl.load_env_file(".env")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "trial_history.jsonl"
    aggregate_path = output_dir / "candidate_aggregates.json"
    best_path = output_dir / "best_hyperparameters.json"
    local_autoload_path = output_dir / "autoload_hyperparameters.json"

    vec_config, autotune_result = resolve_runtime_vec_config(args)
    if vec_config.backend != "native":
        raise RuntimeError(
            "no-opponent field-curriculum tuning currently requires the native vec backend"
        )
    device = train_pufferl.resolve_device(args.device)
    args.device = device
    args.total_agents = (
        train_pufferl.total_sim_envs(vec_config) * args.players_per_team * 2
    )

    runtime_payload = {
        "resolved_device": device,
        "vec_config": train_pufferl.serialize_vec_config(vec_config),
        "total_agents": args.total_agents,
        "autotune_result": None
        if autotune_result is None
        else {
            "backend": autotune_result.backend,
            "shard_num_envs": autotune_result.shard_num_envs,
            "num_shards": autotune_result.num_shards,
            "batch_size": autotune_result.batch_size,
            "sps": autotune_result.sps,
            "cpu_avg": autotune_result.cpu_avg,
            "cpu_peak": autotune_result.cpu_peak,
        },
    }
    train_pufferl.write_json_record(output_dir / "runtime_config.json", runtime_payload)
    if autotune_result is not None:
        print(
            "Autotune selected fixed runtime: "
            f"{train_pufferl.format_benchmark_result(autotune_result)}"
        )
    print(
        "Frozen runtime: "
        f"backend={vec_config.backend}, shard_num_envs={vec_config.shard_num_envs}, "
        f"num_shards={vec_config.num_shards}, batch_size={vec_config.batch_size}, "
        f"total_agents={args.total_agents}, device={device}"
    )

    method_cls = getattr(pufferlib.sweep, args.method)
    sweep = method_cls(build_sweep_config())
    rng = random.Random(time.time_ns())
    all_results: list[TrialResult] = []

    for trial_index in range(1, args.max_runs + 1):
        suggestion, _ = sweep.suggest(None)
        seed = sample_seed(rng)
        result = run_trial(
            phase="search",
            trial_index=trial_index,
            args=args,
            vec_config=vec_config,
            suggestion=suggestion,
            seed=seed,
            output_dir=output_dir,
        )
        all_results.append(result)
        append_jsonl(history_path, result.to_record())
        observe_cost = (
            result.runtime_seconds if np.isfinite(result.runtime_seconds) else 1e12
        )
        sweep.observe(
            suggestion,
            result.objective_score,
            observe_cost,
            is_failure=result.failed,
        )
        print_top_results(all_results)

    successful = sorted(
        (result for result in all_results if not result.failed),
        key=trial_sort_key,
        reverse=True,
    )
    if not successful:
        raise RuntimeError("all tuning trials failed")

    candidate_summaries: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(
        successful[: args.confirm_candidates], start=1
    ):
        candidate_runs = [candidate]
        for extra_idx in range(2, args.candidate_total_seeds + 1):
            rerun = run_trial(
                phase=f"confirm_{candidate_index}",
                trial_index=extra_idx,
                args=args,
                vec_config=vec_config,
                suggestion=candidate.suggestion,
                seed=sample_seed(rng),
                output_dir=output_dir,
            )
            candidate_runs.append(rerun)
            append_jsonl(history_path, rerun.to_record())

        summary = summarize_candidate_runs(candidate_runs)
        summary["candidate_index"] = candidate_index
        candidate_summaries.append(summary)
        print(
            f"Candidate {candidate_index}: mean_multi_goal_rate={summary['mean_multi_goal_rate']:.3f} "
            f"mean_goals_scored={summary['mean_goals_scored']:.3f} "
            f"mean_first_goal_step={summary['mean_first_goal_step']:.1f} "
            f"seeds={len(candidate_runs)}"
        )

    candidate_summaries.sort(
        key=lambda summary: (
            mapping_float(summary, "mean_multi_goal_rate"),
            mapping_float(summary, "mean_goals_scored"),
            mapping_float(summary, "mean_goal_rate"),
            -mapping_float(summary, "mean_own_goal_rate"),
            -mapping_float(summary, "mean_first_goal_step"),
        ),
        reverse=True,
    )
    train_pufferl.write_json_record(aggregate_path, {"candidates": candidate_summaries})
    best_candidate = candidate_summaries[0]
    train_pufferl.write_json_record(best_path, best_candidate)
    best_hypers = cast_mapping(best_candidate["effective_hyperparameters"])
    train_pufferl.write_standardized_hyperparameters(
        path=local_autoload_path,
        effective_hyperparameters=best_hypers,
        source_path=best_path,
        source_label="best_confirmed_no_opponent_tuning_result",
        source_num_agents=args.total_agents,
    )
    train_pufferl.write_standardized_hyperparameters(
        path=train_pufferl.STANDARD_HYPERPARAMETERS_PATH,
        effective_hyperparameters=best_hypers,
        source_path=best_path,
        source_label="best_confirmed_no_opponent_tuning_result",
        source_num_agents=args.total_agents,
    )

    print("Best confirmed hyperparameters")
    for key in (
        "train_batch_size",
        "bptt_horizon",
        "minibatch_size",
        "learning_rate",
        "gamma",
        "gae_lambda",
        "update_epochs",
        "clip_coef",
        "vf_coef",
        "vf_clip_coef",
        "max_grad_norm",
        "ent_coef",
        "prio_alpha",
        "prio_beta0",
        "no_opponent_map_scale_start",
        "no_opponent_map_scale_end",
        "no_opponent_map_scale_power",
        "no_opponent_map_scale_full_progress",
        "total_timesteps",
    ):
        print(f"  {key}={best_hypers[key]}")
    print(
        f"Confirmed mean multi-goal rate={best_candidate['mean_multi_goal_rate']:.3f}, "
        f"mean goals scored={best_candidate['mean_goals_scored']:.3f}, "
        f"mean first-goal step={best_candidate['mean_first_goal_step']:.1f}"
    )
    print(f"Standardized autoload file: {train_pufferl.STANDARD_HYPERPARAMETERS_PATH}")
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
