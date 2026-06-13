"""Tune RL hyperparameters against the current best checkpoint with Puffer sweeps."""

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
from typing import Any

import numpy as np

try:
    import optuna as _optuna  # type: ignore[import-not-found]
except ImportError:
    _optuna = None  # type: ignore[assignment]

optuna: Any | None = _optuna

import train_pufferl


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT_PATH = Path(__file__).resolve().with_name("train_pufferl.py")


@dataclass(frozen=True)
class TrialResult:
    """Store the outcome of one tuning subprocess in a ranking-friendly format.

    The sweep driver launches the normal training script as a separate process so every
    trial starts from a clean interpreter and a fresh random seed. Once that subprocess
    finishes, we distill its summary file into this dataclass. Keeping the result structured
    makes it easy to sort trials, write JSON records, and run extra confirmation seeds on
    the most promising configurations.
    """

    phase: str
    trial_index: int
    seed: int
    returncode: int
    runtime_seconds: float
    objective_win_rate: float
    objective_score_diff: float
    actual_global_step: int
    failed: bool
    suggestion: dict[str, object]
    effective_hyperparameters: dict[str, object]
    summary_path: Path
    log_path: Path
    command: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        """Convert the result into plain JSON data for on-disk history logs.

        The tuning run should be auditable after it finishes. This record keeps the raw
        command, summary path, and effective hyperparameters together with the objective so
        later analysis does not depend on terminal scrollback.
        """

        return {
            "phase": self.phase,
            "trial_index": self.trial_index,
            "seed": self.seed,
            "returncode": self.returncode,
            "runtime_seconds": self.runtime_seconds,
            "objective_win_rate": self.objective_win_rate,
            "objective_score_diff": self.objective_score_diff,
            "actual_global_step": self.actual_global_step,
            "failed": self.failed,
            "suggestion": self.suggestion,
            "effective_hyperparameters": self.effective_hyperparameters,
            "summary_path": str(self.summary_path),
            "log_path": str(self.log_path),
            "command": list(self.command),
        }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI used to tune RL hyperparameters against the fixed best checkpoint.

    The arguments focus on three decisions: the fixed training budget shared by every trial,
    the fixed vector runtime that should be reused for all trials, and the breadth of the
    search plus confirmation stages. This script intentionally does not expose vector-layout
    sweep knobs because the user requested RL-only tuning.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument(
        "--vec-backend",
        type=str,
        default="auto",
        choices=["native", "serial", "multiprocessing", "auto"],
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--vec-num-shards", type=int, default=None)
    parser.add_argument("--vec-batch-size", type=int, default=None)
    parser.add_argument("--autotune-max-num-envs", type=int, default=None)
    parser.add_argument("--autotune-max-num-shards", type=int, default=None)
    parser.add_argument("--autotune-seconds", type=float, default=1.5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--rl-alg",
        type=str,
        default=train_pufferl.RL_ALG_SELF_PLAY,
        choices=[
            train_pufferl.RL_ALG_SELF_PLAY,
            train_pufferl.RL_ALG_LEAGUE,
            train_pufferl.RL_ALG_MARLODONNA,
        ],
    )
    parser.add_argument(
        "--kl-regularization-mode",
        type=str,
        default=train_pufferl.KL_REGULARIZATION_AUTO,
        choices=[
            train_pufferl.KL_REGULARIZATION_AUTO,
            train_pufferl.KL_REGULARIZATION_ON,
            train_pufferl.KL_REGULARIZATION_OFF,
        ],
    )
    parser.add_argument("--total-timesteps", type=int, default=30_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy-hidden-size", type=int, default=512)
    parser.add_argument("--policy-encoder-size", type=int, default=512)
    parser.add_argument("--lstm-hidden-size", type=int, default=512)
    parser.add_argument("--final-eval-games", type=int, default=128)
    parser.add_argument("--max-runs", type=int, default=8)
    parser.add_argument("--confirm-candidates", type=int, default=3)
    parser.add_argument("--candidate-total-seeds", type=int, default=3)
    parser.add_argument("--study-name", type=str, default="best_checkpoint_rl")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument(
        "--runtime-config-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--reuse-runtime-config",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-runtime-config",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--best-checkpoint-config-path",
        type=str,
        default="experiments/best_checkpoint.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "experiments/best_checkpoint_rl_tuning/"
            + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        ),
    )
    return parser


def require_optuna():
    """Return the imported Optuna module or raise a command-focused install error.

    Optuna is only needed by this launcher, not by normal training. Keeping it optional avoids
    adding tuner-only dependencies to every install, while the error still tells the user the
    exact `uv run` form that supplies the package for a long tuning job.
    """

    if optuna is None:
        raise RuntimeError(
            "tune_best_checkpoint_rl.py requires optuna. Run it with "
            "`uv run --with optuna python -u scripts/tune_best_checkpoint_rl.py ...`."
        )
    return optuna


def suggest_hyperparameters(trial) -> dict[str, object]:
    """Sample one RL-only hyperparameter set from the Optuna search space.

    The architecture, environment runtime, best checkpoint, and total training budget are fixed
    outside this function. The sampled values only affect how PPO learns: rollout length,
    effective batch shape, optimizer settings, loss coefficients, prioritization, and the KL
    regularization terms used by the existing training loop. The returned nested dictionary keeps
    the same shape as older Puffer-sweep suggestions so command construction and result logging
    stay simple.
    """

    return {
        "rollout": {
            "horizon": trial.suggest_categorical("rollout_horizon", [64, 128, 256]),
            "batch_multiple": trial.suggest_categorical("rollout_batch_multiple", [1, 2, 4]),
            "minibatch_divisor": trial.suggest_categorical(
                "rollout_minibatch_divisor", [1, 2, 4, 8, 16]
            ),
        },
        "train": {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            "update_epochs": trial.suggest_int("update_epochs", 1, 6),
            "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.995),
            "clip_coef": trial.suggest_float("clip_coef", 0.05, 0.4),
            "vf_coef": trial.suggest_float("vf_coef", 0.25, 4.0),
            "vf_clip_coef": trial.suggest_float("vf_clip_coef", 0.05, 1.0),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.25, 3.0),
            "ent_coef": trial.suggest_float("ent_coef", 1e-6, 1e-2, log=True),
            "prio_alpha": trial.suggest_float("prio_alpha", 0.05, 0.99),
            "prio_beta0": trial.suggest_float("prio_beta0", 0.05, 0.99),
        },
        "regularization": {
            "past_kl_coef": trial.suggest_float("past_kl_coef", 1e-4, 1.0, log=True),
            "uniform_kl_base_coef": trial.suggest_float(
                "uniform_kl_base_coef", 1e-4, 0.2, log=True
            ),
            "uniform_kl_power": trial.suggest_float("uniform_kl_power", 0.0, 1.0),
        },
    }


def append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    """Append one JSON record to a history file for later inspection.

    Sweep runs are long enough that terminal output alone is not a reliable source of truth.
    JSON Lines makes the per-trial history easy to diff, grep, or load into a notebook after
    the run completes.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def sample_seed(rng: random.Random) -> int:
    """Draw one fresh training seed for a trial or confirmation rerun.

    The user explicitly asked not to lock a single seed across the whole tuning job. This
    helper centralizes the policy of sampling a new seed every time so both the initial sweep
    and the follow-up confirmation stage behave the same way.
    """

    return rng.randint(1, 2**31 - 1)


def resolve_runtime_vec_config(
    args,
) -> tuple[train_pufferl.VecEnvConfig, train_pufferl.BenchmarkResult | None]:
    """Pick the one fixed vector runtime that every RL trial should reuse.

    RL-only tuning still benefits from Puffer's runtime autotuner because a bad environment
    layout would waste wall-clock time and make each trial unnecessarily slow. The important
    constraint is that we choose the runtime once up front and then freeze it for the entire
    sweep so the search does not accidentally tune vectorization.
    """

    runtime_config_path = (
        None if args.runtime_config_path is None else Path(args.runtime_config_path)
    )
    if (
        args.reuse_runtime_config
        and runtime_config_path is not None
        and runtime_config_path.exists()
    ):
        payload = json.loads(runtime_config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Expected a JSON object in cached runtime config {runtime_config_path}"
            )
        vec_payload = payload.get("vec_config")
        if not isinstance(vec_payload, Mapping):
            raise ValueError(
                f"Cached runtime config {runtime_config_path} is missing vec_config"
            )
        vec_config = train_pufferl.deserialize_vec_config(vec_payload)
        print(f"Loaded cached runtime config: {runtime_config_path}")
        return vec_config, None

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
    """Render one frozen vector config back into `train_pufferl.py` CLI flags.

    The training script already owns vector environment construction, so the tuning driver
    just needs a faithful way to replay the selected runtime layout for every subprocess.
    Returning a flat argument list keeps command assembly simple and avoids duplicating any
    environment creation logic in the sweep script.
    """

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
    """Convert latent sweep parameters into legal PuffeRL rollout sizes.

    The search space uses `batch_multiple` and `minibatch_divisor` because those are easier
    for a sweep algorithm to explore under PuffeRL's divisibility constraints. This helper
    turns those latent values into the actual `train_batch_size`, `bptt_horizon`, and
    `minibatch_size` passed to the trainer.
    """

    rollout = suggestion["rollout"]
    if not isinstance(rollout, Mapping):
        raise TypeError("expected rollout section in sweep suggestion")

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
    """Assemble the exact subprocess command for one RL training trial.

    The command freezes everything outside RL learning itself: vector layout, total step
    budget, best checkpoint target, and evaluation procedure. Only the sweep-controlled
    learning settings vary between trials.
    """

    rollout = resolve_rollout_hyperparameters(suggestion, args.total_agents)
    train = suggestion["train"]
    regularization = suggestion["regularization"]
    if not isinstance(train, Mapping) or not isinstance(regularization, Mapping):
        raise TypeError(
            "expected train and regularization sections in sweep suggestion"
        )

    command = [
        sys.executable,
        str(TRAIN_SCRIPT_PATH),
        "--players-per-team",
        str(args.players_per_team),
        "--device",
        args.device,
        "--rl-alg",
        str(args.rl_alg),
        "--kl-regularization-mode",
        str(args.kl_regularization_mode),
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
        "--use-lstm",
        "--policy-hidden-size",
        str(args.policy_hidden_size),
        "--policy-encoder-size",
        str(args.policy_encoder_size),
        "--lstm-hidden-size",
        str(args.lstm_hidden_size),
        "--no-opponent-phase-max-iterations",
        "0",
        "--past-kl-coef",
        str(float(regularization["past_kl_coef"])),
        "--uniform-kl-base-coef",
        str(float(regularization["uniform_kl_base_coef"])),
        "--uniform-kl-power",
        str(float(regularization["uniform_kl_power"])),
        "--final-best-eval-games",
        str(args.final_eval_games),
        "--fixed-best-checkpoint",
        "--best-checkpoint-eval",
        "--best-checkpoint-config-path",
        args.best_checkpoint_config_path,
        "--checkpoint-interval",
        str(10**9),
        "--no-past-iterate-eval",
        "--no-export-videos",
        "--wandb",
        "--wandb-group",
        "no-warmstart-best-checkpoint-tuning",
        "--run-summary-path",
        str(summary_path),
    ]
    command.extend(render_vec_cli_args(vec_config))
    return command


def trial_sort_key(result: TrialResult) -> tuple[float, float, float]:
    """Rank trials by the objective we care about: winning the head-to-head match.

    Win rate is the primary metric, score difference breaks ties, and shorter runtime wins a
    final tie so we do not keep slower configs when they perform identically on the task.
    """

    return (
        result.objective_win_rate,
        result.objective_score_diff,
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
    """Read one completed subprocess summary and normalize it into a `TrialResult`.

    Successful and failed subprocesses need to share one representation so the sweep can log
    history consistently and still penalize failed trials. Missing or malformed summaries are
    treated as failures with a zero objective, which keeps the search robust when a sampled
    configuration is unstable.
    """

    failed = returncode != 0 or not summary_path.exists()
    runtime_seconds = float("inf") if failed else 0.0
    objective_win_rate = 0.0
    objective_score_diff = float("-inf")
    actual_global_step = 0
    effective_hyperparameters: dict[str, object] = {}

    if not failed:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        objective = payload.get("objective_metrics") or {}
        runtime_seconds = float(payload.get("runtime_seconds", 0.0))
        objective_win_rate = float(objective.get("win_rate", 0.0))
        objective_score_diff = float(objective.get("score_diff", 0.0))
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
        objective_win_rate=objective_win_rate,
        objective_score_diff=objective_score_diff,
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
    """Execute one training subprocess and return its objective metrics.

    The tuning driver intentionally delegates the actual RL run to the normal training
    script so there is only one training implementation to maintain. Each trial gets its own
    directory with a stdout log and a JSON summary file, which makes failures debuggable even
    if the sweep continues running afterward.
    """

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
    rollout = resolve_rollout_hyperparameters(suggestion, args.total_agents)
    print(
        f"[{phase} {trial_index:03d}] seed={seed} "
        f"lr={mapping_float(train_section, 'learning_rate'):.3g} "
        f"horizon={rollout['bptt_horizon']}"
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
        f"win_rate={result.objective_win_rate:.3f} "
        f"score_diff={result.objective_score_diff:.3f} "
        f"runtime={result.runtime_seconds:.1f}s"
    )
    return result


def cast_mapping(value: object) -> Mapping[str, object]:
    """Type-check that a JSON-like object is a mapping before subscripting it.

    The sweep suggestions come back as nested `object` values. Making the runtime check
    explicit keeps the rest of the script simple and prevents a malformed suggestion from
    failing later with a less helpful error.
    """

    if not isinstance(value, Mapping):
        raise TypeError("expected mapping")
    return value


def mapping_float(mapping: Mapping[str, object], key: str) -> float:
    """Read one mapping value as a float with a clear error if the type is wrong.

    Sweep suggestions and JSON summaries are stored as generic objects. This helper keeps
    the required numeric casts explicit and centralizes the error message used when a record
    is missing a numeric field that the tuning loop depends on.
    """

    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected numeric value for {key}")
    return float(value)


def mapping_int(mapping: Mapping[str, object], key: str) -> int:
    """Read one mapping value as an integer for command assembly and reporting.

    The tuning script only stores JSON-like payloads, so integer-valued hyperparameters flow
    through generic mappings. Pulling the cast into one helper makes the intent obvious at
    call sites and keeps the runtime checks consistent.
    """

    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected integer value for {key}")
    return int(value)


def summarize_candidate_runs(runs: Sequence[TrialResult]) -> dict[str, object]:
    """Aggregate multiple fresh-seed runs for one candidate configuration.

    The initial sweep uses one new seed per trial to cover more of the search space. Once we
    have a few strong candidates, we need a seed-averaged view before calling the winner
    “high quality.” This summary captures that aggregate decision in a machine-readable form.
    """

    win_rates = [run.objective_win_rate for run in runs]
    score_diffs = [run.objective_score_diff for run in runs]
    runtimes = [run.runtime_seconds for run in runs]
    return {
        "mean_win_rate": float(np.mean(win_rates)),
        "std_win_rate": float(np.std(win_rates)),
        "mean_score_diff": float(np.mean(score_diffs)),
        "std_score_diff": float(np.std(score_diffs)),
        "mean_runtime_seconds": float(np.mean(runtimes)),
        "runs": [run.to_record() for run in runs],
        "effective_hyperparameters": runs[0].effective_hyperparameters,
        "suggestion": runs[0].suggestion,
    }


def print_top_results(results: Sequence[TrialResult], limit: int = 5) -> None:
    """Print a compact leaderboard for the strongest finished trials so far.

    Long tuning jobs should still give a human-readable signal about progress. This helper
    prints only the top few successful trials so the terminal stays easy to scan.
    """

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
            f"seed={result.seed} win_rate={result.objective_win_rate:.3f} "
            f"score_diff={result.objective_score_diff:.3f} "
            f"batch={hypers.get('train_batch_size')} horizon={hypers.get('bptt_horizon')} "
            f"minibatch={hypers.get('minibatch_size')} lr={hypers.get('learning_rate')}"
        )


def main() -> None:
    """Tune RL hyperparameters against the fixed best checkpoint and confirm the winner.

    The workflow is intentionally two-stage. First Optuna explores a broad set of RL
    hyperparameters, always using a new seed for each trial and always holding the runtime
    layout fixed. Then we rerun the strongest candidates with additional fresh seeds so the
    final recommendation reflects more than one lucky draw.
    """

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

    vec_config, autotune_result = resolve_runtime_vec_config(args)
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
    runtime_config_path = (
        None if args.runtime_config_path is None else Path(args.runtime_config_path)
    )
    if args.save_runtime_config and runtime_config_path is not None:
        train_pufferl.write_json_record(runtime_config_path, runtime_payload)
        print(f"Saved reusable runtime config: {runtime_config_path}")
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

    optuna_module = require_optuna()
    sampler = optuna_module.samplers.TPESampler(seed=args.seed)
    study = optuna_module.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=args.storage is not None,
        sampler=sampler,
    )
    rng = random.Random(time.time_ns())
    all_results: list[TrialResult] = []

    for trial_index in range(1, args.max_runs + 1):
        trial = study.ask()
        suggestion = suggest_hyperparameters(trial)
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
        if result.failed:
            study.tell(trial, state=optuna_module.trial.TrialState.FAIL)
        else:
            trial.set_user_attr("score_diff", result.objective_score_diff)
            trial.set_user_attr("runtime_seconds", result.runtime_seconds)
            trial.set_user_attr("summary_path", str(result.summary_path))
            study.tell(trial, result.objective_win_rate)
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
            f"Candidate {candidate_index}: mean_win_rate={summary['mean_win_rate']:.3f} "
            f"mean_score_diff={summary['mean_score_diff']:.3f} "
            f"seeds={len(candidate_runs)}"
        )

    candidate_summaries.sort(
        key=lambda summary: (
            mapping_float(summary, "mean_win_rate"),
            mapping_float(summary, "mean_score_diff"),
            -mapping_float(summary, "mean_runtime_seconds"),
        ),
        reverse=True,
    )
    train_pufferl.write_json_record(aggregate_path, {"candidates": candidate_summaries})
    best_candidate = candidate_summaries[0]
    train_pufferl.write_json_record(best_path, best_candidate)

    best_hypers = cast_mapping(best_candidate["effective_hyperparameters"])
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
        "past_kl_coef",
        "uniform_kl_base_coef",
        "uniform_kl_power",
        "total_timesteps",
    ):
        print(f"  {key}={best_hypers[key]}")
    print(
        f"Confirmed mean win rate={best_candidate['mean_win_rate']:.3f}, "
        f"mean score diff={best_candidate['mean_score_diff']:.3f}"
    )
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
