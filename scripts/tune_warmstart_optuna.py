"""Tune the no-opponent warm-start with Optuna and Slurm `srun` trials.

This helper focuses on one narrow question: which warm-start settings let the policy clear the
no-opponent scoring gate reliably enough to hand off to self-play? The main training script
already knows how to run the warm-start and emit machine-readable summaries, so this tuner
reuses that path instead of creating a second training implementation.

The key design choice is that every Optuna trial launches a fresh `srun` subprocess. That
keeps CUDA state, random seeds, and trainer lifetime isolated per trial, which matters much
more here than raw tuning throughput. Warm-start failures are also informative, so this script
parses the final printed no-opponent metrics from the trial log even when the training process
exits with a non-zero status because it did not meet the required thresholds.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import subprocess
import time

import train_pufferl

try:
    import optuna  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised in real CLI use
    optuna = None


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT_PATH = Path(__file__).resolve().with_name("train_pufferl.py")
DEFAULT_HYPERPARAMETERS_PATH = REPO_ROOT / "experiments" / "autoload_hyperparameters.json"

FINAL_EVAL_PATTERN = re.compile(
    r"Final no-opponent eval "
    r"\(epoch=(?P<epoch>\d+), games=(?P<games>\d+)\): "
    r"goal_rate=(?P<goal_rate>[0-9.]+), "
    r"multi_goal_rate=(?P<multi_goal_rate>[0-9.]+), "
    r"mean_goals_scored=(?P<mean_goals_scored>[0-9.]+), "
    r"own_goal_rate=(?P<own_goal_rate>[0-9.]+), "
    r"mean_first_goal_step=(?P<mean_first_goal_step>[0-9.]+)"
)
INTERMEDIATE_EVAL_PATTERN = re.compile(
    r"No-opponent eval "
    r"\(epoch=(?P<epoch>\d+), games=(?P<games>\d+)\): "
    r"train_scale=(?P<train_scale>[0-9.]+), "
    r"goal_rate=(?P<goal_rate>[0-9.]+), "
    r"multi_goal_rate=(?P<multi_goal_rate>[0-9.]+), "
    r"mean_goals_scored=(?P<mean_goals_scored>[0-9.]+), "
    r"own_goal_rate=(?P<own_goal_rate>[0-9.]+), "
    r"mean_first_goal_step=(?P<mean_first_goal_step>[0-9.]+)"
)


@dataclass(frozen=True)
class WarmStartMetrics:
    """Store the scoring metrics that decide whether warm-start solved its drill.

    Keeping these values in one small immutable record makes it easier to rank trials,
    serialize results, and compare successful and unsuccessful settings with the same code.
    The fields mirror the metrics that `train_pufferl.py` already prints and writes into its
    run summary.
    """

    epoch: int
    games: int
    goal_rate: float
    multi_goal_rate: float
    mean_goals_scored: float
    own_goal_rate: float
    mean_first_goal_step: float

    def completion_reached(
        self,
        *,
        goal_rate_threshold: float,
        multi_goal_rate_threshold: float,
    ) -> bool:
        """Return whether the metrics satisfy the configured warm-start gate."""

        return (
            self.goal_rate >= goal_rate_threshold
            and self.multi_goal_rate >= multi_goal_rate_threshold
        )


@dataclass(frozen=True)
class TrialArtifacts:
    """Describe the files and metadata produced by one Optuna trial subprocess.

    The tuning loop needs to retain the exact command, captured log, and optional summary file
    for each trial so the best recipe can be inspected and rerun by hand after the study
    finishes. This record keeps those links together.
    """

    trial_number: int
    seed: int
    trial_dir: Path
    log_path: Path
    summary_path: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class TrialOutcome:
    """Store the normalized result from one warm-start trial.

    Trials can succeed cleanly, fail the scoring gate, or crash before finishing. This record
    keeps enough detail to distinguish those cases while still giving Optuna one scalar score
    to optimize.
    """

    artifacts: TrialArtifacts
    returncode: int
    runtime_seconds: float
    metrics: WarmStartMetrics | None
    score: float
    failed: bool
    effective_hyperparameters: dict[str, object]

    def to_record(self) -> dict[str, object]:
        """Convert the outcome into a plain JSON-ready mapping for history logs."""

        return {
            "trial_number": self.artifacts.trial_number,
            "seed": self.artifacts.seed,
            "trial_dir": str(self.artifacts.trial_dir),
            "log_path": str(self.artifacts.log_path),
            "summary_path": str(self.artifacts.summary_path),
            "command": list(self.artifacts.command),
            "returncode": self.returncode,
            "runtime_seconds": self.runtime_seconds,
            "score": self.score,
            "failed": self.failed,
            "metrics": None if self.metrics is None else self.metrics.__dict__,
            "effective_hyperparameters": self.effective_hyperparameters,
        }


def append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    """Append one JSON record to a log file for later inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def load_recorded_trial_record(
    history_path: Path, *, trial_number: int
) -> dict[str, object]:
    """Load one previously appended trial record from the JSONL history file.

    The Optuna study keeps a compact summary inside the study object, but the richer data we
    need after optimization lives in the history log written by `append_jsonl`: exact command,
    effective hyperparameters, and parsed warm-start metrics. Looking the winner up in that log
    lets the finalization step reuse the already-recorded subprocess result instead of trying to
    reconstruct it from Optuna's lighter-weight `FrozenTrial` snapshot.

    Failing loudly here is intentional. If the winning trial cannot be found in the history
    file, the study output would be incomplete and we would rather stop than silently write an
    unusable autoload file.
    """

    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if int(payload.get("trial_number", -1)) == trial_number:
            if not isinstance(payload, Mapping):
                break
            return dict(payload)
    raise ValueError(
        f"could not find trial_number={trial_number} in warm-start history {history_path}"
    )


def mapping_int(mapping: Mapping[str, object], key: str) -> int:
    """Read one mapping value as an integer with a clear error on bad types.

    The tuner moves JSON-like dictionaries between Optuna, subprocess summaries, and the local
    file format. Centralizing the numeric conversion keeps the command construction code simple
    and makes type expectations explicit.
    """

    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected integer-like value for {key}")
    return int(value)


def mapping_float(mapping: Mapping[str, object], key: str) -> float:
    """Read one mapping value as a float with a clear error on bad types."""

    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected numeric value for {key}")
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI used to tune the warm-start-only training budget and PPO settings.

    The tuner keeps the runtime layout fixed so Optuna spends its budget on learning behavior
    rather than on environment scheduling. The output directory stores both the Optuna study
    summary and all raw trial logs.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyperparameters-path",
        type=Path,
        default=DEFAULT_HYPERPARAMETERS_PATH,
    )
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--study-name", type=str, default="warmstart_optuna")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-max-steps", type=int, default=600)
    parser.add_argument("--goal-rate-threshold", type=float, default=0.80)
    parser.add_argument("--multi-goal-rate-threshold", type=float, default=0.0)
    parser.add_argument(
        "--map-scale-ladder",
        type=str,
        default="0.2,0.4,0.6,0.8,1.0",
    )
    parser.add_argument("--warmstart-min-iterations", type=int, default=32)
    parser.add_argument("--job-time", type=str, default="00:20:00")
    parser.add_argument("--cpus-per-task", type=int, default=16)
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument(
        "--account",
        type=str,
        default="torch_pr_45_tandon_advanced",
    )
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "warmstart_optuna",
    )
    return parser


def read_hyperparameter_defaults(path: Path) -> dict[str, object]:
    """Load the baseline recipe the warm-start study should search around.

    The preferred starting point is the standardized autoload JSON produced by earlier
    tuning runs because it reflects the most recent confirmed warm-start recipe. On a fresh
    checkout, though, that file may not exist yet. Falling back to the training script's own
    built-in defaults keeps the tuner usable in that first-run case and avoids a brittle
    bootstrap dependency between "having tuned once already" and "being allowed to tune now."

    Reconstructing the fallback through `train_pufferl` is important because that module owns
    the canonical parser defaults. If the repo later changes its default ladder or PPO values,
    this tuner automatically inherits the new baseline instead of silently searching around an
    outdated hard-coded recipe.
    """

    if path.exists():
        defaults = train_pufferl.standardized_hyperparameter_defaults(
            train_pufferl.load_standardized_hyperparameter_defaults(path)
        )
        if defaults:
            return defaults
        print(
            "Warm-start tuner found no reusable train defaults in "
            f"{path}; falling back to built-in parser defaults."
        )
    else:
        print(
            "Warm-start tuner baseline file does not exist: "
            f"{path}. Falling back to built-in parser defaults."
        )

    return train_pufferl.standardized_hyperparameter_defaults(
        train_pufferl.base_training_arg_defaults()
    )


def load_preserved_vecenv_payload(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Read vecenv metadata that should survive warm-start retuning.

    Warm-start tuning changes PPO and curriculum defaults, but it does not answer the separate
    question of which rollout backend layout is fastest on a given machine. If a standardized
    autoload file already contains pretuned `vecenv_defaults` and benchmark metadata, writing
    the new warm-start winner should preserve those blocks instead of erasing them.

    The function returns plain dictionaries because the result is passed directly into
    `train_pufferl.write_standardized_hyperparameters`, which expects JSON-like mappings and
    may be called for both the canonical repo-wide autoload file and the experiment-local copy.
    """

    payload = train_pufferl.read_json_record(path)
    if not isinstance(payload, Mapping):
        return {}, {}

    vecenv_defaults_raw = payload.get("vecenv_defaults")
    if isinstance(vecenv_defaults_raw, Mapping):
        vecenv_defaults = train_pufferl.standardized_vecenv_defaults(vecenv_defaults_raw)
    else:
        vecenv_defaults = train_pufferl.standardized_vecenv_defaults(payload)

    vecenv_benchmark_raw = payload.get("vecenv_benchmark")
    vecenv_benchmark = (
        dict(vecenv_benchmark_raw) if isinstance(vecenv_benchmark_raw, Mapping) else {}
    )
    return vecenv_defaults, vecenv_benchmark


def persist_best_hyperparameters(
    *,
    output_dir: Path,
    baseline_hyperparameters_path: Path,
    source_path: Path,
    effective_hyperparameters: Mapping[str, object],
) -> tuple[Path, Path]:
    """Write the winning warm-start recipe into reusable standardized autoload files.

    The Optuna study itself lives under an experiment directory, but the point of the study is
    to make the next real training job easier. This helper therefore writes two copies of the
    winner: one beside the tuning artifacts for inspection and one at the repo's canonical
    autoload path so normal training picks it up automatically on the next launch.

    Preserving vecenv metadata matters because rollout pretuning and warm-start tuning solve
    different problems. The saved file should therefore combine the new PPO and curriculum
    defaults with any existing vecenv defaults and SPS benchmark evidence already stored at the
    target path or, when the target does not exist yet, at the baseline path used for the study.
    """

    local_autoload_path = output_dir / "autoload_hyperparameters.json"
    canonical_autoload_path = train_pufferl.STANDARD_HYPERPARAMETERS_PATH

    for target_path in (local_autoload_path, canonical_autoload_path):
        preserved_source = target_path if target_path.exists() else baseline_hyperparameters_path
        vecenv_defaults, vecenv_benchmark = load_preserved_vecenv_payload(preserved_source)
        train_pufferl.write_standardized_hyperparameters(
            path=target_path,
            effective_hyperparameters=effective_hyperparameters,
            source_path=source_path,
            source_label="best_no_opponent_warmstart_optuna_result",
            vecenv_defaults=vecenv_defaults,
            vecenv_benchmark=vecenv_benchmark,
        )

    return local_autoload_path, canonical_autoload_path


def sample_hyperparameters(trial, baseline: Mapping[str, object]) -> dict[str, object]:
    """Ask Optuna for one warm-start candidate near the current baseline recipe.

    The stage ladder is now part of the main training recipe rather than a continuous curve
    that Optuna needs to rediscover. That lets the study stay focused on the PPO settings
    and warm-start budget that decide whether the fixed ladder can be cleared reliably.
    """

    horizon = trial.suggest_categorical("bptt_horizon", [32, 64, 96, 128])
    batch_multiple = trial.suggest_categorical("batch_multiple", [1, 2, 3, 4])
    minibatch_divisor = trial.suggest_categorical("minibatch_divisor", [2, 4, 8])
    train_batch_size = 10 * 8 * horizon * batch_multiple
    requested_minibatch = max(horizon, train_batch_size // minibatch_divisor)
    minibatch_size = max(horizon, requested_minibatch - (requested_minibatch % horizon))

    return {
        "ppo_iterations": trial.suggest_categorical(
            "ppo_iterations", [64, 96, 128, 160, 192]
        ),
        "train_batch_size": train_batch_size,
        "bptt_horizon": horizon,
        "minibatch_size": minibatch_size,
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.5 * mapping_float(baseline, "learning_rate"),
            2.0 * mapping_float(baseline, "learning_rate"),
            log=True,
        ),
        "gamma": trial.suggest_float("gamma", 0.99, 0.9995),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.93, 0.99),
        "update_epochs": trial.suggest_int("update_epochs", 3, 6),
        "clip_coef": trial.suggest_float("clip_coef", 0.18, 0.30),
        "vf_coef": trial.suggest_float("vf_coef", 0.6, 1.5),
        "vf_clip_coef": trial.suggest_float("vf_clip_coef", 0.10, 0.30),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.4, 1.2),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-5, log=True),
        "prio_alpha": trial.suggest_float("prio_alpha", 0.7, 0.99),
        "prio_beta0": trial.suggest_float("prio_beta0", 0.4, 0.95),
        "no_opponent_map_scale_ladder": str(
            baseline.get("no_opponent_map_scale_ladder", "0.2,0.4,0.6,0.8,1.0")
        ),
    }


def build_training_command(
    *,
    args,
    hyperparameters: Mapping[str, object],
    seed: int,
    summary_path: Path,
) -> list[str]:
    """Render one warm-start-only training run into a subprocess command.

    The command deliberately sets `ppo_iterations` equal to the warm-start max iterations so
    the run ends immediately after the no-opponent phase. That keeps each Slurm trial focused
    on the exact gating problem we are trying to solve.
    """

    ppo_iterations = mapping_int(hyperparameters, "ppo_iterations")
    return [
        "uv",
        "run",
        "python",
        str(TRAIN_SCRIPT_PATH),
        "--hyperparameters-path",
        str(args.hyperparameters_path),
        "--players-per-team",
        str(args.players_per_team),
        "--vec-backend",
        "native",
        "--num-envs",
        str(args.num_envs),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--ppo-iterations",
        str(ppo_iterations),
        "--no-opponent-phase-min-iterations",
        str(min(args.warmstart_min_iterations, ppo_iterations)),
        "--no-opponent-phase-max-iterations",
        str(ppo_iterations),
        "--no-opponent-phase-eval-interval",
        "8",
        "--no-opponent-phase-goal-rate-threshold",
        str(args.goal_rate_threshold),
        "--no-opponent-phase-multi-goal-rate-threshold",
        str(args.multi_goal_rate_threshold),
        "--train-batch-size",
        str(mapping_int(hyperparameters, "train_batch_size")),
        "--bptt-horizon",
        str(mapping_int(hyperparameters, "bptt_horizon")),
        "--minibatch-size",
        str(mapping_int(hyperparameters, "minibatch_size")),
        "--learning-rate",
        str(mapping_float(hyperparameters, "learning_rate")),
        "--gamma",
        str(mapping_float(hyperparameters, "gamma")),
        "--gae-lambda",
        str(mapping_float(hyperparameters, "gae_lambda")),
        "--update-epochs",
        str(mapping_int(hyperparameters, "update_epochs")),
        "--clip-coef",
        str(mapping_float(hyperparameters, "clip_coef")),
        "--vf-coef",
        str(mapping_float(hyperparameters, "vf_coef")),
        "--vf-clip-coef",
        str(mapping_float(hyperparameters, "vf_clip_coef")),
        "--max-grad-norm",
        str(mapping_float(hyperparameters, "max_grad_norm")),
        "--ent-coef",
        str(mapping_float(hyperparameters, "ent_coef")),
        "--prio-alpha",
        str(mapping_float(hyperparameters, "prio_alpha")),
        "--prio-beta0",
        str(mapping_float(hyperparameters, "prio_beta0")),
        "--no-opponent-map-scale-ladder",
        str(hyperparameters.get("no_opponent_map_scale_ladder", args.map_scale_ladder)),
        "--no-opponent-eval-games",
        str(args.eval_games),
        "--no-opponent-eval-max-steps",
        str(args.eval_max_steps),
        "--final-best-eval-games",
        "0",
        "--checkpoint-interval",
        str(10**9),
        "--no-past-iterate-eval",
        "--no-export-videos",
        "--no-wandb",
        "--run-summary-path",
        str(summary_path),
    ]


def build_srun_command(*, args, command: Sequence[str]) -> list[str]:
    """Wrap one training command in the Slurm `srun` allocation requested by the user.

    Running through `srun` matches the real deployment environment and gives each trial clean
    access to the requested compute shape, a fixed CPU budget, and a bounded wall-clock time.

    GPU-backed warm-start tuning remains the default because it matches the real training job,
    but a CPU-only fallback is still valuable when the cluster is temporarily out of GPU quota
    or when a quick smoke test just needs to verify that the Slurm-wrapped tuning path launches
    correctly. Tying the `--gres=gpu:1` request to the chosen device keeps that escape hatch
    available without introducing a second CLI switch for the same idea.
    """

    srun_command = [
        "srun",
        "--account",
        args.account,
        "--cpus-per-task",
        str(args.cpus_per_task),
        "--mem",
        args.mem,
        "--time",
        args.job_time,
        "--chdir",
        str(REPO_ROOT),
        *command,
    ]
    if args.device != "cpu":
        srun_command[4:4] = ["--gres=gpu:1"]
    if args.partition:
        srun_command[4:4] = ["--partition", args.partition]
    return srun_command


def extract_metrics_from_log(log_text: str) -> WarmStartMetrics | None:
    """Recover the latest useful no-opponent metrics from one trial log.

    The trainer now uses the ordinary periodic no-opponent evaluation as the only warm-start
    gate, so newer logs may never print a separate `Final no-opponent eval` line. Older logs
    can still contain that explicit final line from the previous flow. This parser accepts
    both formats so Optuna can keep learning from fresh runs and from earlier archived logs.
    """

    final_match = None
    for final_match in FINAL_EVAL_PATTERN.finditer(log_text):
        pass
    if final_match is not None:
        groups = final_match.groupdict()
        return WarmStartMetrics(
            epoch=int(groups["epoch"]),
            games=int(groups["games"]),
            goal_rate=float(groups["goal_rate"]),
            multi_goal_rate=float(groups["multi_goal_rate"]),
            mean_goals_scored=float(groups["mean_goals_scored"]),
            own_goal_rate=float(groups["own_goal_rate"]),
            mean_first_goal_step=float(groups["mean_first_goal_step"]),
        )

    last_intermediate_match = None
    for last_intermediate_match in INTERMEDIATE_EVAL_PATTERN.finditer(log_text):
        pass
    if last_intermediate_match is None:
        return None

    groups = last_intermediate_match.groupdict()
    return WarmStartMetrics(
        epoch=int(groups["epoch"]),
        games=int(groups["games"]),
        goal_rate=float(groups["goal_rate"]),
        multi_goal_rate=float(groups["multi_goal_rate"]),
        mean_goals_scored=float(groups["mean_goals_scored"]),
        own_goal_rate=float(groups["own_goal_rate"]),
        mean_first_goal_step=float(groups["mean_first_goal_step"]),
    )


def load_trial_outcome(
    *,
    args,
    artifacts: TrialArtifacts,
    returncode: int,
    hyperparameters: Mapping[str, object],
) -> TrialOutcome:
    """Load summary data and log-derived metrics for one finished trial subprocess.

    Successful runs usually provide both a summary file and a zero return code. Failed runs may
    still provide useful metrics via the log. This loader merges both sources into one result
    object so the optimization loop can compare them fairly.
    """

    runtime_seconds = float("inf")
    metrics = None
    effective_hyperparameters: dict[str, object] = dict(hyperparameters)

    if artifacts.summary_path.exists():
        payload = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
        runtime_seconds = float(payload.get("runtime_seconds", float("inf")))
        summary_metrics = payload.get("objective_metrics")
        if isinstance(summary_metrics, Mapping):
            metrics = WarmStartMetrics(
                epoch=int(payload.get("epoch", 0)),
                games=int(summary_metrics.get("games", 0)),
                goal_rate=float(summary_metrics.get("goal_rate", 0.0)),
                multi_goal_rate=float(summary_metrics.get("multi_goal_rate", 0.0)),
                mean_goals_scored=float(summary_metrics.get("mean_goals_scored", 0.0)),
                own_goal_rate=float(summary_metrics.get("own_goal_rate", 1.0)),
                mean_first_goal_step=float(
                    summary_metrics.get("mean_first_goal_step", 10**9)
                ),
            )
        effective = payload.get("effective_hyperparameters")
        if isinstance(effective, Mapping):
            effective_hyperparameters = dict(effective)

    log_text = artifacts.log_path.read_text(encoding="utf-8")
    if metrics is None:
        metrics = extract_metrics_from_log(log_text)

    score = -1e9
    if metrics is not None:
        success_bonus = (
            100.0
            if metrics.completion_reached(
                goal_rate_threshold=args.goal_rate_threshold,
                multi_goal_rate_threshold=args.multi_goal_rate_threshold,
            )
            else 0.0
        )
        score = (
            success_bonus
            + 20.0 * metrics.multi_goal_rate
            + 6.0 * metrics.goal_rate
            + 4.0 * metrics.mean_goals_scored
            - 3.0 * metrics.own_goal_rate
            - 0.01 * metrics.mean_first_goal_step
            - 0.02 * mapping_float(hyperparameters, "ppo_iterations")
        )

    return TrialOutcome(
        artifacts=artifacts,
        returncode=returncode,
        runtime_seconds=runtime_seconds,
        metrics=metrics,
        score=score,
        failed=returncode != 0,
        effective_hyperparameters=effective_hyperparameters,
    )

# pylint: disable=too-many-locals
def run_trial_subprocess(
    *,
    args,
    trial,
    hyperparameters: Mapping[str, object],
    seed: int,
    output_dir: Path,
) -> TrialOutcome:
    """Launch one Slurm warm-start trial and return its normalized outcome.

    Every trial gets its own directory so the search leaves behind a complete paper trail:
    raw logs, summary JSON, and the exact command that was executed. That makes follow-up
    debugging much easier once Optuna surfaces a promising recipe.
    """

    trial_dir = output_dir / f"trial_{trial.number:03d}_seed_{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    summary_path = trial_dir / "summary.json"
    log_path = trial_dir / "output.log"
    train_command = build_training_command(
        args=args,
        hyperparameters=hyperparameters,
        seed=seed,
        summary_path=summary_path,
    )
    srun_command = build_srun_command(args=args, command=train_command)
    artifacts = TrialArtifacts(
        trial_number=trial.number,
        seed=seed,
        trial_dir=trial_dir,
        log_path=log_path,
        summary_path=summary_path,
        command=tuple(srun_command),
    )

    start_time = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            srun_command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    outcome = load_trial_outcome(
        args=args,
        artifacts=artifacts,
        returncode=completed.returncode,
        hyperparameters=hyperparameters,
    )
    measured_runtime = max(0.0, time.time() - start_time)
    if outcome.runtime_seconds == float("inf"):
        outcome = TrialOutcome(
            artifacts=outcome.artifacts,
            returncode=outcome.returncode,
            runtime_seconds=measured_runtime,
            metrics=outcome.metrics,
            score=outcome.score,
            failed=outcome.failed,
            effective_hyperparameters=outcome.effective_hyperparameters,
        )
    return outcome


# pylint: enable=too-many-locals


def require_optuna():
    """Return the imported Optuna module or raise a clear setup error.

    This script is meant to be run with `uv run --with optuna`, so a missing import is a user
    environment issue rather than an internal code path. Keeping that check in one helper makes
    the setup requirement obvious and keeps `main` smaller.
    """

    if optuna is None:
        raise RuntimeError(
            "tune_warmstart_optuna.py requires optuna. Run it with `uv run --with optuna`."
        )
    return optuna


def main() -> None:  # pylint: disable=too-many-locals
    """Run the Optuna study, then save the best warm-start recipe and trial history.

    The study is intentionally small and practical. Its purpose is to find a warm-start recipe
    that actually clears the no-opponent scoring gate on the current cluster setup, not to
    publish a fully converged hyperparameter benchmark.
    """

    parser = build_parser()
    args = parser.parse_args()

    optuna_module = require_optuna()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "trial_history.jsonl"
    best_path = args.output_dir / "best_result.json"
    baseline = read_hyperparameter_defaults(args.hyperparameters_path)
    rng = random.Random(args.seed)

    sampler = optuna_module.samplers.TPESampler(seed=args.seed)
    study = optuna_module.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
    )

    def objective(trial) -> float:
        """Evaluate one Optuna suggestion by launching a warm-start-only Slurm job."""

        seed = rng.randint(1, 2**31 - 1)
        hyperparameters = sample_hyperparameters(trial, baseline)
        outcome = run_trial_subprocess(
            args=args,
            trial=trial,
            hyperparameters=hyperparameters,
            seed=seed,
            output_dir=args.output_dir,
        )
        if outcome.metrics is not None:
            trial.set_user_attr("goal_rate", outcome.metrics.goal_rate)
            trial.set_user_attr("multi_goal_rate", outcome.metrics.multi_goal_rate)
            trial.set_user_attr(
                "mean_goals_scored", outcome.metrics.mean_goals_scored
            )
            trial.set_user_attr(
                "mean_first_goal_step", outcome.metrics.mean_first_goal_step
            )
            trial.set_user_attr("own_goal_rate", outcome.metrics.own_goal_rate)
        trial.set_user_attr("returncode", outcome.returncode)
        trial.set_user_attr("failed", outcome.failed)
        trial.set_user_attr("log_path", str(outcome.artifacts.log_path))
        trial.set_user_attr("summary_path", str(outcome.artifacts.summary_path))
        append_jsonl(history_path, outcome.to_record())
        metrics_label = (
            "none"
            if outcome.metrics is None
            else (
                f"goal_rate={outcome.metrics.goal_rate:.3f} "
                f"multi_goal_rate={outcome.metrics.multi_goal_rate:.3f} "
                f"mean_goals={outcome.metrics.mean_goals_scored:.3f}"
            )
        )
        print(
            f"[trial {trial.number:03d}] returncode={outcome.returncode} "
            f"score={outcome.score:.3f} budget={hyperparameters['ppo_iterations']} "
            f"lr={mapping_float(hyperparameters, 'learning_rate'):.5f} {metrics_label}"
        )
        return outcome.score

    study.optimize(objective, n_trials=args.trials)

    best_trial = study.best_trial
    best_trial_record = load_recorded_trial_record(
        history_path, trial_number=int(best_trial.number)
    )
    best_effective_raw = best_trial_record.get("effective_hyperparameters")
    if not isinstance(best_effective_raw, Mapping):
        raise TypeError(
            "warm-start trial history is missing effective_hyperparameters for "
            f"trial_number={best_trial.number}"
        )
    best_effective_hyperparameters = dict(best_effective_raw)
    best_payload = {
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "user_attrs": dict(best_trial.user_attrs),
        "best_effective_hyperparameters": best_effective_hyperparameters,
        "baseline_hyperparameters_path": str(args.hyperparameters_path),
        "goal_rate_threshold": args.goal_rate_threshold,
        "multi_goal_rate_threshold": args.multi_goal_rate_threshold,
        "trials": args.trials,
    }
    best_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True), encoding="utf-8")
    local_autoload_path, canonical_autoload_path = persist_best_hyperparameters(
        output_dir=args.output_dir,
        baseline_hyperparameters_path=args.hyperparameters_path,
        source_path=best_path,
        effective_hyperparameters=best_effective_hyperparameters,
    )
    print(json.dumps(best_payload, indent=2, sort_keys=True))
    print(f"Wrote best result to {best_path}")
    print(f"Wrote local autoload file to {local_autoload_path}")
    print(f"Wrote canonical autoload file to {canonical_autoload_path}")


if __name__ == "__main__":
    main()
