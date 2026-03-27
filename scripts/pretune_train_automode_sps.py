"""Pretune and persist the highest-SPS vecenv layout for the automode Slurm job.

This script is meant to be launched once per fixed cluster architecture, ideally with
`srun`, before running long self-play jobs through `sbatch/train_automode.sbatch`.
It reuses the repo's existing vecenv autotuner, records the winning layout in the same
standardized autoload JSON that the training script already consumes, and preserves any
existing PPO defaults already stored in that file.

The operational goal is simple: benchmark the machine once, save the result, and then let
every later training run reuse that vecenv layout automatically without editing the batch
script or paying the autotune startup cost again.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import socket
import time

import train_pufferl
from puffer_soccer.autotune import autotune_vecenv, format_benchmark_result


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI for one-shot SPS pretuning on the current machine.

    The tuning surface here is intentionally narrow because the user already fixed the
    hardware shape in the Slurm script. We only expose the environment properties that
    change SPS materially for the vectorizer search, together with the output path so the
    same command can target either the canonical autoload file or an experiment-local copy.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=train_pufferl.STANDARD_HYPERPARAMETERS_PATH,
    )
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument(
        "--vec-backend",
        type=str,
        default="auto",
        choices=["native", "multiprocessing", "auto"],
    )
    parser.add_argument("--autotune-seconds", type=float, default=1.0)
    parser.add_argument("--autotune-max-num-envs", type=int, default=None)
    parser.add_argument("--autotune-max-num-shards", type=int, default=None)
    parser.add_argument(
        "--source-label",
        type=str,
        default="pretuned_train_automode_vecenv",
    )
    return parser


def load_existing_autoload_record(path: Path) -> dict[str, object]:
    """Load the existing standardized autoload record, returning an empty one when absent.

    The pretune step should be additive. Teams may already rely on the autoload file for
    PPO defaults from a previous sweep, so this helper preserves that state and only updates
    the vecenv-specific portion. Returning a mutable plain dictionary makes the later merge
    logic straightforward while still validating that any on-disk payload is JSON-object
    shaped.
    """

    payload = train_pufferl.read_json_record(path)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected mapping in standardized autoload file: {path}")
    return dict(payload)


def update_autoload_record(
    existing: Mapping[str, object],
    *,
    vecenv_defaults: Mapping[str, object],
    vecenv_benchmark: Mapping[str, object],
    selection_reason: str,
    source_label: str,
) -> dict[str, object]:
    """Merge a new vecenv benchmark winner into a standardized autoload JSON record.

    The existing record may already carry train defaults, rollout scaling metadata, and
    provenance about earlier hyperparameter sweeps. This helper updates only the pieces
    related to SPS pretuning while leaving those training defaults untouched. That keeps the
    user's established workflow intact: one shared JSON file can now carry both PPO defaults
    and the hardware-specific vecenv layout for the automode batch job.

    The extra `vecenv_tuning` block is stored alongside the defaults because long-running
    RL jobs benefit from simple postmortem evidence. When someone later asks why a certain
    layout was chosen, the saved hostname, timestamp, and autotuner selection reason are
    already in the same file.
    """

    merged = dict(existing)
    merged["format_version"] = 1
    merged["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    merged["vecenv_defaults"] = train_pufferl.standardized_vecenv_defaults(
        vecenv_defaults
    )
    merged["vecenv_benchmark"] = dict(vecenv_benchmark)
    merged["vecenv_tuning"] = {
        "selection_reason": selection_reason,
        "source_label": source_label,
        "hostname": socket.gethostname(),
        "saved_at_utc": merged["generated_at_utc"],
    }
    return merged


def main() -> None:
    """Run the vecenv autotuner and persist the winning layout for later Slurm jobs.

    This script intentionally benchmarks only the rollout backend, not the PPO optimizer.
    The resulting file is therefore best verified by comparing later training runs' logged
    SPS against the saved benchmark metadata and by confirming that `train_automode.sbatch`
    now prints the same vecenv layout without rerunning autotune on startup.
    """

    args = build_parser().parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    outcome = autotune_vecenv(
        players_per_team=args.players_per_team,
        seconds=args.autotune_seconds,
        action_mode="discrete",
        backend=args.vec_backend,
        max_num_envs=args.autotune_max_num_envs,
        max_num_shards=args.autotune_max_num_shards,
        reporter=print,
    )
    best = outcome.best
    vecenv_defaults = train_pufferl.vecenv_defaults_from_benchmark(best)
    benchmark = train_pufferl.benchmark_record(best)
    existing = load_existing_autoload_record(output_path)
    merged = update_autoload_record(
        existing,
        vecenv_defaults=vecenv_defaults,
        vecenv_benchmark=benchmark,
        selection_reason=outcome.selection_reason,
        source_label=args.source_label,
    )
    train_pufferl.write_json_record(output_path, merged)

    print("Saved pretuned vecenv defaults")
    print(f"  output_path={output_path}")
    print(f"  backend={vecenv_defaults['vec_backend']}")
    print(f"  num_envs={vecenv_defaults['num_envs']}")
    print(f"  vec_num_shards={vecenv_defaults.get('vec_num_shards')}")
    print(f"  vec_batch_size={vecenv_defaults.get('vec_batch_size')}")
    print(f"  benchmark={format_benchmark_result(best)}")
    print(f"  selection_reason={outcome.selection_reason}")


if __name__ == "__main__":
    main()
