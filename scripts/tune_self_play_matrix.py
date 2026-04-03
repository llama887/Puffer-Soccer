"""Launch the full 3 x 2 self-play RL tuning matrix from one entrypoint.

This project now compares three opponent-generation algorithms:

- ordinary self-play
- MARLadona-style league core
- paper-first MARLadona reproduction

and it also wants KL regularization to remain an orthogonal axis. Running all six
combinations by hand would be tedious and error-prone, especially because the user wants
all variants to share the same warm-started self-play phase and the same frozen runtime
configuration discipline. This script orchestrates that matrix by repeatedly invoking the
existing RL tuning driver with the appropriate `--rl-alg` and `--kl-regularization-mode`
flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
TUNE_SCRIPT_PATH = Path(__file__).resolve().with_name("tune_best_checkpoint_rl.py")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI that launches all six RL comparison variants.

    The matrix script intentionally reuses the underlying best-checkpoint RL tuner for the
    actual optimization work. Its own arguments therefore focus on matrix coordination:
    where to store outputs, which variants to include, and which extra tuner flags should be
    forwarded unchanged to every child run.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "experiments/self_play_rl_matrix/"
            + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        ),
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help=(
            "Comma-separated subset of variant labels to run, or `all`. "
            "Labels use the form `<rl_alg>__kl_<on|off>`."
        ),
    )
    return parser


def matrix_variants() -> list[dict[str, str]]:
    """Return the six RL comparison variants in a stable human-readable order.

    Keeping the matrix declaration in one helper makes the orchestration fully explicit and
    avoids accidental drift between the output folder names, the command-line flags, and the
    manifest written at the end of the run.
    """

    variants: list[dict[str, str]] = []
    for rl_alg in ("self_play", "league", "marlodonna"):
        for kl_mode in ("off", "on"):
            variants.append(
                {
                    "label": f"{rl_alg}__kl_{kl_mode}",
                    "rl_alg": rl_alg,
                    "kl_regularization_mode": kl_mode,
                }
            )
    return variants


def selected_variants(selector: str) -> list[dict[str, str]]:
    """Filter the six predefined variants down to the requested subset.

    The default mode runs the entire matrix, but it is still useful to resume a partial
    experiment or rerun one failed variant without editing the script.
    """

    variants = matrix_variants()
    if selector.strip().lower() == "all":
        return variants
    requested = {value.strip() for value in selector.split(",") if value.strip()}
    selected = [variant for variant in variants if variant["label"] in requested]
    if not selected:
        raise ValueError("variants selector did not match any known matrix variant")
    return selected


def build_child_command(
    *,
    variant: dict[str, str],
    output_dir: Path,
    forwarded_args: list[str],
) -> list[str]:
    """Build the exact tuner command for one matrix child run.

    Each child run gets its own output directory and receives the requested RL algorithm and KL
    mode explicitly. All remaining CLI arguments are forwarded unchanged so the matrix shares
    one common runtime budget, runtime layout policy, and sweep breadth.
    """

    return [
        sys.executable,
        str(TUNE_SCRIPT_PATH),
        "--rl-alg",
        variant["rl_alg"],
        "--kl-regularization-mode",
        variant["kl_regularization_mode"],
        "--output-dir",
        str(output_dir / variant["label"]),
        *forwarded_args,
    ]


def main() -> None:
    """Run the six-variant self-play RL tuning matrix and record the launched commands.

    This script is intentionally thin. The underlying tuner already knows how to freeze the
    runtime layout, run warm-started self-play training, and summarize trial results. The
    matrix layer simply ensures that every requested RL-algorithm/KL combination is launched
    with a clean output directory and that the overall experiment leaves behind one manifest
    describing the exact child commands that were executed.
    """

    parser = build_parser()
    args, forwarded = parser.parse_known_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variants": [],
    }

    for variant in selected_variants(args.variants):
        command = build_child_command(
            variant=variant,
            output_dir=output_dir,
            forwarded_args=forwarded,
        )
        print(
            "Launching RL tuning variant: "
            f"{variant['label']} "
            f"(rl_alg={variant['rl_alg']}, "
            f"kl_regularization_mode={variant['kl_regularization_mode']})"
        )
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        manifest_variant = dict(variant)
        manifest_variant["command"] = command
        manifest_variant["returncode"] = int(completed.returncode)
        cast_variants = manifest["variants"]
        assert isinstance(cast_variants, list)
        cast_variants.append(manifest_variant)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    (output_dir / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
