"""Regression tests for the autoresearch Slurm launch script.

The autoresearch loop depends on `sbatch/train_automode.sbatch` to create repo-local
logs and temporary outputs before the containerized training command starts. A recent
failed launch showed that Slurm may execute the batch script from an internal spool
directory such as `/opt/slurm/data/slurmd/sbatch`. In that environment, shell features
like `${BASH_SOURCE[0]}` point at the staged copy rather than the repository checkout,
so deriving the repo root from the script location still lands in an unwritable place.

These tests keep the fix aligned with how Slurm really launches the job. They verify
that the script relies on Slurm's working directory instead of the staged script path,
and that the small debug escape hatch used for the test itself remains available for
future launch debugging.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "sbatch" / "train_automode.sbatch"


def test_train_automode_resolves_repo_root_from_working_directory() -> None:
    """Verify the batch script ignores a bad staged-script path during launch.

    Slurm can stage the submitted script under an internal directory that the job cannot
    write to, which makes `${BASH_SOURCE[0]}` an unreliable way to find the repository
    root. The durable fix is to launch the job with `#SBATCH --chdir=<repo root>` and to
    derive `REPO_ROOT` from `pwd`. This test exercises the debug escape hatch while
    deliberately setting `SLURM_SUBMIT_DIR` to the bad internal path to confirm that the
    shell still reports the real repository root.
    """

    env = dict(os.environ)
    env["SLURM_SUBMIT_DIR"] = "/opt/slurm/data/slurmd/sbatch"
    env["TRAIN_AUTOMODE_PRINT_ROOT_ONLY"] = "1"

    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(REPO_ROOT)


def test_train_automode_uses_standardized_hyperparameter_file() -> None:
    """Verify the batch script opts into the canonical sweep-defaults file explicitly.

    Autoload is enabled by default in the trainer, but the batch script should still make
    the intended source clear in plain text. Keeping this assertion here prevents future
    edits from quietly dropping the standardized path from the production launch command.
    """

    contents = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "#SBATCH --chdir=/scratch/fyy2003/repos/Puffer-Soccer" in contents
    assert 'REPO_ROOT="$(pwd -P)"' in contents
    assert "--hyperparameters-path experiments/autoload_hyperparameters.json" in contents
    assert "--vec-backend auto" in contents
    assert (
        '--no-opponent-phase-min-iterations "$TRAIN_AUTOMODE_NO_OPPONENT_MIN_ITERATIONS"'
        in contents
    )
    assert (
        '--no-opponent-phase-max-iterations "$TRAIN_AUTOMODE_NO_OPPONENT_MAX_ITERATIONS"'
        in contents
    )
    assert (
        '--no-opponent-phase-eval-interval "$TRAIN_AUTOMODE_NO_OPPONENT_EVAL_INTERVAL"'
        in contents
    )
    assert (
        '--no-opponent-phase-goal-rate-threshold "$TRAIN_AUTOMODE_NO_OPPONENT_GOAL_RATE_THRESHOLD"'
        in contents
    )
    assert (
        '--no-opponent-phase-multi-goal-rate-threshold "'
        '$TRAIN_AUTOMODE_NO_OPPONENT_MULTI_GOAL_RATE_THRESHOLD"'
        in contents
    )
    assert (
        '--no-opponent-eval-games "$TRAIN_AUTOMODE_NO_OPPONENT_EVAL_GAMES"'
        in contents
    )
    assert (
        '--no-opponent-map-scale-ladder "$TRAIN_AUTOMODE_NO_OPPONENT_MAP_SCALE_LADDER"'
        in contents
    )
    assert 'TRAIN_AUTOMODE_PPO_ITERATIONS="${TRAIN_AUTOMODE_PPO_ITERATIONS:-100000}"' in contents
    assert (
        'TRAIN_AUTOMODE_NO_OPPONENT_MAX_ITERATIONS="'
        '${TRAIN_AUTOMODE_NO_OPPONENT_MAX_ITERATIONS:-128}"'
        in contents
    )
    assert (
        'TRAIN_AUTOMODE_NO_OPPONENT_GOAL_RATE_THRESHOLD="'
        '${TRAIN_AUTOMODE_NO_OPPONENT_GOAL_RATE_THRESHOLD:-0.90}"'
        in contents
    )
    assert (
        'TRAIN_AUTOMODE_NO_OPPONENT_MULTI_GOAL_RATE_THRESHOLD="'
        '${TRAIN_AUTOMODE_NO_OPPONENT_MULTI_GOAL_RATE_THRESHOLD:-0.0}"'
        in contents
    )
    assert (
        'TRAIN_AUTOMODE_NO_OPPONENT_EVAL_GAMES="'
        '${TRAIN_AUTOMODE_NO_OPPONENT_EVAL_GAMES:-100}"'
        in contents
    )
    assert (
        'TRAIN_AUTOMODE_NO_OPPONENT_MAP_SCALE_LADDER="'
        '${TRAIN_AUTOMODE_NO_OPPONENT_MAP_SCALE_LADDER:-0.2,0.4,0.6,0.8,1.0}"'
        in contents
    )
    assert (
        'UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-'
        '$SCRATCH_BASE/.venvs/puffer-soccer-${SLURM_JOB_ID:-local}}"'
        in contents
    )
