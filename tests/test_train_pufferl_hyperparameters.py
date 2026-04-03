"""Regression tests for training hyperparameter autoload and field-curriculum setup.

These tests protect the two new convenience behaviors added for long training jobs:
autoloading sweep-selected defaults from a standardized file, and forcing the auto vector
backend through native benchmarking whenever map scaling is active. Both behaviors are easy
to break silently because they happen before the main training loop starts.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import cast

import numpy as np

from puffer_soccer.autotune import AutotuneOutcome, BenchmarkResult


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_pufferl


def make_autotune_outcome(backend: str = "native") -> AutotuneOutcome:
    """Build a minimal autotune result for parser-level tests.

    The training-side autotune adapter only needs one successful benchmark record so it can
    convert that result into a `VecEnvConfig`. Using a tiny helper keeps the tests focused
    on backend selection rather than on the full benchmark search machinery.
    """

    result = BenchmarkResult(
        backend=backend,
        shard_num_envs=8,
        num_shards=1,
        batch_size=None,
        players_per_team=3,
        action_mode="discrete",
        sps=1234,
        cpu_avg=98.0,
        cpu_peak=99.0,
    )
    return AutotuneOutcome(
        best=result,
        best_saturated=result,
        all_results=(result,),
        selection_reason="test",
    )


def test_parse_training_args_autoloads_standardized_defaults(tmp_path: Path) -> None:
    """Verify the standardized file changes parser defaults before the main parse.

    The job-launch path should stay short, so the sweep-written file needs to affect the
    same parser defaults a hand-written CLI would have changed. This test uses a small fake
    standardized record and confirms the final parsed namespace exposes those values.
    """

    hyperparameter_path = tmp_path / "autoload.json"
    hyperparameter_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "train_defaults": {
                    "learning_rate": 0.0123,
                    "no_opponent_map_scale_start": 0.35,
                    "no_opponent_map_scale_end": 1.0,
                },
                "rollout_defaults": {
                    "source_num_agents": 192,
                    "batch_multiple": 2,
                    "minibatch_divisor": 4,
                },
            }
        ),
        encoding="utf-8",
    )

    args = train_pufferl.parse_training_args(
        ["--hyperparameters-path", str(hyperparameter_path)]
    )

    assert args.learning_rate == 0.0123
    assert args.no_opponent_map_scale_start == 0.35
    assert args.no_opponent_map_scale_end == 1.0
    assert args._autoload_source_num_agents == 192


def test_parse_training_args_autoloads_vecenv_defaults(tmp_path: Path) -> None:
    """Verify standardized autoload files can seed the self-play vecenv layout too.

    The automode Slurm path now relies on one pretuned JSON record rather than on a
    hand-written `--vec-backend auto` flag inside the batch script. This test keeps that
    contract explicit by checking that parser defaults pick up the saved vecenv layout when
    the standardized file includes a `vecenv_defaults` block.
    """

    hyperparameter_path = tmp_path / "autoload.json"
    hyperparameter_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "train_defaults": {"learning_rate": 0.0123},
                "vecenv_defaults": {
                    "vec_backend": "multiprocessing",
                    "num_envs": 96,
                    "vec_num_shards": 24,
                    "vec_batch_size": 24,
                },
            }
        ),
        encoding="utf-8",
    )

    args = train_pufferl.parse_training_args(
        ["--hyperparameters-path", str(hyperparameter_path)]
    )

    assert args.learning_rate == 0.0123
    assert args.vec_backend == "multiprocessing"
    assert args.num_envs == 96
    assert args.vec_num_shards == 24
    assert args.vec_batch_size == 24


def test_parse_training_args_cli_still_overrides_autoloaded_vecenv_defaults(
    tmp_path: Path,
) -> None:
    """Verify explicit vecenv CLI flags still beat the saved pretuned machine layout.

    Pretuning is meant to remove repeated typing, not to take control away from debugging
    sessions. This regression test keeps the precedence rule clear: the standardized JSON
    provides defaults, but an explicit CLI override for backend or worker layout must still
    win on the final parsed namespace.
    """

    hyperparameter_path = tmp_path / "autoload.json"
    hyperparameter_path.write_text(
        json.dumps(
            {
                "vecenv_defaults": {
                    "vec_backend": "multiprocessing",
                    "num_envs": 96,
                    "vec_num_shards": 24,
                    "vec_batch_size": 24,
                }
            }
        ),
        encoding="utf-8",
    )

    args = train_pufferl.parse_training_args(
        [
            "--hyperparameters-path",
            str(hyperparameter_path),
            "--vec-backend",
            "native",
            "--num-envs",
            "32",
        ]
    )

    assert args.vec_backend == "native"
    assert args.num_envs == 32
    assert args.vec_num_shards == 24
    assert args.vec_batch_size == 24


def test_parse_training_args_repo_defaults_enable_no_opponent_curriculum() -> None:
    """Verify plain repo-default runs now start with an active warm-start curriculum.

    The no-opponent warm-start is supposed to be easier than full-field sparse-reward play.
    This test keeps that expectation explicit by checking that the built-in parser defaults
    now enable a shrinking-to-full-field curriculum even when no autoload file is present.
    """

    args = train_pufferl.parse_training_args(["--no-autoload-hyperparameters"])

    assert args.no_opponent_map_scale_ladder == "0.2,0.4,0.6,0.8,1.0"
    assert args.no_opponent_phase_goal_rate_threshold == 0.8
    assert args.no_opponent_phase_multi_goal_rate_threshold == 0.0
    assert args.no_opponent_eval_games == 100
    assert train_pufferl.field_curriculum_enabled(args) is True


def test_resolve_no_opponent_game_length_matches_eval_horizon_with_floor() -> None:
    """Verify the shared no-opponent task horizon keeps parity without going below 400.

    Warm-start training and no-opponent evaluation now share one resolver so the gate cannot
    silently grade the policy on a longer task than the trainer actually optimized. The
    historical 400-step floor still matters for shorter diagnostic runs, so both behaviors
    are checked together here.
    """

    assert train_pufferl.resolve_no_opponent_game_length(250) == 400
    assert train_pufferl.resolve_no_opponent_game_length(600) == 600


def test_build_train_config_adapts_autoloaded_rollout_sizes_to_new_agent_count(
    tmp_path: Path,
) -> None:
    """Verify autoloaded batch sizes scale with the current number of live agents.

    The standardized file may come from a sweep that used a different team size or vector
    layout. The trainer should preserve the tuned rollout ratios in that case, because the
    old absolute `train_batch_size` can become illegal once the active agent count changes.
    """

    hyperparameter_path = tmp_path / "autoload.json"
    hyperparameter_path.write_text(
        json.dumps(
            {
                "train_defaults": {
                    "bptt_horizon": 64,
                    "train_batch_size": 24576,
                    "minibatch_size": 6144,
                },
                "rollout_defaults": {
                    "source_num_agents": 192,
                    "batch_multiple": 2,
                    "minibatch_divisor": 4,
                },
            }
        ),
        encoding="utf-8",
    )

    args = train_pufferl.parse_training_args(
        ["--hyperparameters-path", str(hyperparameter_path)]
    )
    vecenv = SimpleNamespace(num_agents=60)

    config = train_pufferl.build_train_config(args, vecenv, device="cpu")

    assert config["batch_size"] == 7680
    assert config["minibatch_size"] == 1920


def test_run_no_opponent_rollouts_uses_resolved_no_opponent_game_length(
    monkeypatch,
) -> None:
    """Verify vectorized no-opponent evaluation uses the shared task horizon and stage scale.

    The warm-start failure came from evaluation allowing goals past the point where training
    episodes had already ended. The new native-vector evaluation path also needs to run on
    the active map stage rather than on a silently different field scale. This test keeps
    both parts honest by checking that `run_no_opponent_rollouts` forwards the resolved game
    length into the vector env factory and applies the requested stage scale before rollout.
    """

    observed: dict[str, float | int] = {}

    class FakeWrappedEnv:
        """Minimal vectorized no-opponent env stub that terminates immediately for tests."""

        num_envs = 1

        def reset(self, seed: int | None = 0):
            return np.zeros((1, 1), dtype=np.float32), []

        def step(self, actions):
            return (
                np.zeros((1, 1), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
                np.ones((1,), dtype=bool),
                np.zeros((1,), dtype=bool),
                [],
            )

        def get_state(self, env_idx: int = 0) -> dict[str, tuple[int, int]]:
            return {"goals": (0, 0)}

        def set_field_scale(self, scale: float) -> None:
            observed["field_scale"] = float(scale)

        def close(self) -> None:
            return None

    class FakePolicy:
        """Minimal policy stub that exposes the eval API used by rollout helpers."""

        training = False

        def eval(self) -> None:
            return None

        def train(self) -> None:
            return None

        def forward_eval(self, obs):
            return train_pufferl.torch.zeros((obs.shape[0], 9)), None

    def fake_make_soccer_vecenv(**kwargs):
        observed["game_length"] = int(kwargs["game_length"])
        observed["num_envs"] = int(kwargs["vec"].shard_num_envs)
        return object()

    monkeypatch.setattr(train_pufferl, "make_soccer_vecenv", fake_make_soccer_vecenv)
    monkeypatch.setattr(
        train_pufferl,
        "BlueTeamNoOpponentWrapper",
        lambda env, players_per_team: FakeWrappedEnv(),
    )

    train_pufferl.run_no_opponent_rollouts(
        FakePolicy(),
        players_per_team=1,
        seed=7,
        device="cpu",
        num_games=1,
        max_steps=600,
        field_scale=0.8,
    )

    assert observed["game_length"] == 600
    assert observed["num_envs"] == 1
    assert observed["field_scale"] == 0.8


def test_resolve_requested_train_sizes_clamps_to_max_minibatch_size() -> None:
    """Verify oversized requested minibatches are projected below PuffeRL's hard cap.

    The self-play crash came from a requested minibatch that was structurally valid for the
    batch and horizon, but larger than the trainer's `max_minibatch_size`. This test keeps
    the low-level projection rule explicit so future refactors do not let that runtime-only
    constraint slip past config assembly again.
    """

    batch_size, minibatch_size = train_pufferl.resolve_requested_train_sizes(
        2520,
        horizon=64,
        requested_batch_size=322560,
        requested_minibatch_size=80640,
        max_minibatch_size=32768,
    )

    assert batch_size == 322560
    assert minibatch_size == 32256


def test_build_train_config_clamps_oversized_autoloaded_self_play_minibatch(
    tmp_path: Path,
) -> None:
    """Verify large self-play autoload scaling stays legal under PuffeRL's minibatch cap.

    The failing cluster run warmed up on a small native layout, then switched into a much
    larger self-play vecenv. That agent-count jump scaled the tuned rollout ratio to an
    `80640` minibatch, which PuffeRL rejected against its `32768` hard limit. This test
    recreates that shape directly and confirms the trainer now clamps the minibatch while
    leaving the scaled batch size intact.
    """

    hyperparameter_path = tmp_path / "autoload.json"
    hyperparameter_path.write_text(
        json.dumps(
            {
                "train_defaults": {
                    "bptt_horizon": 64,
                    "train_batch_size": 24576,
                    "minibatch_size": 6144,
                },
                "rollout_defaults": {
                    "source_num_agents": 192,
                    "batch_multiple": 2,
                    "minibatch_divisor": 4,
                },
            }
        ),
        encoding="utf-8",
    )

    args = train_pufferl.parse_training_args(
        ["--hyperparameters-path", str(hyperparameter_path)]
    )
    vecenv = SimpleNamespace(num_agents=2520)

    config = train_pufferl.build_train_config(args, vecenv, device="cpu")

    assert config["batch_size"] == 322560
    assert config["minibatch_size"] == 32256


def test_build_train_config_projects_explicit_minibatch_override_under_cap() -> None:
    """Verify explicit CLI minibatch overrides are still legalized instead of ignored.

    Users should be able to request a rollout size directly and rely on the trainer to map
    it onto the nearest legal PuffeRL value. This test makes sure the new cap-aware path
    still applies to explicit CLI requests, not only to autoloaded hyperparameters.
    """

    args = train_pufferl.parse_training_args(
        [
            "--players-per-team",
            "5",
            "--num-envs",
            "252",
            "--vec-backend",
            "native",
            "--train-batch-size",
            "322560",
            "--minibatch-size",
            "80640",
        ]
    )
    vecenv = SimpleNamespace(num_agents=2520)

    config = train_pufferl.build_train_config(args, vecenv, device="cpu")

    assert config["batch_size"] == 322560
    assert config["minibatch_size"] == 32256


def test_resolve_requested_train_sizes_preserves_legal_minibatch_under_cap() -> None:
    """Verify already-legal minibatch requests are left alone when under the hard cap.

    The new limit-aware projection should only intervene when the requested minibatch is
    illegal. This test keeps the common case stable by checking that a legal request below
    the configured cap still comes back unchanged.
    """

    batch_size, minibatch_size = train_pufferl.resolve_requested_train_sizes(
        60,
        horizon=64,
        requested_batch_size=7680,
        requested_minibatch_size=1920,
        max_minibatch_size=32768,
    )

    assert batch_size == 7680
    assert minibatch_size == 1920


def test_parse_training_args_cli_still_overrides_autoload(tmp_path: Path) -> None:
    """Verify explicit CLI flags continue to beat the autoloaded defaults.

    The standardized file is meant to save typing, not to make the trainer harder to
    control. This test confirms normal argparse precedence still holds after the two-stage
    parse.
    """

    hyperparameter_path = tmp_path / "autoload.json"
    hyperparameter_path.write_text(
        json.dumps({"train_defaults": {"learning_rate": 0.0123}}),
        encoding="utf-8",
    )

    args = train_pufferl.parse_training_args(
        [
            "--hyperparameters-path",
            str(hyperparameter_path),
            "--learning-rate",
            "0.0042",
        ]
    )

    assert args.learning_rate == 0.0042


def test_save_match_video_uses_resolved_no_opponent_game_length(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Verify no-opponent replay export uses the same task horizon as warm-start eval.

    The no-opponent video is part of the debugging loop for the warm-start gate. This test
    keeps that replay meaningful by ensuring the no-opponent export path creates the env with
    the same shared episode length that training and greedy evaluation now use.
    """

    observed: dict[str, int] = {}

    class FakeVideoEnv:
        """Small render-capable env stub for exercising the video export loop."""

        def reset(self, seed: int | None = 0):
            return np.zeros((1, 1), dtype=np.float32), []

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def step(self, actions):
            return (
                np.zeros((1, 1), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
                np.ones((1,), dtype=bool),
                np.zeros((1,), dtype=bool),
                [],
            )

        def close(self) -> None:
            return None

    class FakeVideoPolicy(train_pufferl.torch.nn.Module):
        """Tiny policy module that provides one parameter and deterministic logits."""

        def __init__(self) -> None:
            super().__init__()
            self.bias = train_pufferl.torch.nn.Parameter(
                train_pufferl.torch.zeros((1,))
            )

        def forward_eval(self, obs):
            return train_pufferl.torch.zeros((obs.shape[0], 9)), None

    def fake_make_puffer_env(**kwargs):
        observed["game_length"] = int(kwargs["game_length"])
        return object()

    monkeypatch.setattr(train_pufferl, "make_puffer_env", fake_make_puffer_env)
    monkeypatch.setattr(
        train_pufferl,
        "BlueTeamNoOpponentWrapper",
        lambda env, players_per_team: FakeVideoEnv(),
    )
    monkeypatch.setattr(
        train_pufferl,
        "_write_video_frames",
        lambda frames, requested_path, fps, label, overwrite_existing=False: requested_path,
    )

    args = SimpleNamespace(
        players_per_team=1,
        seed=3,
        video_max_steps=5,
        video_fps=20,
        no_opponent_eval_max_steps=600,
    )

    result = train_pufferl.save_match_video(
        FakeVideoPolicy(),
        args,
        output_path=tmp_path / "no_opponent.mp4",
        label="test video",
        opponents_enabled=False,
        overwrite_existing=True,
    )

    assert observed["game_length"] == 600
    assert result == tmp_path / "no_opponent.mp4"


def test_parse_training_args_preserves_explicit_video_output_overrides() -> None:
    """Verify explicit video path flags are not remapped into the run-scoped defaults.

    The new per-run video directory should only affect the built-in default paths. When a user
    manually points either video artifact at a custom location, that choice needs to survive the
    later run-output resolution step unchanged.
    """

    args = train_pufferl.parse_training_args(
        [
            "--video-output",
            "custom/self_play.mp4",
            "--best-checkpoint-video-output",
            "custom/best_checkpoint.mp4",
        ]
    )

    train_pufferl.configure_run_video_outputs(
        args,
        logger=SimpleNamespace(run_id="run-123"),
        run_start_time=123.0,
    )

    assert args._explicit_video_output is True
    assert args._explicit_best_checkpoint_video_output is True
    assert args.video_output == "custom/self_play.mp4"
    assert args.best_checkpoint_video_output == "custom/best_checkpoint.mp4"


def test_build_run_summary_records_no_opponent_task_config() -> None:
    """Verify the machine-readable summary includes resolved warm-start task details.

    Tuning and postmortem analysis both read the JSON run summary rather than terminal logs.
    This test keeps the new observability path stable by checking that the summary records
    both the resolved no-opponent game length and whether the field curriculum is active.
    """

    args = train_pufferl.parse_training_args(
        [
            "--no-autoload-hyperparameters",
            "--no-opponent-eval-max-steps",
            "600",
        ]
    )
    train_config = train_pufferl.build_train_config(
        args,
        SimpleNamespace(num_agents=60),
        device="cpu",
    )

    summary = train_pufferl._build_run_summary(
        args=args,
        trainer=SimpleNamespace(start_time=0.0, global_step=123, epoch=4),
        train_config=train_config,
        eval_interval_epochs=2,
        vec_config=train_pufferl.VecEnvConfig(),
        eval_vec_config=None,
        best_record=None,
        latest_best_metrics=None,
        final_best_metrics=None,
        latest_no_opponent_metrics=None,
        final_no_opponent_metrics=None,
        model_path=Path("model.pt"),
    )

    task_config = cast(dict[str, object], summary["no_opponent_task_config"])
    effective_hyperparameters = cast(
        dict[str, object], summary["effective_hyperparameters"]
    )
    assert task_config["training_game_length"] == 600
    assert task_config["eval_max_steps"] == 600
    assert task_config["field_curriculum_enabled"] is True
    assert task_config["map_scale_ladder"] == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert task_config["stage_count"] == 5
    assert effective_hyperparameters["no_opponent_training_game_length"] == 600


def test_autotune_training_vec_config_keeps_auto_for_self_play_stage(
    monkeypatch,
) -> None:
    """Verify self-play autotune stays on `auto` even when warm-start curriculum is enabled.

    The no-opponent warm-start may still need a native environment for map scaling, but the
    user's requested behavior is to autotune the long self-play phase rather than letting the
    warm-start restrict the main vector-layout search. This regression test checks that the
    benchmark entry point still receives `auto`.
    """

    observed: dict[str, object] = {}

    def fake_autotune_vecenv(**kwargs):
        observed.update(kwargs)
        return make_autotune_outcome()

    monkeypatch.setattr(train_pufferl, "autotune_vecenv", fake_autotune_vecenv)
    args = SimpleNamespace(
        vec_backend="auto",
        players_per_team=3,
        autotune_seconds=0.01,
        autotune_max_num_envs=8,
        autotune_max_num_shards=4,
        no_opponent_map_scale_ladder="0.2,0.4",
        no_opponent_map_scale_start=0.3,
        no_opponent_map_scale_end=1.0,
    )

    vec_config, benchmark = train_pufferl.autotune_training_vec_config(args)

    assert observed["backend"] == "auto"
    assert vec_config.backend == "native"
    assert benchmark.backend == "native"
