import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from puffer_soccer.vector_env import VecEnvConfig, make_soccer_vecenv

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_pufferl.py"
SPEC = importlib.util.spec_from_file_location("train_pufferl", MODULE_PATH)
assert SPEC is not None
train_pufferl = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = train_pufferl
SPEC.loader.exec_module(train_pufferl)

compute_eval_interval_epochs = train_pufferl.compute_eval_interval_epochs
compute_train_sizes = train_pufferl.compute_train_sizes
make_side_assignment = train_pufferl.make_side_assignment
score_metrics_from_perspective = train_pufferl.score_metrics_from_perspective


def test_compute_eval_interval_epochs():
    assert compute_eval_interval_epochs(1000, 10) == 100
    assert compute_eval_interval_epochs(5, 10) == 1


def test_compute_train_sizes_rounds_small_batches_up_to_horizon():
    batch_size, minibatch_size = compute_train_sizes(total_agents=2, horizon=64)
    assert batch_size == 128
    assert minibatch_size == 64


def test_compute_train_sizes_caps_minibatch_at_batch_size():
    batch_size, minibatch_size = compute_train_sizes(total_agents=1, horizon=64)
    assert batch_size == 64
    assert minibatch_size == 64


def test_score_metrics_from_perspective():
    assert score_metrics_from_perspective(3, 1, current_on_blue=True) == (2.0, 1.0)
    assert score_metrics_from_perspective(3, 1, current_on_blue=False) == (-2.0, 0.0)
    assert score_metrics_from_perspective(2, 2, current_on_blue=True) == (0.0, 0.5)


def test_make_side_assignment():
    assert make_side_assignment(4).tolist() == [True, True, False, False]
    assert make_side_assignment(5).tolist() == [True, True, True, False, False]


def test_resolve_vec_config_manual_native():
    args = SimpleNamespace(
        vec_backend="native",
        num_envs=12,
        vec_num_shards=None,
        vec_batch_size=None,
    )

    config = train_pufferl.resolve_vec_config(args)

    assert config == VecEnvConfig(backend="native", shard_num_envs=12, num_shards=1)


def test_resolve_training_vec_config_runs_autotune_for_auto_backend():
    args = SimpleNamespace(
        vec_backend="auto",
        players_per_team=5,
        autotune_seconds=1.25,
        autotune_max_num_envs=96,
        autotune_max_num_shards=12,
    )
    benchmark = train_pufferl.BenchmarkResult(
        backend="multiprocessing",
        shard_num_envs=8,
        num_shards=12,
        batch_size=3,
        players_per_team=5,
        action_mode="discrete",
        sps=4321,
        cpu_avg=98.0,
        cpu_peak=99.0,
    )

    with patch.object(
        train_pufferl, "autotune_vecenv", return_value=SimpleNamespace(best=benchmark)
    ) as mocked:
        config, result = train_pufferl.resolve_training_vec_config(args)

    mocked.assert_called_once_with(
        players_per_team=5,
        seconds=1.25,
        action_mode="discrete",
        backend="auto",
        max_num_envs=96,
        max_num_shards=12,
        reporter=print,
    )
    assert result == benchmark
    assert config == VecEnvConfig(
        backend="multiprocessing",
        shard_num_envs=8,
        num_shards=12,
        num_workers=12,
        batch_size=3,
    )


def test_resolve_training_vec_config_manual_backend_skips_autotune():
    args = SimpleNamespace(
        vec_backend="multiprocessing",
        num_envs=24,
        vec_num_shards=6,
        vec_batch_size=3,
    )

    with patch.object(train_pufferl, "autotune_training_vec_config") as mocked:
        config, result = train_pufferl.resolve_training_vec_config(args)

    mocked.assert_not_called()
    assert result is None
    assert config == VecEnvConfig(
        backend="multiprocessing",
        shard_num_envs=4,
        num_shards=6,
        num_workers=6,
        batch_size=3,
    )


def test_resolve_eval_vec_config_defaults_to_training_layout():
    training_config = VecEnvConfig(
        backend="multiprocessing",
        shard_num_envs=5,
        num_shards=4,
        num_workers=4,
        batch_size=2,
    )

    assert train_pufferl.resolve_eval_vec_config(training_config, None) == VecEnvConfig(
        backend="native", shard_num_envs=20, num_shards=1
    )


def test_resolve_eval_vec_config_supports_native_override():
    training_config = VecEnvConfig(
        backend="multiprocessing",
        shard_num_envs=5,
        num_shards=4,
        num_workers=4,
        batch_size=2,
    )

    assert train_pufferl.resolve_eval_vec_config(training_config, 7) == VecEnvConfig(
        backend="native", shard_num_envs=7, num_shards=1
    )


def test_last_episode_scores_available_after_terminal_step():
    env = make_soccer_vecenv(
        players_per_team=1,
        action_mode="discrete",
        game_length=1,
        render_mode=None,
        seed=0,
        vec=VecEnvConfig(backend="native", shard_num_envs=2, num_shards=1),
    )
    env.reset(seed=0)

    actions = np.zeros((env.num_agents,), dtype=np.int32)
    _, _, terminals, _, _ = env.step(actions)

    assert terminals.reshape(env.num_envs, env.num_players).all(axis=1).tolist() == [
        True,
        True,
    ]
    assert env.get_last_episode_scores(0) is not None
    assert env.get_last_episode_scores(1) is not None
    assert env.get_last_episode_scores(0) is None
    env.close()


def test_evaluate_against_past_iterate_returns_metrics():
    env = make_soccer_vecenv(
        players_per_team=1,
        action_mode="discrete",
        game_length=4,
        render_mode=None,
        seed=0,
        vec=VecEnvConfig(backend="native", shard_num_envs=2, num_shards=1),
    )
    policy = train_pufferl.Policy(env)
    evaluator = train_pufferl.HeadToHeadEvaluator(
        players_per_team=1,
        game_length=4,
        vec_config=VecEnvConfig(backend="native", shard_num_envs=2, num_shards=1),
        device="cpu",
    )

    metrics = train_pufferl.evaluate_against_past_iterate(
        policy,
        train_pufferl.snapshot_policy_state(policy),
        evaluator=evaluator,
        games=2,
        seed=1,
    )

    assert metrics["games"] == 2.0
    assert isinstance(metrics["win_rate"], float)
    assert isinstance(metrics["score_diff"], float)
    evaluator.close()
    env.close()


def test_snapshot_policy_state_clones_tensors():
    env = make_soccer_vecenv(
        players_per_team=1,
        action_mode="discrete",
        game_length=4,
        render_mode=None,
        seed=0,
        vec=VecEnvConfig(backend="native", shard_num_envs=1, num_shards=1),
    )
    policy = train_pufferl.Policy(env)

    snapshot = train_pufferl.snapshot_policy_state(policy)
    first_name, first_param = next(iter(policy.state_dict().items()))

    assert snapshot[first_name] is not first_param
    assert np.array_equal(snapshot[first_name].numpy(), first_param.numpy())
    env.close()


def test_run_promotion_evaluation_stops_after_confident_batches():
    evaluator = SimpleNamespace(num_envs=4)
    batches = iter([([1.0] * 4, [2.0] * 4)] * 6)

    def fake_run_games(*args, **kwargs):
        return next(batches)

    evaluator.run_games = fake_run_games
    metrics = train_pufferl.run_promotion_evaluation(
        object(),
        {},
        evaluator,
        confidence=0.95,
        min_batches=4,
        max_batches=8,
        seed=3,
    )

    assert metrics["promoted"] == 1.0
    assert metrics["batches"] == 4.0
    assert metrics["games"] == 16.0
    assert metrics["win_rate_lcb"] > 0.5


def test_write_json_record_round_trip(tmp_path):
    path = tmp_path / "best_checkpoint.json"
    payload = {
        "artifact_ref": "entity/project/checkpoint:best",
        "cached_checkpoint_path": "/tmp/checkpoint.pt",
        "epoch": 12,
    }

    train_pufferl.write_json_record(path, payload)

    assert train_pufferl.read_json_record(path) == payload


def test_load_best_checkpoint_state_falls_back_to_cached_path(tmp_path):
    env = make_soccer_vecenv(
        players_per_team=1,
        action_mode="discrete",
        game_length=4,
        render_mode=None,
        seed=0,
        vec=VecEnvConfig(backend="native", shard_num_envs=1, num_shards=1),
    )
    policy = train_pufferl.Policy(env)
    checkpoint_path = tmp_path / "cached_best.pt"
    torch.save(policy.state_dict(), checkpoint_path)

    state_dict, loaded_path = train_pufferl.load_best_checkpoint_state(
        {
            "artifact_ref": "entity/project/checkpoint:best",
            "cached_checkpoint_path": str(checkpoint_path),
        },
        logger=None,
        cache_dir=tmp_path,
    )

    assert loaded_path == checkpoint_path
    assert set(state_dict) == set(policy.state_dict())
    env.close()


class DummyArtifact:
    def __init__(self, name, *, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path, name=None):
        self.files.append((path, name))


class DummyRun:
    def __init__(self):
        self.entity = "entity"
        self.project = "project"
        self.logged_artifacts = []

    def log_artifact(self, artifact, aliases):
        self.logged_artifacts.append((artifact, aliases))


class DummyWandb:
    Artifact = DummyArtifact

    def __init__(self):
        self.run = DummyRun()


class DummyLogger:
    def __init__(self):
        self.wandb = DummyWandb()
        self.run_id = "run-123"


def test_register_best_checkpoint_updates_pointer_and_history(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"weights")
    config_path = tmp_path / "best_checkpoint.json"
    history_path = tmp_path / "best_checkpoint_history.jsonl"
    logger = DummyLogger()

    record = train_pufferl.register_best_checkpoint(
        logger=logger,
        checkpoint_path=checkpoint_path,
        best_config_path=config_path,
        best_history_path=history_path,
        previous_best={"artifact_ref": "entity/project/older:best"},
        vec_config=VecEnvConfig(backend="native", shard_num_envs=8, num_shards=1),
        run_id="run-123",
        epoch=7,
        global_step=1234,
        event="promotion",
        promotion_metrics={
            "batches": 4.0,
            "confidence": 0.95,
            "games": 32.0,
            "win_rate": 0.75,
            "score_diff": 1.25,
            "win_rate_lcb": 0.61,
        },
    )

    assert (
        record["artifact_ref"]
        == "entity/project/best-checkpoint-run-123-epoch-000007:best"
    )
    assert json.loads(config_path.read_text())["artifact_ref"] == record["artifact_ref"]
    history_record = json.loads(history_path.read_text().strip())
    assert history_record["promotion_win_rate_vs_previous_best"] == 0.75
    assert history_record["previous_best_artifact_ref"] == "entity/project/older:best"
    logged_artifact, aliases = logger.wandb.run.logged_artifacts[0]
    assert logged_artifact.files == [(str(checkpoint_path), "model.pt")]
    assert aliases == ["best", "epoch-000007"]
