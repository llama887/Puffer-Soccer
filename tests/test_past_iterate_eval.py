import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from puffer_soccer.vector_env import VecEnvConfig, make_soccer_vecenv

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_pufferl.py"
SPEC = importlib.util.spec_from_file_location("train_pufferl", MODULE_PATH)
train_pufferl = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
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
    args = SimpleNamespace(
        players_per_team=1,
        seed=0,
        past_iterate_eval_envs=2,
        past_iterate_eval_game_length=4,
        past_iterate_eval_games=2,
    )

    metrics = train_pufferl.evaluate_against_past_iterate(
        policy, train_pufferl.clone_state_dict(policy), args, epoch=1
    )

    assert metrics["games"] == 2.0
    assert isinstance(metrics["win_rate"], float)
    assert isinstance(metrics["score_diff"], float)
    env.close()
