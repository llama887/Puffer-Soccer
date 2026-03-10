import numpy as np

from puffer_soccer.envs.marl2d import make_puffer_env
from puffer_soccer.envs.marl2d.core import MAX_SIGNED_ENV_SEED, normalize_env_seed
from puffer_soccer.vector_env import VecEnvConfig, make_soccer_vecenv


def test_scalar_puffer_env_shapes_discrete():
    env = make_puffer_env(players_per_team=3, action_mode="discrete")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (6, 58)
    assert env.global_states.shape == (6, 107)

    actions = np.zeros((6,), dtype=np.int32)
    obs, rewards, terminals, truncations, _ = env.step(actions)
    assert obs.shape == (6, 58)
    assert rewards.shape == (6,)
    assert terminals.shape == (6,)
    assert truncations.shape == (6,)
    env.close()


def test_scalar_puffer_env_roundtrip():
    env = make_puffer_env(players_per_team=2, action_mode="discrete")
    obs, infos = env.reset(seed=123)
    assert obs.shape == (4, 44)
    assert infos == []

    actions = np.zeros((4,), dtype=np.int32)
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert obs.shape == (4, 44)
    assert rewards.shape == (4,)
    assert terms.shape == (4,)
    assert truncs.shape == (4,)
    assert isinstance(infos, list)
    env.close()


def test_scalar_puffer_env_rgb_array_render():
    env = make_puffer_env(
        players_per_team=2, action_mode="discrete", render_mode="rgb_array"
    )
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    env.close()


def test_scalar_puffer_env_render_uses_terminal_snapshot_until_next_step():
    env = make_puffer_env(
        players_per_team=2,
        action_mode="discrete",
        game_length=3,
        render_mode="rgb_array",
    )
    env.reset(seed=0)

    actions = np.zeros((4,), dtype=np.int32)
    terminals = np.zeros((4,), dtype=bool)
    for _ in range(3):
        _, _, terminals, _, _ = env.step(actions)

    assert terminals.all()

    terminal_state = env.get_state()
    assert terminal_state["num_steps"] == 3

    terminal_frame = env.render()
    assert terminal_frame is not None
    assert terminal_frame.ndim == 3

    env.step(actions)
    live_state = env.get_state()
    assert live_state["num_steps"] == 1
    env.close()


def test_native_vec_env_shapes_and_second_render():
    env = make_soccer_vecenv(
        players_per_team=2,
        action_mode="discrete",
        game_length=400,
        render_mode="rgb_array",
        seed=0,
        vec=VecEnvConfig(backend="native", shard_num_envs=2, num_shards=1),
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == (8, 44)
    assert env.global_states.shape == (8, 73)

    actions = np.zeros((8,), dtype=np.int32)
    obs, rewards, terminals, truncations, _ = env.step(actions)
    assert obs.shape == (8, 44)
    assert rewards.shape == (8,)
    assert terminals.shape == (8,)
    assert truncations.shape == (8,)

    frame = env.render(env_idx=1)
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    env.close()


def test_large_reset_seed_is_folded_into_signed_env_range():
    env = make_puffer_env(players_per_team=2, action_mode="discrete")
    large_seed = MAX_SIGNED_ENV_SEED + 123_456
    obs, infos = env.reset(seed=large_seed)

    assert obs.shape == (4, 44)
    assert infos == []
    assert normalize_env_seed(large_seed) == 123_456
    env.close()
