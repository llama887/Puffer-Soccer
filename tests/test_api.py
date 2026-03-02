import numpy as np

from puffer_soccer.envs.marl2d import make_puffer_env


def test_puffer_env_shapes_discrete():
    env = make_puffer_env(num_envs=2, players_per_team=3, action_mode="discrete")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (12, 58)
    assert env.global_states.shape == (12, 107)

    actions = np.zeros((12,), dtype=np.int32)
    obs, rewards, terminals, truncations, _ = env.step(actions)
    assert obs.shape == (12, 58)
    assert rewards.shape == (12,)
    assert terminals.shape == (12,)
    assert truncations.shape == (12,)
    env.close()


def test_puffer_env_roundtrip():
    env = make_puffer_env(num_envs=1, players_per_team=2, action_mode="discrete")
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


def test_puffer_env_rgb_array_render():
    env = make_puffer_env(num_envs=1, players_per_team=2, action_mode="discrete", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    env.close()


def test_puffer_env_rgb_array_render_second_env():
    env = make_puffer_env(num_envs=2, players_per_team=2, action_mode="discrete", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render(env_idx=1)
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    env.close()
