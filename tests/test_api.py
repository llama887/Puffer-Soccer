import numpy as np

from puffer_soccer.envs.marl2d import make_parallel_env, make_puffer_env


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


def test_parallel_env_roundtrip():
    env = make_parallel_env(players_per_team=2, action_mode="discrete")
    obs, infos = env.reset(seed=123)
    assert len(obs) == 4
    assert len(infos) == 4

    actions = {a: 0 for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert len(obs) == 4
    assert len(rewards) == 4
    assert all("global_state" in infos[a] for a in infos)
    assert set(terms.keys()) == set(obs.keys())
    assert set(truncs.keys()) == set(obs.keys())
    env.close()


def test_parallel_env_rgb_array_render():
    env = make_parallel_env(players_per_team=2, action_mode="discrete", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    env.close()
