import numpy as np
import pytest

from puffer_soccer.envs.marl2d import make_puffer_env


def _reference_available():
    try:
        from puffer_soccer.envs.marl2d.reference_adapter import ReferenceEnvAdapter

        ReferenceEnvAdapter(players_per_team=1)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reference_available(), reason="third-party reference dependencies unavailable")


@pytest.mark.parametrize("players_per_team", [1, 2, 5, 11])
@pytest.mark.parametrize("action_mode", ["continuous", "discrete"])
def test_reset_parity(players_per_team, action_mode):
    from puffer_soccer.envs.marl2d.reference_adapter import ReferenceEnvAdapter

    ref = ReferenceEnvAdapter(players_per_team=players_per_team, action_mode=action_mode)
    new = make_puffer_env(num_envs=1, players_per_team=players_per_team, action_mode=action_mode)

    ref_step = ref.reset()
    new_obs, _ = new.reset(seed=0)

    assert ref_step.obs.shape == new_obs.shape
    assert ref_step.state.shape == new.global_states.shape

    # Strict parity for 11v11 where one-hot ids are deterministic upstream.
    if players_per_team == 11:
        np.testing.assert_allclose(new_obs, ref_step.obs, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(new.global_states, ref_step.state, rtol=1e-4, atol=1e-4)

    new.close()


@pytest.mark.parametrize("players_per_team", [1, 2, 5, 11])
@pytest.mark.parametrize("action_mode", ["continuous", "discrete"])
def test_step_parity(players_per_team, action_mode):
    from puffer_soccer.envs.marl2d.reference_adapter import ReferenceEnvAdapter

    ref = ReferenceEnvAdapter(players_per_team=players_per_team, action_mode=action_mode)
    new = make_puffer_env(num_envs=1, players_per_team=players_per_team, action_mode=action_mode)

    ref.reset()
    new.reset(seed=0)

    rng = np.random.default_rng(7)
    for _ in range(10):
        if action_mode == "continuous":
            actions = rng.uniform(-1.0, 1.0, size=(players_per_team * 2, 2)).astype(np.float32)
        else:
            actions = rng.integers(0, 9, size=(players_per_team * 2,), dtype=np.int32)

        ref_step = ref.step(actions)
        new_obs, new_rew, new_done, _, _ = new.step(actions)

        assert ref_step.obs.shape == new_obs.shape
        assert ref_step.rewards.shape == new_rew.shape

        # Numeric parity target (most stable at full teams).
        if players_per_team == 11:
            np.testing.assert_allclose(new_obs, ref_step.obs, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(new_rew, ref_step.rewards, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(new.global_states, ref_step.state, rtol=1e-4, atol=1e-4)
            assert bool(new_done.all()) == ref_step.done

    new.close()
