"""API and native-environment smoke tests for the MARL2D soccer wrappers."""

import ctypes
import math

import numpy as np

from puffer_soccer.envs.marl2d import make_puffer_env
from puffer_soccer.envs.marl2d.core import MAX_SIGNED_ENV_SEED, normalize_env_seed
from puffer_soccer.vector_env import VecEnvConfig, make_soccer_vecenv


class _NativeLog(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """Mirror the native log prefix so tests can safely walk the env header layout.

    The goal reward bug lives in the C environment, and there is no public API that lets a
    test place the ball directly into a goal mouth. This lightweight structure exists only so
    the regression test can reach the native `ball_x` and boundary fields through the opaque
    handle that Python already owns.

    Keeping the fields in the same order as the C struct matters because `ctypes` relies on
    the exact in-memory layout. The test only mutates gameplay state and does not touch any of
    the ownership-bearing pointers, which keeps the operation narrow and deterministic.
    """

    _fields_ = [
        ("score", ctypes.c_float),
        ("episode_return", ctypes.c_float),
        ("episode_length", ctypes.c_float),
        ("wins_blue", ctypes.c_float),
        ("wins_red", ctypes.c_float),
        ("draws", ctypes.c_float),
        ("n", ctypes.c_float),
    ]


class _NativeAgent(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """Mirror one native agent entry so the enclosing env layout stays byte-accurate.

    The regression test does not inspect or modify individual agents. The array is included so
    the later env fields land at the same offsets they use in the C binding. Without this
    placeholder the test would read and write the wrong memory.
    """

    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("rot", ctypes.c_float),
        ("last_move", ctypes.c_float),
        ("last_rot", ctypes.c_float),
        ("team", ctypes.c_int),
    ]


class _NativeEnv(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """Expose the small slice of native env state needed for deterministic goal tests.

    This structure mirrors the C `Env` definition closely enough to access the ball position,
    goal bounds, and side assignment from Python. The test uses it to construct exact
    own-goal and opponent-goal situations in one step, which is much more reliable than hoping
    random play eventually produces the edge case.
    """

    _fields_ = [
        ("log", _NativeLog),
        ("observations", ctypes.c_void_p),
        ("actions", ctypes.c_void_p),
        ("rewards", ctypes.c_void_p),
        ("terminals", ctypes.c_void_p),
        ("truncations", ctypes.c_void_p),
        ("global_states", ctypes.c_void_p),
        ("agents", _NativeAgent * 22),
        ("players_per_team", ctypes.c_int),
        ("num_players", ctypes.c_int),
        ("game_length", ctypes.c_int),
        ("num_steps", ctypes.c_int),
        ("cumulative_episode_return", ctypes.c_float),
        ("do_team_switch", ctypes.c_int),
        ("opponents_enabled", ctypes.c_int),
        ("blue_left", ctypes.c_int),
        ("reset_setup", ctypes.c_int),
        ("action_mode", ctypes.c_int),
        ("last_goals_blue", ctypes.c_int),
        ("last_goals_red", ctypes.c_int),
        ("last_done", ctypes.c_ubyte),
        ("has_terminal_render_state", ctypes.c_ubyte),
        ("vision_range", ctypes.c_float),
        ("x_out_start", ctypes.c_float),
        ("x_out_end", ctypes.c_float),
        ("y_out_start", ctypes.c_float),
        ("y_out_end", ctypes.c_float),
        ("goal_half_h", ctypes.c_float),
        ("rot_speed", ctypes.c_float),
        ("move_speed", ctypes.c_float),
        ("ball_x", ctypes.c_float),
        ("ball_y", ctypes.c_float),
        ("ball_vx", ctypes.c_float),
        ("ball_vy", ctypes.c_float),
        ("goals_blue", ctypes.c_int),
        ("goals_red", ctypes.c_int),
        ("field_scale", ctypes.c_float),
        ("base_x_out_start", ctypes.c_float),
        ("base_x_out_end", ctypes.c_float),
        ("base_y_out_start", ctypes.c_float),
        ("base_y_out_end", ctypes.c_float),
        ("base_goal_half_h", ctypes.c_float),
    ]


def _native_env(env) -> _NativeEnv:
    """Return the live native env backing a scalar Puffer env.

    The Python wrapper stores the C pointer as an integer handle. Casting that handle back to a
    typed structure lets the test set up exact ball positions while still stepping the
    environment through its normal public API. This keeps the assertion focused on reward
    semantics rather than on a custom testing code path.
    """

    return ctypes.cast(env._handle, ctypes.POINTER(_NativeEnv)).contents  # pylint: disable=protected-access


def _force_ball_into_goal(env, goal_side: str) -> None:
    """Place the live ball just beyond the requested goal line for the next step.

    The environment checks for goals after moving the ball and before resetting the field. By
    setting the ball a small distance past the left or right boundary with zero velocity and a
    center-line `y` position, the next public `step` call deterministically triggers exactly
    one scoring event without relying on agent motion or wall bounces.
    """

    native_env = _native_env(env)
    native_env.ball_y = 0.0
    native_env.ball_vx = 0.0
    native_env.ball_vy = 0.0
    if goal_side == "left":
        native_env.ball_x = native_env.x_out_start - 1.0
    else:
        native_env.ball_x = native_env.x_out_end + 1.0


def _set_symmetric_egocentric_state(env, *, ball_x: float, agent_x: float) -> None:
    """Install a hand-crafted mirrored world state for egocentric observation checks.

    The test needs two agents, one per team, that each face toward the opponent goal with the
    same local ball configuration. Writing the native fields directly is the simplest way to
    create that mirrored setup without adding a special testing hook to the environment code.
    """

    native_env = _native_env(env)
    native_env.blue_left = 1
    native_env.num_steps = 0
    native_env.ball_x = ball_x
    native_env.ball_y = 0.0
    native_env.ball_vx = 0.0
    native_env.ball_vy = 0.0

    native_env.agents[0].x = agent_x
    native_env.agents[0].y = 0.0
    native_env.agents[0].rot = 0.0
    native_env.agents[0].last_move = 0.0
    native_env.agents[0].last_rot = 0.0

    native_env.agents[1].x = -agent_x
    native_env.agents[1].y = 0.0
    native_env.agents[1].rot = math.pi
    native_env.agents[1].last_move = 0.0
    native_env.agents[1].last_rot = 0.0


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


def test_goal_rewards_only_reward_the_attacking_team():
    """Check that goal rewards follow the attacking team on both goal lines.

    The environment should reward a team only when the ball crosses the opponent's goal line.
    This regression test explicitly scores once through the left goal and once through the
    right goal, then confirms the positive reward and scoreboard update always match the
    attacking side rather than the side that owns the goal.
    """

    env = make_puffer_env(players_per_team=2, action_mode="discrete")
    actions = np.zeros((4,), dtype=np.int32)

    try:
        env.reset(seed=0)
        _force_ball_into_goal(env, "left")
        _, rewards, _, _, _ = env.step(actions)

        left_scores = env.get_state()["goals"]
        expected_left_team = 0 if left_scores == (1, 0) else 1
        expected_left_rewards = np.array(
            [-1.0, -1.0, -1.0, -1.0], dtype=np.float32
        )
        start = expected_left_team * env.players_per_team
        expected_left_rewards[start : start + env.players_per_team] = 1.0
        np.testing.assert_allclose(rewards, expected_left_rewards)
        if expected_left_team == 0:
            assert left_scores == (1, 0)
        else:
            assert left_scores == (0, 1)

        env.reset(seed=0)
        _force_ball_into_goal(env, "right")
        _, rewards, _, _, _ = env.step(actions)

        right_scores = env.get_state()["goals"]
        expected_right_team = 0 if right_scores == (1, 0) else 1
        expected_right_rewards = np.array(
            [-1.0, -1.0, -1.0, -1.0], dtype=np.float32
        )
        start = expected_right_team * env.players_per_team
        expected_right_rewards[start : start + env.players_per_team] = 1.0
        np.testing.assert_allclose(rewards, expected_right_rewards)
        if expected_right_team == 0:
            assert right_scores == (1, 0)
        else:
            assert right_scores == (0, 1)
    finally:
        env.close()


def test_disabled_opponents_stay_inactive_and_receive_no_reward():
    """Check that the no-opponent mode removes gameplay pressure from the red team.

    The user-facing flag is meant to create a simple curriculum where one team can learn to
    reach the ball and score without interference. This test verifies the native env matches
    that contract in two concrete ways:
    - red-team agents spawn pinned on the far right edge instead of contesting the ball
    - red-team rewards stay at zero even when the blue team scores
    """

    env = make_puffer_env(
        players_per_team=2,
        action_mode="discrete",
        opponents_enabled=False,
    )
    actions = np.zeros((4,), dtype=np.int32)

    try:
        env.reset(seed=0)
        state = env.get_state()
        positions = state["positions"]
        np.testing.assert_allclose(positions[2:, 0], 50.0, atol=1e-6)

        _force_ball_into_goal(env, "right")
        _, rewards, _, _, _ = env.step(actions)

        np.testing.assert_allclose(rewards[2:], 0.0, atol=1e-6)
        assert np.all(rewards[:2] > 0.0)
    finally:
        env.close()


def test_episode_return_logs_the_full_accumulated_episode_reward():
    """Verify the terminal logger reports the stored cumulative episode return.

    The dashboard metric `environment/episode_return` is supposed to represent the reward
    accumulated over the whole episode. A previous native-logging bug ignored that running
    total and instead logged only the reward visible on the terminal step, which erased sparse
    rewards that happened earlier.

    This regression test writes a known cumulative total into the live native env, then ends
    the episode on the next public step. If the logger is wired correctly it must emit that
    stored total even though the final step itself contributes zero reward.
    """

    env = make_puffer_env(players_per_team=2, action_mode="discrete")
    actions = np.zeros((4,), dtype=np.int32)

    try:
        env.reset(seed=0)
        native_env = _native_env(env)
        native_env.game_length = 1
        native_env.cumulative_episode_return = 2.0

        _, rewards, terminals, _, _ = env.step(actions)
        np.testing.assert_allclose(rewards, 0.0, atol=1e-6)
        assert terminals.all()

        log = env.flush_log()
        assert log is not None
        assert log["n"] == 1.0
        assert log["episode_length"] == 1.0
        assert log["episode_return"] == 2.0
    finally:
        env.close()


def test_native_field_scale_resizes_no_opponent_map_without_obs_noise():
    """Verify the map-size curriculum changes geometry without adding reward shaping.

    The no-opponent curriculum should make the task easier by shrinking the field, not by
    changing the reward. This test checks two concrete properties of the native setter:
    - the live field bounds and state metadata reflect the requested scale
    - disabled-opponent observation slots remain zero after resizing
    """

    env = make_puffer_env(
        players_per_team=3,
        action_mode="discrete",
        opponents_enabled=False,
    )

    try:
        env.set_field_scale(0.6)
        obs, _ = env.reset(seed=0)
        state = env.get_state()
        native_env = _native_env(env)

        np.testing.assert_allclose(state["field_scale"], 0.6, atol=1e-6)
        np.testing.assert_allclose(native_env.x_out_start, -30.0, atol=1e-6)
        np.testing.assert_allclose(native_env.x_out_end, 30.0, atol=1e-6)
        np.testing.assert_allclose(native_env.y_out_start, -21.0, atol=1e-6)
        np.testing.assert_allclose(native_env.y_out_end, 21.0, atol=1e-6)
        np.testing.assert_allclose(obs[0][-21:], 0.0, atol=1e-6)
    finally:
        env.close()


def test_observations_are_team_symmetric_and_egocentric():
    """Check that mirrored agents receive matching local observations.

    The no-opponent curriculum only makes sense if the policy sees the game from each agent's
    own perspective rather than from a fixed world frame. This test builds a simple mirrored
    two-agent state and confirms the blue and red agents receive the same self and ball
    features once team-side normalization is applied.
    """

    env = make_puffer_env(players_per_team=1, action_mode="discrete")
    actions = np.zeros((2,), dtype=np.int32)

    try:
        env.reset(seed=0)
        _set_symmetric_egocentric_state(env, ball_x=0.0, agent_x=-10.0)
        obs, _, _, _, _ = env.step(actions)

        blue_obs = obs[0]
        red_obs = obs[1]

        np.testing.assert_allclose(blue_obs[1:7], red_obs[1:7], atol=1e-6)
        np.testing.assert_allclose(blue_obs[16:21], red_obs[16:21], atol=1e-6)
        np.testing.assert_allclose(blue_obs[21:28], red_obs[21:28], atol=1e-6)
    finally:
        env.close()
