"""End-to-end correctness tests for the C soccer env.

These tests construct deterministic env states (via the ctypes overlay) and verify the
intended one-step semantics. They cover every game mechanic the policy interacts with so a
silent regression in the C binding shows up as a unit-test failure long before training is
launched. A companion script (`scripts/render_env_check_videos.py`) renders the same
scenarios as short videos that can be inspected visually.

The ctypes layout deliberately mirrors the C `Env` struct up through
`warm_start_red_in_formation`; tests do not touch fields beyond that, so the truncated
mirror remains byte-accurate for all asserted offsets.
"""

from __future__ import annotations

import ctypes
import math
from typing import Tuple

import numpy as np
import pytest

from puffer_soccer.envs.marl2d import make_puffer_env
from puffer_soccer.envs.marl2d.core import DISCRETE_ACTION_COUNT


MAX_PLAYERS = 22
NOOP_ACTION = 0
MOVE_FORWARD_ACTION = 1
MOVE_BACKWARD_ACTION = 2
ROTATE_LEFT_ACTION = 3
ROTATE_RIGHT_ACTION = 4
KICK_MAX_ACTION = 12  # action index for max-power kick (DISCRETE_KICK_BUCKETS-1)


class _NativeAgent(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("rot", ctypes.c_float),
        ("last_move", ctypes.c_float),
        ("last_rot", ctypes.c_float),
        ("team", ctypes.c_int),
        ("stat_kick", ctypes.c_float),
        ("stat_speed", ctypes.c_float),
        ("stat_turn", ctypes.c_float),
        ("steer_angle", ctypes.c_float),
    ]


class _NativeLog(ctypes.Structure):
    _fields_ = [
        ("score", ctypes.c_float),
        ("episode_return", ctypes.c_float),
        ("blue_team_episode_return", ctypes.c_float),
        ("red_team_episode_return", ctypes.c_float),
        ("episode_length", ctypes.c_float),
        ("wins_blue", ctypes.c_float),
        ("wins_red", ctypes.c_float),
        ("draws", ctypes.c_float),
        ("n", ctypes.c_float),
    ]


class _NativeEnv(ctypes.Structure):
    _fields_ = [
        ("log", _NativeLog),
        ("observations", ctypes.c_void_p),
        ("actions", ctypes.c_void_p),
        ("rewards", ctypes.c_void_p),
        ("terminals", ctypes.c_void_p),
        ("truncations", ctypes.c_void_p),
        ("global_states", ctypes.c_void_p),
        ("agents", _NativeAgent * MAX_PLAYERS),
        ("players_per_team", ctypes.c_int),
        ("num_players", ctypes.c_int),
        ("game_length", ctypes.c_int),
        ("num_steps", ctypes.c_int),
        ("cumulative_episode_return", ctypes.c_float),
        ("cumulative_blue_team_episode_return", ctypes.c_float),
        ("cumulative_red_team_episode_return", ctypes.c_float),
        ("do_team_switch", ctypes.c_int),
        ("warm_start_reward_shaping", ctypes.c_int),
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
        ("last_touch_team", ctypes.c_int),
        ("throw_in_active", ctypes.c_int),
        ("throw_in_player", ctypes.c_int),
        ("shaping_distance_penalty", ctypes.c_float),
        ("shaping_touch_bonus", ctypes.c_float),
        ("shaping_velocity_bonus", ctypes.c_float),
        ("warm_start_red_in_formation", ctypes.c_int),
    ]


def _ne(env) -> _NativeEnv:
    return ctypes.cast(env._handle, ctypes.POINTER(_NativeEnv)).contents


def _make_env(
    *,
    players_per_team: int = 5,
    warm_start: bool = False,
) -> object:
    """One-stop env factory used by every test for consistent defaults."""

    return make_puffer_env(
        players_per_team=players_per_team,
        action_mode="discrete",
        warm_start_reward_shaping=warm_start,
    )


def _park_all_agents(ne: _NativeEnv) -> None:
    """Place every agent far from the ball and other interesting state.

    Many of the tests want to exercise one mechanic at a time. Parking every agent at the
    field corners removes accidental ball collisions or offside triggers from the picture.
    """

    n = ne.num_players
    half = ne.players_per_team
    for i in range(n):
        if ne.agents[i].team == 0:
            ne.agents[i].x = ne.x_out_start + 1.0
            ne.agents[i].y = ne.y_out_start + 1.0 + (i % half)
        else:
            ne.agents[i].x = ne.x_out_end - 1.0
            ne.agents[i].y = ne.y_out_end - 1.0 - (i % half)
        ne.agents[i].rot = 0.0
        ne.agents[i].last_move = 0.0
        ne.agents[i].last_rot = 0.0
        ne.agents[i].steer_angle = 0.0
        ne.agents[i].stat_kick = 1.0
        ne.agents[i].stat_speed = 1.0
        ne.agents[i].stat_turn = 1.0


def _all_noop_actions(env) -> np.ndarray:
    return np.full((env.num_agents,), NOOP_ACTION, dtype=np.int32)


# ---------------------------------------------------------------------------
# Throw-in mechanics
# ---------------------------------------------------------------------------


class TestThrowIns:
    def test_top_sideline_triggers_throw_in_with_opposing_player(self):
        env = _make_env(players_per_team=2)
        env.reset(seed=0)
        ne = _ne(env)
        _park_all_agents(ne)
        # Place a red defender at midfield/top so they are the closest opponent.
        ne.agents[2].x = 0.0
        ne.agents[2].y = ne.y_out_end - 5.0
        # Ball just past top sideline, last touched by blue.
        ne.ball_x = 0.0
        ne.ball_y = ne.y_out_end + 0.5
        ne.ball_vx = 0.0
        ne.ball_vy = 0.5
        ne.last_touch_team = 0
        ne.throw_in_active = 0

        env.step(_all_noop_actions(env))

        assert ne.throw_in_active == 1
        assert ne.agents[ne.throw_in_player].team == 1, "thrower must be opposing team"
        assert ne.ball_y == pytest.approx(ne.y_out_end, abs=1e-4), "ball clamped to sideline"
        assert ne.ball_vx == pytest.approx(0.0, abs=1e-4)
        assert ne.ball_vy == pytest.approx(0.0, abs=1e-4)
        assert ne.agents[ne.throw_in_player].x == pytest.approx(ne.ball_x, abs=1e-4)
        assert ne.agents[ne.throw_in_player].y == pytest.approx(ne.ball_y, abs=1e-4)
        env.close()

    def test_bottom_sideline_triggers_throw_in_with_opposing_player(self):
        env = _make_env(players_per_team=2)
        env.reset(seed=0)
        ne = _ne(env)
        _park_all_agents(ne)
        ne.agents[2].x = 0.0
        ne.agents[2].y = ne.y_out_start + 5.0  # nearest red to bottom side
        ne.ball_x = 0.0
        ne.ball_y = ne.y_out_start - 0.5
        ne.ball_vy = -0.5
        ne.last_touch_team = 0
        ne.throw_in_active = 0

        env.step(_all_noop_actions(env))

        assert ne.throw_in_active == 1
        assert ne.agents[ne.throw_in_player].team == 1
        assert ne.ball_y == pytest.approx(ne.y_out_start, abs=1e-4)
        env.close()

    def test_ball_unconditionally_clamped_even_when_throw_in_already_active(self):
        env = _make_env(players_per_team=2)
        env.reset(seed=0)
        ne = _ne(env)
        _park_all_agents(ne)
        # Pretend a throw-in is already in progress for blue. Push ball past sideline
        # with non-zero velocity to mimic a stray body impulse from another agent.
        ne.ball_x = 5.0
        ne.ball_y = ne.y_out_end + 1.5
        ne.ball_vx = 1.0
        ne.ball_vy = 1.0
        ne.last_touch_team = 0
        ne.throw_in_active = 1
        ne.throw_in_player = 0  # arbitrary; gate just needs to be active

        env.step(_all_noop_actions(env))

        # Sideline clamp must run regardless of throw_in_active so the ball never
        # drifts off-field. The setup logic (teleport + velocity zero) does stay
        # gated, so velocities here may persist; the position must be clamped.
        assert ne.ball_y <= ne.y_out_end + 1e-4
        env.close()

    def test_throw_in_locked_player_cannot_translate(self):
        env = _make_env(players_per_team=2)
        env.reset(seed=0)
        ne = _ne(env)
        _park_all_agents(ne)
        # Mark agent 0 (blue) as a locked thrower.
        ne.agents[0].x = 5.0
        ne.agents[0].y = 0.0
        ne.agents[0].rot = 0.0
        ne.throw_in_active = 1
        ne.throw_in_player = 0
        ne.last_touch_team = 1  # red threw out, blue is now thrower
        actions = _all_noop_actions(env)
        actions[0] = MOVE_FORWARD_ACTION  # would normally move +x

        prev_x = ne.agents[0].x
        prev_y = ne.agents[0].y
        env.step(actions)

        assert ne.agents[0].x == pytest.approx(prev_x, abs=1e-4)
        assert ne.agents[0].y == pytest.approx(prev_y, abs=1e-4)
        env.close()

    def test_throw_in_clears_when_locked_player_kicks(self):
        env = _make_env(players_per_team=2)
        env.reset(seed=0)
        ne = _ne(env)
        _park_all_agents(ne)
        # Place locked thrower coincident with ball so the kick definitely fires.
        ne.ball_x = 0.0
        ne.ball_y = ne.y_out_end
        ne.ball_vx = 0.0
        ne.ball_vy = 0.0
        ne.agents[0].x = 0.0
        ne.agents[0].y = ne.y_out_end
        ne.agents[0].rot = -math.pi / 2  # face downfield (toward -y, into pitch)
        ne.throw_in_active = 1
        ne.throw_in_player = 0
        ne.last_touch_team = 1
        actions = _all_noop_actions(env)
        actions[0] = KICK_MAX_ACTION

        env.step(actions)

        assert ne.throw_in_active == 0
        assert ne.throw_in_player == -1
        # Ball should now have non-zero velocity from the kick.
        assert abs(ne.ball_vx) + abs(ne.ball_vy) > 0.1
        env.close()


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


class TestGoals:
    def test_ball_past_right_line_within_posts_scores_for_attacking_team(self):
        env = _make_env(players_per_team=2)
        env.reset(seed=0)
        ne = _ne(env)
        _park_all_agents(ne)
        # Blue defends left (default); blue scores when ball crosses right line.
        ne.ball_x = ne.x_out_end + 0.5
        ne.ball_y = 0.0
        ne.ball_vx = 0.0
        ne.ball_vy = 0.0
        prev_blue = ne.goals_blue
        prev_red = ne.goals_red

        env.step(_all_noop_actions(env))

        assert ne.goals_blue == prev_blue + 1
        assert ne.goals_red == prev_red
        env.close()

    def test_ball_past_left_line_within_posts_scores_for_red(self):
        env = _make_env(players_per_team=2)
        env.reset(seed=0)
        ne = _ne(env)
        _park_all_agents(ne)
        ne.ball_x = ne.x_out_start - 0.5
        ne.ball_y = 0.0
        ne.ball_vx = 0.0
        ne.ball_vy = 0.0
        prev_blue = ne.goals_blue
        prev_red = ne.goals_red

        env.step(_all_noop_actions(env))

        assert ne.goals_red == prev_red + 1
        assert ne.goals_blue == prev_blue
        env.close()

    def test_ball_past_right_line_outside_posts_bounces_no_goal(self):
        env = _make_env(players_per_team=2)
        env.reset(seed=0)
        ne = _ne(env)
        _park_all_agents(ne)
        ne.ball_x = ne.x_out_end + 0.5
        ne.ball_y = ne.goal_half_h + 5.0  # well outside the goalmouth
        ne.ball_vx = 1.0
        ne.ball_vy = 0.0
        prev_blue = ne.goals_blue
        prev_red = ne.goals_red

        env.step(_all_noop_actions(env))

        assert ne.goals_blue == prev_blue
        assert ne.goals_red == prev_red
        # Ball should be clamped back into bounds and vx reflected.
        assert ne.ball_x <= ne.x_out_end + 1e-4
        assert ne.ball_vx < 0.0  # reversed
        env.close()


# ---------------------------------------------------------------------------
# Offside
# ---------------------------------------------------------------------------


class TestOffside:
    """Cover the FIFA-style offside check, including the ball-position guard."""

    def _setup_offside_state(
        self,
        env,
        *,
        attacker_x: float,
        ball_x: float,
        defender_xs: Tuple[float, float],
    ) -> _NativeEnv:
        ne = _ne(env)
        _park_all_agents(ne)
        # Blue (team 0) attacks right (blue_left=1 default). Red defenders define line.
        # Place red defenders at the requested x. Both must be in red's half (>=0)
        # so the second-deepest is meaningful.
        red_start = ne.players_per_team
        ne.agents[red_start].x = defender_xs[0]
        ne.agents[red_start].y = -2.0
        ne.agents[red_start + 1].x = defender_xs[1]
        ne.agents[red_start + 1].y = 2.0
        # Place attacker (agent 0, blue) at requested x.
        ne.agents[0].x = attacker_x
        ne.agents[0].y = 0.0
        # Ball at requested x, pulled away from sidelines.
        ne.ball_x = ball_x
        ne.ball_y = 0.0
        ne.ball_vx = 0.0
        ne.ball_vy = 0.0
        ne.throw_in_active = 0
        ne.last_touch_team = 0
        return ne

    def test_attacker_past_defender_and_past_ball_triggers_offside(self):
        env = _make_env(players_per_team=2)
        env.reset(seed=0)
        # Red defenders at +20 and +25; second-to-last (less deep) = +20.
        # Attacker at +30 (past the line) and AHEAD of ball at +5.
        ne = self._setup_offside_state(
            env, attacker_x=30.0, ball_x=5.0, defender_xs=(20.0, 25.0)
        )

        env.step(_all_noop_actions(env))

        # Offside reuses the throw-in lock plumbing.
        assert ne.throw_in_active == 1
        assert ne.agents[ne.throw_in_player].team == 1, "lock goes to defender team"
        env.close()

    def test_ball_carrier_is_not_offside_when_at_ball(self):
        """Regression test for the ball-position guard: standing at the ball never offside."""

        env = _make_env(players_per_team=2)
        env.reset(seed=0)
        ne = self._setup_offside_state(
            env, attacker_x=30.0, ball_x=30.0, defender_xs=(20.0, 25.0)
        )

        env.step(_all_noop_actions(env))

        assert ne.throw_in_active == 0, "ball-carrier must never be flagged offside"
        env.close()

    def test_attacker_behind_ball_is_not_offside(self):
        """Attacker past defender line but behind the ball is onside."""

        env = _make_env(players_per_team=2)
        env.reset(seed=0)
        ne = self._setup_offside_state(
            env, attacker_x=30.0, ball_x=40.0, defender_xs=(20.0, 25.0)
        )

        env.step(_all_noop_actions(env))

        assert ne.throw_in_active == 0
        env.close()

    def test_attacker_in_own_half_is_not_offside(self):
        env = _make_env(players_per_team=2)
        env.reset(seed=0)
        ne = self._setup_offside_state(
            env, attacker_x=-10.0, ball_x=-15.0, defender_xs=(20.0, 25.0)
        )

        env.step(_all_noop_actions(env))

        assert ne.throw_in_active == 0
        env.close()

    def test_offside_skipped_during_warm_start(self):
        env = _make_env(players_per_team=2, warm_start=True)
        env.reset(seed=0)
        ne = self._setup_offside_state(
            env, attacker_x=30.0, ball_x=5.0, defender_xs=(20.0, 25.0)
        )

        env.step(_all_noop_actions(env))

        assert ne.throw_in_active == 0, "warm-start disables offside entirely"
        env.close()


# ---------------------------------------------------------------------------
# Field scaling (curriculum geometry)
# ---------------------------------------------------------------------------


class TestFieldScaling:
    @pytest.mark.parametrize("scale", [0.2, 0.5, 1.0])
    def test_field_bounds_track_requested_scale(self, scale: float):
        env = _make_env(players_per_team=3)
        env.set_field_scale(scale)
        env.reset(seed=0)
        ne = _ne(env)
        np.testing.assert_allclose(ne.field_scale, scale, atol=1e-6)
        np.testing.assert_allclose(ne.x_out_end, 50.0 * scale, atol=1e-5)
        np.testing.assert_allclose(ne.y_out_end, 35.0 * scale, atol=1e-5)
        # All agents and the ball stay within the new bounds.
        for i in range(ne.num_players):
            assert ne.x_out_start - 1e-4 <= ne.agents[i].x <= ne.x_out_end + 1e-4
            assert ne.y_out_start - 1e-4 <= ne.agents[i].y <= ne.y_out_end + 1e-4
        assert ne.x_out_start - 1e-4 <= ne.ball_x <= ne.x_out_end + 1e-4
        env.close()


# ---------------------------------------------------------------------------
# Warm-start red placement
# ---------------------------------------------------------------------------


class TestRedPlacement:
    def test_corners_mode_places_red_far_from_goalmouth(self):
        env = _make_env(players_per_team=5, warm_start=True)
        env.reset(seed=0)
        state = env.get_state()
        red_positions = state["positions"][5:]
        # Every red agent should be near a sideline (|y| > goal_half_h)
        # and on the right half (x > 0).
        ne = _ne(env)
        for x, y in red_positions:
            assert x > 0, "red on its half"
            assert abs(y) > ne.goal_half_h * 0.5, "clear of goalmouth"
        env.close()

    def test_formation_mode_uses_self_play_layout(self):
        env = _make_env(players_per_team=5, warm_start=True)
        env.set_red_in_formation(True)
        env.reset(seed=0)
        state = env.get_state()
        red_positions = state["positions"][5:]
        # Formation spreads red across their defensive half rather than clustering at corners.
        ne = _ne(env)
        x_values = [x for x, _ in red_positions]
        assert max(x_values) - min(x_values) > 5.0, (
            "formation should span more than corner cluster"
        )
        env.close()


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------


class TestRewardShaping:
    def test_warm_start_distance_penalty_only_applies_to_blue(self):
        env = _make_env(players_per_team=2, warm_start=True)
        env.reset(seed=0)
        ne = _ne(env)
        _park_all_agents(ne)
        # Make sure no goals fire so the only reward source is shaping.
        ne.ball_x = 0.0
        ne.ball_y = 0.0
        ne.ball_vx = 0.0
        ne.ball_vy = 0.0
        actions = _all_noop_actions(env)
        _, rewards, _, _, _ = env.step(actions)
        assert np.all(rewards[:2] != 0.0), "blue should accrue distance penalty"
        np.testing.assert_allclose(rewards[2:], 0.0, atol=1e-6)
        env.close()

    def test_self_play_mode_emits_only_sparse_goal_reward(self):
        env = _make_env(players_per_team=2, warm_start=False)
        env.reset(seed=0)
        ne = _ne(env)
        _park_all_agents(ne)
        ne.ball_x = 0.0
        ne.ball_y = 0.0
        ne.ball_vx = 0.0
        ne.ball_vy = 0.0
        _, rewards, _, _, _ = env.step(_all_noop_actions(env))
        np.testing.assert_allclose(rewards, 0.0, atol=1e-6)
        env.close()


# ---------------------------------------------------------------------------
# Discrete action mechanics
# ---------------------------------------------------------------------------


class TestDiscreteActions:
    def _isolated_agent_env(self):
        env = _make_env(players_per_team=1)
        env.reset(seed=0)
        ne = _ne(env)
        _park_all_agents(ne)
        # Place the ball far away so kicks don't accidentally launch it.
        ne.ball_x = ne.x_out_end - 1.0
        ne.ball_y = ne.y_out_end - 1.0
        ne.ball_vx = 0.0
        ne.ball_vy = 0.0
        # Place agent 0 in the middle of their half.
        ne.agents[0].x = -10.0
        ne.agents[0].y = 0.0
        ne.agents[0].rot = 0.0
        return env, ne

    def test_noop_action_leaves_pose_unchanged(self):
        env, ne = self._isolated_agent_env()
        prev = (ne.agents[0].x, ne.agents[0].y, ne.agents[0].rot)
        actions = _all_noop_actions(env)
        env.step(actions)
        np.testing.assert_allclose((ne.agents[0].x, ne.agents[0].y), prev[:2], atol=1e-4)
        np.testing.assert_allclose(ne.agents[0].rot, prev[2], atol=1e-4)
        env.close()

    def test_move_forward_advances_along_facing(self):
        env, ne = self._isolated_agent_env()
        prev_x = ne.agents[0].x
        actions = _all_noop_actions(env)
        actions[0] = MOVE_FORWARD_ACTION
        env.step(actions)
        # Bicycle model: forward action with rot=0 increases x.
        assert ne.agents[0].x > prev_x, "forward action with rot=0 must increase x"
        env.close()

    def test_rotate_actions_change_steering_angle(self):
        env, ne = self._isolated_agent_env()
        prev_steer = ne.agents[0].steer_angle
        actions = _all_noop_actions(env)
        actions[0] = ROTATE_LEFT_ACTION
        env.step(actions)
        assert ne.agents[0].steer_angle != prev_steer
        env.close()

    def test_kick_action_does_not_translate_agent(self):
        env, ne = self._isolated_agent_env()
        prev_x, prev_y = ne.agents[0].x, ne.agents[0].y
        actions = _all_noop_actions(env)
        actions[0] = KICK_MAX_ACTION
        env.step(actions)
        # Discrete kick is exclusive of move; agent stays put (no move/rot).
        np.testing.assert_allclose((ne.agents[0].x, ne.agents[0].y), (prev_x, prev_y), atol=1e-4)
        env.close()

    def test_action_count_matches_constant(self):
        env = _make_env(players_per_team=1)
        try:
            assert env.single_action_space.n == DISCRETE_ACTION_COUNT
        finally:
            env.close()
