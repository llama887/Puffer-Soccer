"""Tests for the opt-in referee module (out-of-bounds restarts and offside).

These use the `env_debug_*` native bindings to place the ball/agents and to trigger touch
events directly, instead of orchestrating exact kick-contact geometry. That keeps the tests
focused on the referee state machine rather than on kick physics, which is already covered
elsewhere.
"""

import numpy as np
import pytest

from puffer_soccer.envs.marl2d import make_puffer_env
from puffer_soccer.envs.marl2d.core import DISCRETE_ACTION_NOOP
from puffer_soccer.envs.marl2d.csrc import binding

PLAY_STATE_IN_PLAY = 0
PLAY_STATE_THROW_IN = 1
PLAY_STATE_GOAL_KICK = 2
PLAY_STATE_CORNER_KICK = 3
PLAY_STATE_OFFSIDE_FREE_KICK = 4


def _noop_step(env):
    actions = np.full((env.num_players,), DISCRETE_ACTION_NOOP, dtype=np.int32)
    return env.step(actions)


def test_referee_disabled_obs_and_state_sizes_match_base_layout():
    env = make_puffer_env(players_per_team=2, action_mode="discrete", enable_referee=False)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (4, 44)
        assert env.global_states.shape == (4, 73)
    finally:
        env.close()


def test_referee_enabled_obs_and_state_sizes_add_extra_fields():
    env = make_puffer_env(players_per_team=2, action_mode="discrete", enable_referee=True)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (4, 49)
        assert env.global_states.shape == (4, 78)
    finally:
        env.close()


def test_throw_in_restarts_to_team_opposite_last_toucher():
    env = make_puffer_env(players_per_team=2, action_mode="discrete", enable_referee=True)
    try:
        env.reset(seed=0)
        binding.env_debug_trigger_touch(env._handle, 0)  # last touch: team 0 (blue)
        binding.env_debug_set_ball(env._handle, 0.0, 40.0, 0.0, 0.0)  # beyond y_out_end (35)

        _noop_step(env)
        state = env.get_state()

        assert state["play_state"] == PLAY_STATE_THROW_IN
        assert state["restart_team"] == 1
        assert state["restart_spot"] == pytest.approx((0.0, 35.0))
        assert state["ball"] == pytest.approx((0.0, 35.0, 0.0, 0.0))
    finally:
        env.close()


def test_dead_ball_freezes_ball_and_clamps_opponents_out_of_radius():
    env = make_puffer_env(players_per_team=2, action_mode="discrete", enable_referee=True)
    try:
        env.reset(seed=0)
        binding.env_debug_trigger_touch(env._handle, 0)
        binding.env_debug_set_ball(env._handle, 0.0, 40.0, 0.0, 0.0)
        _noop_step(env)
        state = env.get_state()
        assert state["play_state"] == PLAY_STATE_THROW_IN
        assert state["restart_team"] == 1

        # Player 0 is on the non-restart team (blue) and starts 1 unit from the restart spot.
        binding.env_debug_set_agent(env._handle, 0, 0.0, 34.0, 0.0)
        _noop_step(env)
        state = env.get_state()

        pos = state["positions"][0]
        dist = float(np.hypot(pos[0] - 0.0, pos[1] - 35.0))
        assert dist == pytest.approx(9.0, abs=1e-3)
        assert state["ball"] == pytest.approx((0.0, 35.0, 0.0, 0.0))
    finally:
        env.close()


def test_throw_in_resumes_when_restart_team_touches_ball():
    env = make_puffer_env(players_per_team=2, action_mode="discrete", enable_referee=True)
    try:
        env.reset(seed=0)
        binding.env_debug_trigger_touch(env._handle, 0)
        binding.env_debug_set_ball(env._handle, 0.0, 40.0, 0.0, 0.0)
        _noop_step(env)
        assert env.get_state()["play_state"] == PLAY_STATE_THROW_IN

        binding.env_debug_trigger_touch(env._handle, 2)  # restart team (red) touches
        state = env.get_state()
        assert state["play_state"] == PLAY_STATE_IN_PLAY
    finally:
        env.close()


def test_goal_kick_awarded_when_attacking_team_put_ball_out_over_byline():
    env = make_puffer_env(players_per_team=2, action_mode="discrete", enable_referee=True)
    try:
        env.reset(seed=0)
        binding.env_debug_trigger_touch(env._handle, 0)  # blue (attacks +x) touched last
        # Beyond x_out_end (50) and outside the goal mouth (|y| > goal_half_h=20), i.e. wide.
        binding.env_debug_set_ball(env._handle, 60.0, 30.0, 0.0, 0.0)

        _noop_step(env)
        state = env.get_state()

        assert state["play_state"] == PLAY_STATE_GOAL_KICK
        assert state["restart_team"] == 1
    finally:
        env.close()


def test_corner_kick_awarded_when_defending_team_put_ball_out_over_byline():
    env = make_puffer_env(players_per_team=2, action_mode="discrete", enable_referee=True)
    try:
        env.reset(seed=0)
        binding.env_debug_trigger_touch(env._handle, 2)  # red (defends +x end) touched last
        binding.env_debug_set_ball(env._handle, 60.0, 30.0, 0.0, 0.0)

        _noop_step(env)
        state = env.get_state()

        assert state["play_state"] == PLAY_STATE_CORNER_KICK
        assert state["restart_team"] == 0
        assert state["restart_spot"][0] == pytest.approx(50.0)
    finally:
        env.close()


def test_offside_marks_and_infringes_on_receiving_touch():
    env = make_puffer_env(players_per_team=2, action_mode="discrete", enable_referee=True)
    try:
        env.reset(seed=0)
        # Blue (team 0) attacks +x. Passer stays back; receiver is ahead of both defenders.
        binding.env_debug_set_agent(env._handle, 0, -10.0, 0.0, 0.0)  # passer
        binding.env_debug_set_agent(env._handle, 1, 30.0, 0.0, 0.0)  # receiver (offside)
        binding.env_debug_set_agent(env._handle, 2, 20.0, 5.0, 0.0)  # last defender
        binding.env_debug_set_agent(env._handle, 3, 15.0, -5.0, 0.0)  # 2nd-to-last defender
        binding.env_debug_set_ball(env._handle, -10.0, 0.0, 0.0, 0.0)

        binding.env_debug_trigger_touch(env._handle, 0)  # passer's first touch: no mark yet
        state = env.get_state()
        assert not bool(state["offside_marked"][1])

        binding.env_debug_trigger_touch(env._handle, 1)  # receiver touches while offside
        state = env.get_state()

        assert state["play_state"] == PLAY_STATE_OFFSIDE_FREE_KICK
        assert state["restart_team"] == 1
        assert state["restart_spot"] == pytest.approx((30.0, 0.0))
    finally:
        env.close()


def test_no_offside_mark_when_ball_is_won_by_opponent():
    env = make_puffer_env(players_per_team=2, action_mode="discrete", enable_referee=True)
    try:
        env.reset(seed=0)
        binding.env_debug_set_agent(env._handle, 0, -10.0, 0.0, 0.0)
        binding.env_debug_set_agent(env._handle, 1, 30.0, 0.0, 0.0)
        binding.env_debug_set_agent(env._handle, 2, 20.0, 5.0, 0.0)
        binding.env_debug_set_agent(env._handle, 3, 15.0, -5.0, 0.0)
        binding.env_debug_set_ball(env._handle, -10.0, 0.0, 0.0, 0.0)

        binding.env_debug_trigger_touch(env._handle, 0)  # blue touches first
        binding.env_debug_trigger_touch(env._handle, 2)  # red intercepts (team change)
        state = env.get_state()

        assert not bool(state["offside_marked"].any())
        assert state["play_state"] == PLAY_STATE_IN_PLAY
    finally:
        env.close()
