"""Render short visual scenarios that demonstrate every env mechanic the tests cover.

Companion to `tests/test_env_correctness.py`. Each scenario forces a deterministic env
state (via the ctypes overlay), takes a noop or scripted action sequence, and renders the
result as an mp4 in `videos/env_check/`. Running both the pytest assertions and the
videos together gives you both a pass/fail signal and a visual sanity check that nothing
silly is happening between the asserted steps.
"""

from __future__ import annotations

import argparse
import ctypes
import math
from pathlib import Path

import imageio
import numpy as np

from puffer_soccer.envs.marl2d import make_puffer_env

# Reuse the ctypes mirror from the correctness test module so layout stays in lockstep.
import sys
TESTS_DIR = Path(__file__).resolve().parents[1] / "tests"
sys.path.insert(0, str(TESTS_DIR))
from test_env_correctness import (  # type: ignore[import-not-found]
    _NativeEnv,
    _ne,
    _park_all_agents,
    _all_noop_actions,
    KICK_MAX_ACTION,
    MOVE_FORWARD_ACTION,
    NOOP_ACTION,
)


VIDEOS_ROOT = Path("videos/env_check")
DEFAULT_FPS = 12


def _make_env(*, players_per_team: int = 5, warm_start: bool = False, seed: int = 0):
    return make_puffer_env(
        players_per_team=players_per_team,
        action_mode="discrete",
        warm_start_reward_shaping=warm_start,
        render_mode="rgb_array",
        seed=seed,
    )


def _render_frames(env, num_frames: int, action_fn=None) -> list[np.ndarray]:
    """Step the env, capturing one frame per step.

    `action_fn(step_idx) -> np.ndarray` controls actions per step. Defaults to all-noop so
    most scenarios show the consequence of the seeded state without confounding inputs.
    """

    frames: list[np.ndarray] = []
    for step in range(num_frames):
        frame = env.render()
        if frame is not None:
            frames.append(frame.astype(np.uint8, copy=False))
        actions = action_fn(step) if action_fn else _all_noop_actions(env)
        env.step(actions)
    final_frame = env.render()
    if final_frame is not None:
        frames.append(final_frame.astype(np.uint8, copy=False))
    return frames


def _save(frames: list[np.ndarray], name: str, *, fps: int = DEFAULT_FPS) -> Path:
    VIDEOS_ROOT.mkdir(parents=True, exist_ok=True)
    out = VIDEOS_ROOT / f"{name}.mp4"
    if not frames:
        raise RuntimeError(f"no frames for {name}")
    imageio.mimsave(out, frames, fps=fps, macro_block_size=None)
    return out


# ---------------------------------------------------------------------------
# Throw-in scenarios
# ---------------------------------------------------------------------------


def render_throwin_top():
    env = _make_env(players_per_team=3)
    env.reset(seed=0)
    ne = _ne(env)
    _park_all_agents(ne)
    # Place a red defender near top mid-field as the eventual thrower.
    ne.agents[3].x = 0.0
    ne.agents[3].y = ne.y_out_end - 3.0
    # Place blue near the top sideline so the visual story is "blue kicked it out."
    ne.agents[0].x = -2.0
    ne.agents[0].y = ne.y_out_end - 1.0
    # Ball just past top sideline, last touched by blue.
    ne.ball_x = 0.0
    ne.ball_y = ne.y_out_end + 0.5
    ne.ball_vy = 0.5
    ne.last_touch_team = 0
    ne.throw_in_active = 0

    return _save(_render_frames(env, num_frames=24), "throwin_top_sideline")


def render_throwin_bottom():
    env = _make_env(players_per_team=3)
    env.reset(seed=0)
    ne = _ne(env)
    _park_all_agents(ne)
    ne.agents[3].x = 0.0
    ne.agents[3].y = ne.y_out_start + 3.0
    ne.ball_x = 0.0
    ne.ball_y = ne.y_out_start - 0.5
    ne.ball_vy = -0.5
    ne.last_touch_team = 0
    ne.throw_in_active = 0
    return _save(_render_frames(env, num_frames=24), "throwin_bottom_sideline")


def render_throwin_locked_kick():
    """Show the locked thrower kicking and clearing the lock."""

    env = _make_env(players_per_team=3)
    env.reset(seed=0)
    ne = _ne(env)
    _park_all_agents(ne)
    ne.ball_x = 0.0
    ne.ball_y = ne.y_out_end
    ne.agents[0].x = 0.0
    ne.agents[0].y = ne.y_out_end
    ne.agents[0].rot = -math.pi / 2  # face into the pitch
    ne.throw_in_active = 1
    ne.throw_in_player = 0
    ne.last_touch_team = 1

    def action(step):
        a = _all_noop_actions(env)
        # Hold the kick for the first few frames so the kick definitely fires while the
        # ball is still at the locked spot.
        if step < 4:
            a[0] = KICK_MAX_ACTION
        return a

    return _save(_render_frames(env, num_frames=30, action_fn=action), "throwin_locked_kick")


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


def render_goal_for_blue():
    env = _make_env(players_per_team=3)
    env.reset(seed=0)
    ne = _ne(env)
    _park_all_agents(ne)
    ne.ball_x = ne.x_out_end + 0.5
    ne.ball_y = 0.0
    return _save(_render_frames(env, num_frames=20), "goal_for_blue")


def render_goal_for_red():
    env = _make_env(players_per_team=3)
    env.reset(seed=0)
    ne = _ne(env)
    _park_all_agents(ne)
    ne.ball_x = ne.x_out_start - 0.5
    ne.ball_y = 0.0
    return _save(_render_frames(env, num_frames=20), "goal_for_red")


def render_wall_bounce_above_post():
    """Ball crosses x-boundary outside the goal posts → bounce, no goal."""

    env = _make_env(players_per_team=3)
    env.reset(seed=0)
    ne = _ne(env)
    _park_all_agents(ne)
    ne.ball_x = ne.x_out_end - 1.0
    ne.ball_y = ne.goal_half_h + 5.0
    ne.ball_vx = 4.0
    ne.ball_vy = 0.0
    return _save(_render_frames(env, num_frames=20), "wall_bounce_above_post")


# ---------------------------------------------------------------------------
# Offside
# ---------------------------------------------------------------------------


def _setup_offside(env, *, attacker_x, ball_x, defender_xs):
    ne = _ne(env)
    _park_all_agents(ne)
    red_start = ne.players_per_team
    ne.agents[red_start].x = defender_xs[0]
    ne.agents[red_start].y = -2.0
    ne.agents[red_start + 1].x = defender_xs[1]
    ne.agents[red_start + 1].y = 2.0
    ne.agents[0].x = attacker_x
    ne.agents[0].y = 0.0
    ne.ball_x = ball_x
    ne.ball_y = 0.0
    ne.throw_in_active = 0
    ne.last_touch_team = 0
    return ne


def render_offside_caught():
    env = _make_env(players_per_team=3)
    env.reset(seed=0)
    _setup_offside(env, attacker_x=30.0, ball_x=5.0, defender_xs=(20.0, 25.0))
    return _save(_render_frames(env, num_frames=20), "offside_caught_past_defender_and_ball")


def render_offside_ball_carrier_safe():
    """Ball-carrier (attacker at ball position) must NOT be offside."""

    env = _make_env(players_per_team=3)
    env.reset(seed=0)
    _setup_offside(env, attacker_x=30.0, ball_x=30.0, defender_xs=(20.0, 25.0))
    return _save(_render_frames(env, num_frames=20), "offside_ball_carrier_safe")


def render_offside_behind_ball_safe():
    """Attacker past defender but BEHIND the ball is onside."""

    env = _make_env(players_per_team=3)
    env.reset(seed=0)
    _setup_offside(env, attacker_x=30.0, ball_x=40.0, defender_xs=(20.0, 25.0))
    return _save(_render_frames(env, num_frames=20), "offside_behind_ball_safe")


def render_offside_disabled_in_warm_start():
    """Same offside-ish state but with warm_start_reward_shaping=True → no trigger."""

    env = _make_env(players_per_team=3, warm_start=True)
    env.reset(seed=0)
    _setup_offside(env, attacker_x=30.0, ball_x=5.0, defender_xs=(20.0, 25.0))
    return _save(_render_frames(env, num_frames=20), "offside_disabled_in_warm_start")


# ---------------------------------------------------------------------------
# Field scaling + red placement
# ---------------------------------------------------------------------------


def render_field_scale_small():
    env = _make_env(players_per_team=5, warm_start=True)
    env.set_field_scale(0.3)
    env.reset(seed=0)
    return _save(_render_frames(env, num_frames=20), "field_scale_0_3")


def render_field_scale_full():
    env = _make_env(players_per_team=5)
    env.reset(seed=0)
    return _save(_render_frames(env, num_frames=20), "field_scale_1_0_self_play_formation")


def render_red_corners():
    env = _make_env(players_per_team=5, warm_start=True)
    env.reset(seed=0)
    return _save(_render_frames(env, num_frames=20), "warm_start_red_corners")


def render_red_formation():
    env = _make_env(players_per_team=5, warm_start=True)
    env.set_red_in_formation(True)
    env.reset(seed=0)
    return _save(_render_frames(env, num_frames=20), "warm_start_red_formation")


# ---------------------------------------------------------------------------
# Action mechanics
# ---------------------------------------------------------------------------


def render_move_forward():
    env = _make_env(players_per_team=1)
    env.reset(seed=0)
    ne = _ne(env)
    _park_all_agents(ne)
    ne.agents[0].x = -10.0
    ne.agents[0].y = 0.0
    ne.agents[0].rot = 0.0
    ne.ball_x = ne.x_out_end - 1.0
    ne.ball_y = ne.y_out_end - 1.0

    def action(_step):
        a = _all_noop_actions(env)
        a[0] = MOVE_FORWARD_ACTION
        return a

    return _save(_render_frames(env, num_frames=30, action_fn=action), "action_move_forward")


def render_kick_only_no_translate():
    env = _make_env(players_per_team=1)
    env.reset(seed=0)
    ne = _ne(env)
    _park_all_agents(ne)
    ne.agents[0].x = -10.0
    ne.agents[0].y = 0.0
    ne.agents[0].rot = 0.0
    # Place ball next to the agent so the kick visibly launches it.
    ne.ball_x = -8.5
    ne.ball_y = 0.0

    def action(step):
        a = _all_noop_actions(env)
        if step < 3:
            a[0] = KICK_MAX_ACTION
        return a

    return _save(
        _render_frames(env, num_frames=40, action_fn=action),
        "action_kick_only_no_translate",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only render the listed scenario function names")
    args = parser.parse_args()

    scenarios = [
        render_throwin_top,
        render_throwin_bottom,
        render_throwin_locked_kick,
        render_goal_for_blue,
        render_goal_for_red,
        render_wall_bounce_above_post,
        render_offside_caught,
        render_offside_ball_carrier_safe,
        render_offside_behind_ball_safe,
        render_offside_disabled_in_warm_start,
        render_field_scale_small,
        render_field_scale_full,
        render_red_corners,
        render_red_formation,
        render_move_forward,
        render_kick_only_no_translate,
    ]
    if args.only:
        scenarios = [s for s in scenarios if s.__name__ in set(args.only)]
        if not scenarios:
            raise SystemExit(f"No scenarios matched: {args.only}")

    for fn in scenarios:
        path = fn()
        print(f"saved {path}")


if __name__ == "__main__":
    main()
