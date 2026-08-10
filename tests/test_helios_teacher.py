"""Tests for the Helios-inspired scripted teacher."""

# pylint: disable=protected-access

from __future__ import annotations

import numpy as np

from puffer_soccer.envs.marl2d.core import (
    DISCRETE_ACTION_MOVE_FORWARD,
    DISCRETE_ACTION_ROTATE_LEFT,
    DISCRETE_ACTION_ROTATE_RIGHT,
    encode_discrete_kick_action,
)
from puffer_soccer.helios_teacher import HeliosTeacher, role_templates_for_team_size
from puffer_soccer.helios_teacher import CarrierContext, PlannedActionKind


def _state(
    positions: np.ndarray,
    rotations: np.ndarray,
    ball: tuple[float, float, float, float],
    *,
    blue_left: bool = True,
) -> dict[str, object]:
    """Build the public env state shape consumed by ``HeliosTeacher.actions``.

    The teacher deliberately uses only the dictionary returned by ``env.get_state``. Tests should
    mirror that public contract instead of reaching into private helper methods, because the
    whole point of the scripted teacher is that it can be reused from videos, evaluation, or
    imitation-data collection without relying on hidden simulator internals.
    """

    return {
        "positions": positions.astype(np.float32),
        "rotations": rotations.astype(np.float32),
        "ball": ball,
        "goals": (0, 0),
        "num_steps": 0,
        "blue_left": blue_left,
        "field_scale": 1.0,
        "spawn_difficulty": 1.0,
    }


def test_role_compression_keeps_attack_and_defense() -> None:
    """Small-team role lists should preserve a useful soccer spine.

    A direct prefix of the 11v11 Helios role list would give tiny teams only a goalie and
    defenders. The teacher instead compresses the role list so smaller games still contain
    defensive support and an attacker, which is important for warm-start videos and quick tests
    that cannot afford full 11v11 rollouts.
    """

    roles = role_templates_for_team_size(5)
    names = [role.name for role in roles]
    assert names[0] == "goalie"
    assert "center_forward" in names
    assert any("back" in name for name in names)
    assert any("half" in name for name in names)


def test_role_compression_rejects_invalid_team_size() -> None:
    """Invalid team sizes should fail before the teacher reaches the env."""

    try:
        role_templates_for_team_size(0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected players_per_team=0 to raise ValueError")

    try:
        role_templates_for_team_size(12)
    except ValueError:
        pass
    else:
        raise AssertionError("expected players_per_team=12 to raise ValueError")


def test_carrier_shoots_when_aligned_near_goal() -> None:
    """A carrier facing the opponent goal near the box should use the strongest kick."""

    teacher = HeliosTeacher(players_per_team=1)
    positions = np.array([[38.0, 0.0], [-25.0, 0.0]], dtype=np.float32)
    rotations = np.array([0.0, np.pi], dtype=np.float32)
    actions = teacher.actions(_state(positions, rotations, (38.0, 0.0, 0.0, 0.0)))
    assert int(actions[0]) == encode_discrete_kick_action(7)


def test_carrier_turns_before_kicking_when_misaligned() -> None:
    """The teacher should respect the env's body-facing kick direction."""

    teacher = HeliosTeacher(players_per_team=1)
    positions = np.array([[38.0, 0.0], [-25.0, 0.0]], dtype=np.float32)
    rotations = np.array([np.pi, np.pi], dtype=np.float32)
    actions = teacher.actions(_state(positions, rotations, (38.0, 0.0, 0.0, 0.0)))
    assert int(actions[0]) in (DISCRETE_ACTION_ROTATE_LEFT, DISCRETE_ACTION_ROTATE_RIGHT)


def test_nearest_player_chases_ball() -> None:
    """The best local interceptor should leave formation and move toward the ball."""

    teacher = HeliosTeacher(players_per_team=3)
    positions = np.array(
        [
            [-45.0, 0.0],
            [-10.0, 0.0],
            [20.0, 20.0],
            [40.0, 0.0],
            [20.0, -20.0],
            [25.0, 18.0],
        ],
        dtype=np.float32,
    )
    rotations = np.array([0.0, 0.0, 0.0, np.pi, np.pi, np.pi], dtype=np.float32)
    actions = teacher.actions(_state(positions, rotations, (-4.0, 0.0, 0.0, 0.0)))
    assert int(actions[1]) == DISCRETE_ACTION_MOVE_FORWARD


def test_planner_generates_direct_leading_and_through_passes() -> None:
    """The carrier planner should expose the main Helios pass families."""

    teacher = HeliosTeacher(players_per_team=3)
    context = CarrierContext(
        pos=np.array([0.0, 0.0], dtype=np.float32),
        rot=0.0,
        team_positions=np.array(
            [[0.0, 0.0], [18.0, -9.0], [26.0, 12.0]],
            dtype=np.float32,
        ),
        opponent_positions=np.array(
            [[-20.0, 30.0], [-22.0, -30.0], [-30.0, 0.0]],
            dtype=np.float32,
        ),
        ball_xy=np.array([0.0, 0.0], dtype=np.float32),
        ball_velocity=np.array([0.0, 0.0], dtype=np.float32),
        attack_direction=1.0,
        half_x=50.0,
        half_y=35.0,
    )
    kinds = {candidate.kind for candidate in teacher._pass_candidates(context)}
    assert PlannedActionKind.DIRECT_PASS in kinds
    assert PlannedActionKind.LEADING_PASS in kinds
    assert PlannedActionKind.THROUGH_PASS in kinds


def test_planner_generates_cross_from_wide_attack() -> None:
    """Wide attacking carriers should consider a cross into the goal mouth."""

    teacher = HeliosTeacher(players_per_team=3)
    context = CarrierContext(
        pos=np.array([34.0, 24.0], dtype=np.float32),
        rot=0.0,
        team_positions=np.array(
            [[34.0, 24.0], [42.0, 2.0], [30.0, -12.0]],
            dtype=np.float32,
        ),
        opponent_positions=np.array(
            [[5.0, -28.0], [-8.0, 24.0], [-20.0, 0.0]],
            dtype=np.float32,
        ),
        ball_xy=np.array([34.0, 24.0], dtype=np.float32),
        ball_velocity=np.array([0.0, 0.0], dtype=np.float32),
        attack_direction=1.0,
        half_x=50.0,
        half_y=35.0,
    )
    kinds = {candidate.kind for candidate in teacher._pass_candidates(context)}
    assert PlannedActionKind.CROSS in kinds
