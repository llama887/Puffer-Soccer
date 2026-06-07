"""Helios-inspired scripted soccer teacher for local evaluation and warm-starts."""

# pylint: disable=too-many-lines

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from puffer_soccer.envs.marl2d.core import (
    DISCRETE_ACTION_MOVE_BACKWARD,
    DISCRETE_ACTION_MOVE_FORWARD,
    DISCRETE_ACTION_NOOP,
    DISCRETE_ACTION_ROTATE_LEFT,
    DISCRETE_ACTION_ROTATE_RIGHT,
    encode_discrete_kick_action,
)


FIELD_HALF_X = 50.0
FIELD_HALF_Y = 35.0
GOAL_HALF_Y = 20.0
BALL_CONTROL_RADIUS = 4.4
BALL_CHASE_MARGIN = 8.0
TARGET_ANGLE_TOLERANCE = 0.35
KICK_ANGLE_TOLERANCE = 0.55
BALL_DECAY = 0.85
MAX_BALL_SPEED = 5.0
PLAYER_REACH_SPEED = 1.65


@dataclass(frozen=True)
class RoleTemplate:
    """Describe one formation slot in a compact, side-independent coordinate system.

    Helios uses named roles such as goalie, center back, side back, defensive half, offensive
    half, side forward, and center forward. Our environment does not expose player numbers,
    stamina, neck actions, or the RoboCup formation parser, so this small data class keeps only
    the transferable part: where a role wants to stand when the team is defending, neutral, or
    attacking. The x values are written in a team-local frame where positive x means toward the
    opponent goal. The teacher mirrors them automatically for the team that attacks left.

    The role name is intentionally retained for logs and future diagnostics. It does not affect
    action selection directly, but it makes it easy to inspect which compressed role assignment
    was used when running fewer than eleven players per side.
    """

    name: str
    defense: tuple[float, float]
    normal: tuple[float, float]
    offense: tuple[float, float]


@dataclass(frozen=True)
class HeliosTeacherConfig:  # pylint: disable=too-many-instance-attributes
    """Tune the scripted approximation without changing the teacher's public API.

    The teacher is meant to capture the spirit of Helios, not to be a perfect RoboCup port.
    These constants therefore control high-level soccer choices: how close the ball must be
    before an agent counts as the carrier, how quickly teammates collapse toward the ball, and
    which kick strengths represent shots, passes, dribbles, and clears in our discrete action
    space. Keeping them in one immutable config lets future experiments sweep the approximation
    while leaving the evaluation script and possible imitation-learning code unchanged.
    """

    ball_control_radius: float = BALL_CONTROL_RADIUS
    chase_margin: float = BALL_CHASE_MARGIN
    target_angle_tolerance: float = TARGET_ANGLE_TOLERANCE
    kick_angle_tolerance: float = KICK_ANGLE_TOLERANCE
    shot_kick_index: int = 7
    pass_kick_index: int = 4
    clear_kick_index: int = 6
    dribble_kick_index: int = 1
    receiver_reach_buffer: float = 0.5
    opponent_reach_buffer: float = 0.0
    defensive_block_distance: float = 8.0


@dataclass(frozen=True)
class TeamContext:  # pylint: disable=too-many-instance-attributes
    """Bundle one team's live state after slicing it out of the full environment state.

    Several teacher decisions need the same small group of values: team positions, opponent
    positions, the ball, field direction, and scaled field bounds. Passing this object keeps the
    helper API compact and makes it clear which values belong to team-level reasoning instead of
    per-agent actuation. It is intentionally immutable because each action decision should read
    from the same pre-step snapshot.
    """

    positions: np.ndarray
    rotations: np.ndarray
    opponent_positions: np.ndarray
    ball_xy: np.ndarray
    ball_velocity: np.ndarray
    attack_direction: float
    half_x: float
    half_y: float


@dataclass(frozen=True)
class CarrierContext:  # pylint: disable=too-many-instance-attributes
    """Bundle the geometry used by the ball-carrier planner.

    The carrier branch is the only place that considers shots, passes, clears, and dribbles.
    Keeping those inputs together makes it easier to replace the current cheap geometry with a
    deeper Helios-style lookahead later. The current implementation still stays simple and fast,
    but future code can add predicted ball states without widening every helper signature.
    """

    pos: np.ndarray
    rot: float
    team_positions: np.ndarray
    opponent_positions: np.ndarray
    ball_xy: np.ndarray
    ball_velocity: np.ndarray
    attack_direction: float
    half_x: float
    half_y: float


class PlannedActionKind(str, Enum):
    """Name the compact action categories used by the Helios-style planner.

    Open-source Helios represents planned actions as objects such as ``Shoot``, ``Pass``,
    ``Dribble``, and ``ClearBall`` inside an action-chain graph. Our environment has a much
    smaller action space, so this enum keeps only the tactical category and lets the executor
    translate the chosen candidate into a turn-or-kick command. Keeping these names explicit
    makes debugging videos easier because each candidate score can be traced back to a familiar
    soccer idea rather than to an anonymous target point.
    """

    SHOOT = "shoot"
    DIRECT_PASS = "direct_pass"
    LEADING_PASS = "leading_pass"
    THROUGH_PASS = "through_pass"
    CROSS = "cross"
    CLEAR = "clear"
    DRIBBLE = "dribble"
    HOLD = "hold"


@dataclass(frozen=True)
class PlannedAction:
    """Represent one candidate ball-carrier plan before conversion to a discrete action.

    The real Helios planner searches possible future states and returns the first action of the
    best chain. This data class is a lightweight version of that same concept. Each candidate
    stores the intended target point, the kick strength to use once aligned, and a scalar score
    from the teacher's field evaluator. A ``None`` kick index means the best action is to hold or
    turn without kicking. The score is intentionally simple and deterministic so we can inspect
    policy videos and unit tests without hidden stochastic choices.
    """

    kind: PlannedActionKind
    target: np.ndarray
    kick_index: int | None
    score: float


ROLE_LIBRARY: tuple[RoleTemplate, ...] = (
    RoleTemplate("goalie", (-48.0, 0.0), (-48.0, 0.0), (-47.0, 0.0)),
    RoleTemplate("center_back_left", (-32.0, -8.0), (-24.0, -8.0), (-10.0, -10.0)),
    RoleTemplate("center_back_right", (-32.0, 8.0), (-24.0, 8.0), (-10.0, 10.0)),
    RoleTemplate("side_back_left", (-28.0, -22.0), (-18.0, -23.0), (0.0, -24.0)),
    RoleTemplate("side_back_right", (-28.0, 22.0), (-18.0, 23.0), (0.0, 24.0)),
    RoleTemplate("defensive_half", (-16.0, 0.0), (-6.0, 0.0), (12.0, 0.0)),
    RoleTemplate("offensive_half_left", (-4.0, -14.0), (8.0, -13.0), (26.0, -15.0)),
    RoleTemplate("offensive_half_right", (-4.0, 14.0), (8.0, 13.0), (26.0, 15.0)),
    RoleTemplate("side_forward_left", (10.0, -24.0), (22.0, -24.0), (38.0, -23.0)),
    RoleTemplate("side_forward_right", (10.0, 24.0), (22.0, 24.0), (38.0, 23.0)),
    RoleTemplate("center_forward", (16.0, 0.0), (30.0, 0.0), (42.0, 0.0)),
)


def role_templates_for_team_size(players_per_team: int) -> tuple[RoleTemplate, ...]:
    """Return a compressed Helios-style role list for the requested team size.

    Helios assumes exactly eleven players, but this project often trains and debugs smaller
    games. A naive prefix of the eleven-player formation would create goalie-heavy defensive
    teams with no forwards, which is not a useful teacher. This helper instead samples the
    canonical role list across the field so every team size keeps the most important tactical
    structure: a keeper if there is room, at least one defender, at least one support player,
    and at least one attacker.

    The output order matches the environment's fixed player-slot order. That makes the teacher
    deterministic and simple to compare across runs, while still allowing fewer-player tests to
    exercise the same chase, support, and ball-carrier logic used in full 11v11.
    """

    if players_per_team < 1 or players_per_team > len(ROLE_LIBRARY):
        raise ValueError("players_per_team must be in [1, 11]")
    if players_per_team == len(ROLE_LIBRARY):
        return ROLE_LIBRARY
    if players_per_team == 1:
        return (ROLE_LIBRARY[-1],)

    selected = [ROLE_LIBRARY[0], ROLE_LIBRARY[-1]]
    if players_per_team >= 3:
        selected.insert(1, ROLE_LIBRARY[5])
    if players_per_team >= 4:
        selected.insert(1, ROLE_LIBRARY[2])
    if players_per_team >= 5:
        selected.insert(-1, ROLE_LIBRARY[6])
    if players_per_team >= 6:
        selected.insert(2, ROLE_LIBRARY[3])
    if players_per_team >= 7:
        selected.insert(-1, ROLE_LIBRARY[8])
    if players_per_team >= 8:
        selected.insert(3, ROLE_LIBRARY[4])
    if players_per_team >= 9:
        selected.insert(-1, ROLE_LIBRARY[7])
    if players_per_team >= 10:
        selected.insert(-1, ROLE_LIBRARY[9])
    return tuple(selected[:players_per_team])


class HeliosTeacher:  # pylint: disable=too-few-public-methods
    """Choose discrete actions with a compact Helios-inspired decision hierarchy.

    The teacher reads the scalar environment's public ``get_state`` dictionary and returns one
    action per player. It intentionally does not use private C structs, observations, or policy
    network state. That keeps it suitable for three different future uses: recording baseline
    videos, serving as a frozen scripted opponent, or producing imitation targets for a short
    warm-start phase.

    The behavior copies strategic ideas rather than source code. Off-ball players move toward
    role targets that shift between defense, neutral play, and offense. The closest useful
    teammate breaks shape to chase the ball, matching Helios' intercept-first behavior. A player
    close to the ball acts as the carrier and selects among shot, pass, clear, dribble, or hold
    using cheap geometry. Because our action space has no direct "kick to point" command, the
    teacher first turns toward a chosen target and only kicks once roughly aligned.
    """

    def __init__(
        self,
        players_per_team: int,
        config: HeliosTeacherConfig | None = None,
    ) -> None:
        self.players_per_team = int(players_per_team)
        self.config = HeliosTeacherConfig() if config is None else config
        self.roles = role_templates_for_team_size(self.players_per_team)

    def actions(self, state: dict[str, Any]) -> np.ndarray:
        """Return one discrete action for every blue and red player in ``state``.

        The environment state is full-field and side-neutral: blue may attack right or left
        depending on reset options. This method computes each team's attack direction from
        ``blue_left`` and mirrors all role targets and tactical tests through that direction.
        The returned array has dtype ``int32`` because the native binding expects integer action
        ids and the existing training/video loops use that dtype for policy actions.

        If future reset modes mask inactive player slots by zeroing observations, this method
        will still output valid actions for all slots. The native environment ignores inactive
        agents, so keeping a dense action array avoids adding ragged team-size handling here.
        """

        positions = np.asarray(state["positions"], dtype=np.float32)
        rotations = np.asarray(state["rotations"], dtype=np.float32)
        ball = np.asarray(state["ball"], dtype=np.float32)
        field_scale = float(state.get("field_scale", 1.0))
        blue_attack = 1.0 if bool(state["blue_left"]) else -1.0
        actions = np.empty(self.players_per_team * 2, dtype=np.int32)

        actions[: self.players_per_team] = self._team_actions(
            positions,
            rotations,
            ball,
            start=0,
            attack_direction=blue_attack,
            field_scale=field_scale,
        )
        actions[self.players_per_team :] = self._team_actions(
            positions,
            rotations,
            ball,
            start=self.players_per_team,
            attack_direction=-blue_attack,
            field_scale=field_scale,
        )
        return actions

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def _team_actions(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        ball: np.ndarray,
        *,
        start: int,
        attack_direction: float,
        field_scale: float,
    ) -> np.ndarray:
        """Compute actions for one team after orienting the field into attack coordinates.

        The implementation separates team-level reasoning from per-agent actuation. First it
        estimates possession and the team's best ball chaser using only distances to the live
        ball. Then each player either acts as a carrier, chaser, or role player. This mirrors
        the Helios structure while keeping the code fast: all team positions are sliced once and
        no allocation-heavy search tree is built on every environment step.
        """

        end = start + self.players_per_team
        team_positions = positions[start:end]
        team_rotations = rotations[start:end]
        opponent_positions = (
            positions[self.players_per_team :]
            if start == 0
            else positions[: self.players_per_team]
        )
        ball_xy = ball[:2]
        ball_velocity = ball[2:4]
        context = TeamContext(
            positions=team_positions,
            rotations=team_rotations,
            opponent_positions=opponent_positions,
            ball_xy=ball_xy,
            ball_velocity=ball_velocity,
            attack_direction=attack_direction,
            half_x=FIELD_HALF_X * field_scale,
            half_y=FIELD_HALF_Y * field_scale,
        )
        team_dists = np.linalg.norm(team_positions - ball_xy, axis=1)
        opp_dists = np.linalg.norm(opponent_positions - ball_xy, axis=1)
        chaser = int(np.argmin(team_dists))
        nearest_team = float(team_dists[chaser])
        nearest_opp = float(np.min(opp_dists))
        possession = self._estimate_possession(nearest_team, nearest_opp)

        team_actions = np.empty(self.players_per_team, dtype=np.int32)
        for player_idx in range(self.players_per_team):
            global_idx = start + player_idx
            pos = team_positions[player_idx]
            rot = float(team_rotations[player_idx])
            if team_dists[player_idx] <= self.config.ball_control_radius:
                team_actions[player_idx] = self._carrier_action(
                    CarrierContext(
                        pos=pos,
                        rot=rot,
                        team_positions=context.positions,
                        opponent_positions=context.opponent_positions,
                        ball_xy=context.ball_xy,
                        ball_velocity=context.ball_velocity,
                        attack_direction=context.attack_direction,
                        half_x=context.half_x,
                        half_y=context.half_y,
                    )
                )
            elif player_idx == chaser and nearest_team <= nearest_opp + self.config.chase_margin:
                team_actions[player_idx] = self._move_toward(pos, rot, ball_xy)
            else:
                role_target = self._role_target(
                    player_idx,
                    context,
                    possession,
                )
                target = self._support_target(
                    player_idx,
                    context,
                    role_target,
                    possession,
                    chaser,
                )
                if possession == "defense":
                    target = self._defensive_target(
                        player_idx,
                        context,
                        target,
                    )
                team_actions[player_idx] = self._move_toward(positions[global_idx], rot, target)
        return team_actions

    def _estimate_possession(self, nearest_team: float, nearest_opp: float) -> str:
        """Classify possession from ball-race distances for formation selection.

        Helios uses intercept tables to decide whether the team is attacking, defending, or in a
        balanced state. Our public state does not expose the native intercept calculation, so the
        teacher approximates the same idea with current distance to the ball. A clear distance
        advantage moves the team into offense; a clear deficit moves it into defense; otherwise
        players keep a neutral shape.
        """

        if nearest_team + 2.0 < nearest_opp:
            return "offense"
        if nearest_opp + 2.0 < nearest_team:
            return "defense"
        return "normal"

    def _role_target(
        self,
        player_idx: int,
        context: TeamContext,
        possession: str,
    ) -> np.ndarray:
        """Return the live formation target for one player slot.

        Static role points are only a starting point. The target also shifts with the ball so
        the team slides toward the active side of the field, which is one of the most visible
        strategic traits in Helios formations. X movement is conservative for defenders and
        stronger for midfielders/forwards, preventing the whole team from collapsing into one
        line while still keeping support close enough for passes and second balls.
        """

        role = self.roles[player_idx]
        base = {
            "defense": role.defense,
            "offense": role.offense,
            "normal": role.normal,
        }[possession]
        predicted_ball = self._predict_ball_position(context.ball_xy, context.ball_velocity, 5)
        local_ball_x = float(predicted_ball[0] * context.attack_direction)
        local_ball_y = float(predicted_ball[1])
        role_depth = base[0]
        slide_x = 0.18 * local_ball_x
        if role_depth > 15.0:
            slide_x = 0.28 * local_ball_x
        elif role_depth < -20.0:
            slide_x = 0.08 * local_ball_x
        target_local_x = np.clip(
            base[0] + slide_x,
            -context.half_x + 2.0,
            context.half_x - 2.0,
        )
        target_y = np.clip(
            base[1] + 0.25 * local_ball_y,
            -context.half_y + 2.0,
            context.half_y - 2.0,
        )
        if role.name == "goalie":
            return self._goalie_target(context)
        return np.array(
            [target_local_x * context.attack_direction, target_y],
            dtype=np.float32,
        )

    def _goalie_target(self, context: TeamContext) -> np.ndarray:
        """Return a goalie position that protects the goal mouth and shooting angle.

        Open-source Helios has a dedicated goalie behavior that tracks dangerous shot angles,
        guards posts, and only chases when the ball is safe to attack. Our environment has no
        catch command or goalie-only mechanics, but the same strategic idea still transfers:
        the keeper should sit close to its own goal line, slide along the goal mouth with the
        ball, and bias toward the near post when the ball is wide. This target is separate from
        ordinary role movement so goalie behavior does not become just another defender slot.
        """

        local_ball_x = float(context.ball_xy[0] * context.attack_direction)
        local_ball_y = float(context.ball_xy[1])
        own_goal_x = -context.half_x + 2.0
        y_from_angle = 0.45 * local_ball_y
        if local_ball_x < -context.half_x + 16.0 and abs(local_ball_y) > GOAL_HALF_Y:
            y_from_angle = np.sign(local_ball_y) * (GOAL_HALF_Y - 1.5)
        target_y = np.clip(y_from_angle, -GOAL_HALF_Y + 1.0, GOAL_HALF_Y - 1.0)
        return np.array(
            [own_goal_x * context.attack_direction, target_y],
            dtype=np.float32,
        )

    def _defensive_target(
        self,
        player_idx: int,
        context: TeamContext,
        role_target: np.ndarray,
    ) -> np.ndarray:
        """Return a defensive target that blocks central lanes before chasing blindly.

        RoboCup Helios has explicit tackle and emergency movement behaviors. This environment
        does not expose a tackle action, so the closest useful translation is spatial: defenders
        step into the line between ball and goal, midfielders screen passing lanes, and forwards
        stay high only when their role says so. This gives the teacher a way to challenge the
        learned policy without inventing actions the environment cannot execute.
        """

        role = self.roles[player_idx]
        if role.name == "goalie" or "forward" in role.name:
            return role_target

        own_goal = np.array(
            [-context.attack_direction * context.half_x, 0.0],
            dtype=np.float32,
        )
        ball_to_goal = own_goal - context.ball_xy
        norm = float(np.linalg.norm(ball_to_goal))
        if norm < 1e-6:
            return role_target
        unit = ball_to_goal / norm
        block_distance = self.config.defensive_block_distance
        if "back" in role.name:
            block_distance += 5.0
        block_target = context.ball_xy + unit * block_distance
        side_offset = -7.0 if player_idx % 2 == 0 else 7.0
        block_target[1] += side_offset if "back" in role.name else 0.45 * side_offset
        block_target[0] = np.clip(block_target[0], -context.half_x + 3.0, context.half_x - 3.0)
        block_target[1] = np.clip(block_target[1], -context.half_y + 3.0, context.half_y - 3.0)
        return (0.35 * role_target + 0.65 * block_target).astype(np.float32)

    def _support_target(
        self,
        player_idx: int,
        context: TeamContext,
        role_target: np.ndarray,
        possession: str,
        chaser_idx: int,
    ) -> np.ndarray:
        """Move off-ball players toward useful pass and second-ball support spaces.

        In real Helios, role movement is not just a static formation lookup. Players react to
        the ball carrier by offering direct passes, stretching the defense, and preparing for
        crosses or loose balls. This helper keeps that idea while staying simple: defenders keep
        more of their role shape, midfielders slide into diagonal support, and forwards move
        into through-ball or cross-receiving lanes. The final target is blended with the base
        role point so the team does not collapse into one cluster around the ball.
        """

        role = self.roles[player_idx]
        if role.name == "goalie" or possession == "defense":
            return role_target

        local_ball_x = float(context.ball_xy[0] * context.attack_direction)
        local_ball_y = float(context.ball_xy[1])
        role_local_x = float(role_target[0] * context.attack_direction)
        if player_idx == chaser_idx:
            return role_target

        if "forward" in role.name:
            support_local_x = np.clip(
                max(role_local_x, local_ball_x + 14.0),
                -context.half_x + 6.0,
                context.half_x - 7.0,
            )
            if abs(local_ball_y) > context.half_y * 0.35:
                support_y = -0.35 * local_ball_y
            elif "left" in role.name:
                support_y = -0.55 * context.half_y
            elif "right" in role.name:
                support_y = 0.55 * context.half_y
            else:
                support_y = 0.0
            blend = 0.65
        elif "half" in role.name:
            support_local_x = np.clip(
                local_ball_x + 8.0,
                -context.half_x + 8.0,
                context.half_x - 12.0,
            )
            support_y = local_ball_y + (-9.0 if player_idx % 2 == 0 else 9.0)
            blend = 0.45
        else:
            support_local_x = np.clip(
                min(role_local_x, local_ball_x - 12.0),
                -context.half_x + 5.0,
                context.half_x - 15.0,
            )
            support_y = 0.6 * float(role_target[1]) + 0.2 * local_ball_y
            blend = 0.25

        support = np.array(
            [
                support_local_x * context.attack_direction,
                np.clip(support_y, -context.half_y + 3.0, context.half_y - 3.0),
            ],
            dtype=np.float32,
        )
        return ((1.0 - blend) * role_target + blend * support).astype(np.float32)

    def _carrier_action(self, context: CarrierContext) -> int:
        """Choose the first executable step of the best Helios-style carrier candidate.

        This is the main place where the approximation now moves closer to real Helios. Instead
        of a fixed priority ladder, the carrier generates candidate shots, direct passes,
        leading passes, through passes, crosses, clears, and dribbles. Each candidate is scored
        by a small field evaluator that rewards goal threat, forward progress, safe lanes,
        receiver usefulness, and pressure relief. The chosen candidate is then translated into
        our environment's limited action space: turn toward the target until aligned, then kick
        forward with the candidate's strength.
        """

        candidate = self._best_planned_action(context)
        if candidate.kick_index is None:
            return self._move_toward(context.pos, context.rot, candidate.target)
        return self._kick_or_turn(
            context.pos,
            context.rot,
            candidate.target,
            candidate.kick_index,
        )

    def _best_planned_action(self, context: CarrierContext) -> PlannedAction:
        """Return the highest-scoring generated carrier action for one state snapshot.

        The real Helios action-chain graph can search several actions into the future. Our
        version is deliberately shallower, because the native API only lets the bot choose a
        single discrete action per frame. Even so, generating a diverse candidate set captures
        the important Helios habit: the ball carrier compares several soccer ideas rather than
        blindly following one priority rule.
        """

        candidates = [
            *self._shoot_candidates(context),
            *self._pass_candidates(context),
            *self._clear_candidates(context),
            *self._dribble_candidates(context),
        ]
        if not candidates:
            candidates.append(
                PlannedAction(
                    PlannedActionKind.HOLD,
                    context.pos.astype(np.float32, copy=True),
                    None,
                    -1_000.0,
                )
            )
        return max(candidates, key=lambda action: action.score)

    def _shoot_candidates(self, context: CarrierContext) -> list[PlannedAction]:
        """Generate goal-directed shots when the carrier has a plausible shooting angle."""

        local_x = float(context.pos[0] * context.attack_direction)
        goal_center = np.array(
            [context.attack_direction * (context.half_x + 3.0), 0.0],
            dtype=np.float32,
        )
        distance_to_goal = float(np.linalg.norm(goal_center - context.pos))
        angle_width = self._goal_angle_width(
            context.pos,
            context.attack_direction,
            context.half_x,
        )
        lane_gap = self._min_distance_to_segment(
            context.opponent_positions,
            context.pos,
            goal_center,
        )
        if local_x < 0.0 and angle_width < 0.18:
            return []
        if lane_gap < 2.5 and local_x < context.half_x - 12.0:
            return []
        score = (
            180.0
            + 2.0 * local_x
            + 90.0 * angle_width
            + 3.0 * lane_gap
            - 1.2 * distance_to_goal
            - 1.4 * abs(float(context.pos[1]))
        )
        return [
            PlannedAction(
                PlannedActionKind.SHOOT,
                goal_center,
                self.config.shot_kick_index,
                score,
            )
        ]

    def _pass_candidates(self, context: CarrierContext) -> list[PlannedAction]:
        """Generate direct, leading, through, and cross pass candidates.

        Helios spends much of its attacking intelligence on pass generation. This approximation
        keeps the same categories while simplifying the physics. Direct passes target the
        receiver's current position, leading passes hit space just ahead of a teammate, through
        passes target space behind the defense, and crosses aim from wide attacking areas toward
        the central goal mouth.
        """

        candidates: list[PlannedAction] = []
        carrier_local_x = float(context.pos[0] * context.attack_direction)
        for receiver in context.team_positions:
            if float(np.linalg.norm(receiver - context.pos)) < 6.0:
                continue
            receiver_local_x = float(receiver[0] * context.attack_direction)
            if receiver_local_x < carrier_local_x - 10.0:
                continue
            candidates.extend(
                self._receiver_pass_candidates(
                    context,
                    receiver,
                    carrier_local_x,
                    receiver_local_x,
                )
            )

        local_y = float(context.pos[1])
        if (
            carrier_local_x > context.half_x - 24.0
            and abs(local_y) > context.half_y * 0.35
        ):
            cross_target = np.array(
                [
                    context.attack_direction * (context.half_x - 8.0),
                    -0.25 * local_y,
                ],
                dtype=np.float32,
            )
            receiver = self._best_cross_receiver(context)
            if receiver is not None:
                cross_target = receiver
            cross_kick_index = self._kick_index_for_pass_distance(
                float(np.linalg.norm(cross_target - context.pos))
            )
            lane_gap = self._min_distance_to_segment(
                context.opponent_positions,
                context.pos,
                cross_target,
            )
            receiver_step, opponent_step = self._reach_steps_for_pass(
                context,
                cross_target,
                cross_kick_index,
            )
            if self._pass_is_safe(receiver_step, opponent_step, lane_gap):
                candidates.append(
                    PlannedAction(
                        PlannedActionKind.CROSS,
                        cross_target,
                        cross_kick_index,
                        (
                            115.0
                            + 2.0 * lane_gap
                            + 0.8 * carrier_local_x
                            + 1.5 * (opponent_step - receiver_step)
                        ),
                    )
                )
        return candidates

    def _receiver_pass_candidates(
        self,
        context: CarrierContext,
        receiver: np.ndarray,
        carrier_local_x: float,
        receiver_local_x: float,
    ) -> list[PlannedAction]:
        """Generate pass variants for one possible receiver.

        This helper is the compact substitute for Helios' strict pass generator. It evaluates
        the same receiver through several target points because in our env a useful pass is more
        often "kick into reachable space" than "kick exactly to the teammate's current feet."
        """

        targets = [
            (
                PlannedActionKind.DIRECT_PASS,
                receiver.astype(np.float32, copy=True),
                self._kick_index_for_pass_distance(float(np.linalg.norm(receiver - context.pos))),
            ),
            (
                PlannedActionKind.LEADING_PASS,
                leading_target := np.array(
                    [
                        receiver[0] + context.attack_direction * 7.0,
                        receiver[1],
                    ],
                    dtype=np.float32,
                ),
                self._kick_index_for_pass_distance(
                    float(np.linalg.norm(leading_target - context.pos))
                ),
            ),
            (
                PlannedActionKind.THROUGH_PASS,
                through_target := np.array(
                    [
                        min(
                            context.half_x - 4.0,
                            receiver_local_x + 12.0,
                        )
                        * context.attack_direction,
                        receiver[1] * 0.75,
                    ],
                    dtype=np.float32,
                ),
                self._kick_index_for_pass_distance(
                    float(np.linalg.norm(through_target - context.pos))
                ),
            ),
        ]
        candidates: list[PlannedAction] = []
        for kind, target, kick_index in targets:
            if abs(float(target[0])) > context.half_x or abs(float(target[1])) > context.half_y:
                continue
            lane_gap = self._min_distance_to_segment(
                context.opponent_positions,
                context.pos,
                target,
            )
            receiver_step, opponent_step = self._reach_steps_for_pass(
                context,
                target,
                kick_index,
            )
            if not self._pass_is_safe(receiver_step, opponent_step, lane_gap):
                continue
            target_local_x = float(target[0] * context.attack_direction)
            forward_gain = target_local_x - carrier_local_x
            receiver_dist = float(np.linalg.norm(receiver - target))
            pass_dist = float(np.linalg.norm(target - context.pos))
            pressure_at_target = self._nearest_distance(context.opponent_positions, target)
            timing_margin = opponent_step - receiver_step
            score = (
                70.0
                + 1.9 * forward_gain
                + 2.6 * lane_gap
                + 0.8 * pressure_at_target
                + 2.4 * timing_margin
                - 0.18 * pass_dist
                - 0.35 * receiver_dist
                + 0.45 * abs(float(target[1] - context.pos[1]))
            )
            if kind == PlannedActionKind.THROUGH_PASS:
                score += 12.0 if target_local_x > 0.0 else -15.0
            if kind == PlannedActionKind.LEADING_PASS:
                score += 5.0
            candidates.append(PlannedAction(kind, target, kick_index, score))
        return candidates

    def _best_cross_receiver(self, context: CarrierContext) -> np.ndarray | None:
        """Return a central attacking teammate suitable for a cross target.

        Open-source Helios has a dedicated cross generator rather than always crossing to a
        fixed point. This helper maps that idea to our environment: if a teammate is already
        near the goal mouth, cross to that player or the small space around them. If no one is
        central enough, fall back to the default open-space cross target.
        """

        best_score = -np.inf
        best_receiver: np.ndarray | None = None
        for receiver in context.team_positions:
            local_x = float(receiver[0] * context.attack_direction)
            if local_x < context.half_x - 18.0 or abs(float(receiver[1])) > GOAL_HALF_Y:
                continue
            pressure = self._nearest_distance(context.opponent_positions, receiver)
            score = local_x + 1.5 * pressure - 0.5 * abs(float(receiver[1]))
            if score > best_score:
                best_score = score
                best_receiver = receiver.astype(np.float32, copy=True)
        return best_receiver

    def _kick_index_for_pass_distance(self, distance: float) -> int:
        """Choose a discrete kick strength for the requested pass distance.

        Real Helios computes pass speed from the distance and ball decay. Our previous teacher
        used one fixed medium kick for all passes, which made long crosses and through balls
        arrive too slowly or not at all. This helper keeps the same spirit with the available
        eight discrete strengths.
        """

        if distance > 28.0:
            return self.config.shot_kick_index
        if distance > 22.0:
            return self.config.clear_kick_index
        if distance > 14.0:
            return max(self.config.pass_kick_index, 5)
        return self.config.pass_kick_index

    def _pass_is_safe(
        self,
        receiver_step: float,
        opponent_step: float,
        lane_gap: float,
    ) -> bool:
        """Return whether a pass target is likely to be reached by our team first.

        This is the key approximation of Helios' strict pass checker. The old teacher only
        looked at static line clearance. That misses many turnovers because a pass can have a
        clear lane but still arrive slowly enough for an opponent to win the race. This check
        combines line clearance with a reach-time margin so a pass must be both spatially and
        temporally safe.
        """

        if lane_gap < 2.8:
            return False
        if lane_gap > 10.0:
            return receiver_step <= opponent_step + 0.25
        return receiver_step + self.config.receiver_reach_buffer <= opponent_step

    def _reach_steps_for_pass(
        self,
        context: CarrierContext,
        target: np.ndarray,
        kick_index: int,
    ) -> tuple[float, float]:
        """Estimate teammate and opponent arrival times at a kicked pass target.

        The native action space does not let us choose an exact kick speed, but kick strengths
        are monotonic. We map the chosen strength to an approximate initial ball speed and use
        the environment's ball decay to estimate when the ball reaches the target. Players are
        modeled with a simple constant reach speed plus a one-step reaction penalty. The result
        is not a full physics simulator, but it captures the main Helios idea: compare who can
        arrive first before calling something a safe pass.
        """

        ball_steps = self._ball_travel_steps(
            context.pos,
            target,
            kick_index,
        )
        teammate_steps = self._player_reach_steps(context.team_positions, target)
        opponent_steps = self._player_reach_steps(context.opponent_positions, target)
        receiver_step = max(ball_steps, teammate_steps)
        opponent_step = max(0.0, opponent_steps - self.config.opponent_reach_buffer)
        return receiver_step, opponent_step

    def _ball_travel_steps(
        self,
        start: np.ndarray,
        target: np.ndarray,
        kick_index: int,
    ) -> float:
        """Approximate how many environment frames a kicked ball needs to reach a target."""

        distance = float(np.linalg.norm(target - start))
        speed = MAX_BALL_SPEED * max(0.15, (kick_index + 1) / 8.0)
        traveled = 0.0
        for step in range(1, 40):
            traveled += speed
            if traveled >= distance:
                return float(step)
            speed *= BALL_DECAY
            if speed < 0.05:
                break
        return 40.0

    def _player_reach_steps(self, positions: np.ndarray, target: np.ndarray) -> float:
        """Estimate the fastest player arrival time to a target point."""

        if positions.size == 0:
            return float("inf")
        nearest = float(np.min(np.linalg.norm(positions - target, axis=1)))
        return 1.0 + nearest / PLAYER_REACH_SPEED

    def _predict_ball_position(
        self,
        ball_xy: np.ndarray,
        ball_velocity: np.ndarray,
        steps: int,
    ) -> np.ndarray:
        """Predict the ball location after repeated velocity decay."""

        pos = ball_xy.astype(np.float32, copy=True)
        vel = ball_velocity.astype(np.float32, copy=True)
        for _ in range(max(0, steps)):
            pos = pos + vel
            vel = vel * BALL_DECAY
        return pos

    def _clear_candidates(self, context: CarrierContext) -> list[PlannedAction]:
        """Generate defensive clears when the carrier is under pressure near its own goal."""

        local_x = float(context.pos[0] * context.attack_direction)
        pressure = self._nearest_distance(context.opponent_positions, context.pos)
        if local_x > -context.half_x + 24.0 and pressure > 7.5:
            return []
        targets = [
            np.array(
                [context.attack_direction * 0.10 * context.half_x, 0.62 * context.half_y],
                dtype=np.float32,
            ),
            np.array(
                [context.attack_direction * 0.10 * context.half_x, -0.62 * context.half_y],
                dtype=np.float32,
            ),
            np.array(
                [context.attack_direction * 0.35 * context.half_x, 0.0],
                dtype=np.float32,
            ),
        ]
        candidates: list[PlannedAction] = []
        for target in targets:
            lane_gap = self._min_distance_to_segment(
                context.opponent_positions,
                context.pos,
                target,
            )
            score = 85.0 - 2.0 * local_x - 3.0 * pressure + 1.8 * lane_gap
            candidates.append(
                PlannedAction(
                    PlannedActionKind.CLEAR,
                    target,
                    self.config.clear_kick_index,
                    score,
                )
            )
        return candidates

    def _dribble_candidates(self, context: CarrierContext) -> list[PlannedAction]:
        """Generate short dribbles into nearby open space.

        Helios has separate short-dribble, simple-dribble, and self-pass generators. Our action
        space cannot express those exactly, so we approximate them as weak forward kicks toward
        several nearby lanes and score the lane by progress and opponent distance.
        """

        local_x = float(context.pos[0] * context.attack_direction)
        candidates: list[PlannedAction] = []
        for forward_step, y_step in ((9.0, 0.0), (8.0, 7.0), (8.0, -7.0), (14.0, 0.0)):
            target = np.array(
                [
                    np.clip(
                        context.pos[0] + context.attack_direction * forward_step,
                        -context.half_x + 2.0,
                        context.half_x - 2.0,
                    ),
                    np.clip(
                        context.pos[1] + y_step,
                        -context.half_y + 2.0,
                        context.half_y - 2.0,
                    ),
                ],
                dtype=np.float32,
            )
            target_pressure = self._nearest_distance(context.opponent_positions, target)
            lane_gap = self._min_distance_to_segment(
                context.opponent_positions,
                context.pos,
                target,
            )
            target_local_x = float(target[0] * context.attack_direction)
            score = (
                55.0
                + 1.2 * (target_local_x - local_x)
                + 2.3 * target_pressure
                + lane_gap
                - 0.4 * abs(float(target[1]))
            )
            candidates.append(
                PlannedAction(
                    PlannedActionKind.DRIBBLE,
                    target,
                    self.config.dribble_kick_index,
                    score,
                )
            )
        return candidates

    # pylint: disable=too-many-arguments,too-many-locals
    def _best_pass_target(
        self,
        carrier_pos: np.ndarray,
        team_positions: np.ndarray,
        opponent_positions: np.ndarray,
        attack_direction: float,
        half_x: float,
    ) -> np.ndarray | None:
        """Pick a simple direct-pass target that is useful and not obviously blocked.

        Helios' direct pass generator rejects stale holders, inaccurate players, and lanes that
        opponents can intercept. In this environment we only have exact positions, so a pass is
        considered useful when the receiver is not the carrier, is not behind the carrier by a
        large amount, and the closest opponent to the line segment is far enough away. The score
        prefers forward progress, width, and open lanes. Returning ``None`` tells the carrier to
        dribble, clear, or shoot instead.
        """

        best_score = -np.inf
        best_target: np.ndarray | None = None
        carrier_local_x = float(carrier_pos[0] * attack_direction)
        for receiver in team_positions:
            dist = float(np.linalg.norm(receiver - carrier_pos))
            if dist < 6.0:
                continue
            receiver_local_x = float(receiver[0] * attack_direction)
            if receiver_local_x < carrier_local_x - 8.0:
                continue
            lane_gap = self._min_distance_to_segment(opponent_positions, carrier_pos, receiver)
            if lane_gap < 4.5:
                continue
            forward_gain = receiver_local_x - carrier_local_x
            width = abs(float(receiver[1] - carrier_pos[1]))
            score = 1.5 * forward_gain + 0.4 * width + lane_gap - 0.1 * dist
            if receiver_local_x > half_x - 12.0:
                score += 15.0
            if score > best_score:
                best_score = score
                best_target = receiver.astype(np.float32, copy=True)
        return best_target

    def _goal_angle_width(
        self,
        pos: np.ndarray,
        attack_direction: float,
        half_x: float,
    ) -> float:
        """Approximate how open the goal mouth looks from one ball-carrier position."""

        upper = np.array([attack_direction * half_x, GOAL_HALF_Y], dtype=np.float32)
        lower = np.array([attack_direction * half_x, -GOAL_HALF_Y], dtype=np.float32)
        a0 = float(np.arctan2(upper[1] - pos[1], upper[0] - pos[0]))
        a1 = float(np.arctan2(lower[1] - pos[1], lower[0] - pos[0]))
        return abs(_wrap_angle(a0 - a1))

    def _nearest_distance(self, points: np.ndarray, target: np.ndarray) -> float:
        """Return the nearest player distance to a target point."""

        if points.size == 0:
            return float("inf")
        return float(np.min(np.linalg.norm(points - target, axis=1)))

    def _move_toward(self, pos: np.ndarray, rot: float, target: np.ndarray) -> int:
        """Turn or move toward ``target`` using the environment's one-intent action space.

        Movement in this environment is body-relative: an agent cannot rotate and translate in
        the same step. The teacher therefore chooses rotation until the target is inside a small
        angular cone, then moves forward. If the target is almost directly behind the player,
        moving backward is faster than turning all the way around and matches the simple action
        set already available to learned policies.
        """

        delta = target - pos
        dist = float(np.linalg.norm(delta))
        if dist < 1.0:
            return DISCRETE_ACTION_NOOP
        angle_error = _wrap_angle(float(np.arctan2(delta[1], delta[0])) - rot)
        if abs(angle_error) < self.config.target_angle_tolerance:
            return DISCRETE_ACTION_MOVE_FORWARD
        if abs(angle_error) > np.pi - self.config.target_angle_tolerance:
            return DISCRETE_ACTION_MOVE_BACKWARD
        return DISCRETE_ACTION_ROTATE_LEFT if angle_error > 0.0 else DISCRETE_ACTION_ROTATE_RIGHT

    def _kick_or_turn(
        self,
        pos: np.ndarray,
        rot: float,
        target: np.ndarray,
        kick_index: int,
    ) -> int:
        """Kick with the requested strength once the carrier roughly faces ``target``.

        The native action API exposes kick strength but not kick angle, so direction comes from
        the player's current body rotation. Turning before kicking is slower than an ideal
        planner, but it makes the scripted behavior compatible with the same action space the
        PPO policy must learn. This also creates useful imitation targets: align first, then
        kick, instead of asking the learner to copy impossible one-step pass commands.
        """

        delta = target - pos
        if float(np.linalg.norm(delta)) < 1e-6:
            return DISCRETE_ACTION_NOOP
        angle_error = _wrap_angle(float(np.arctan2(delta[1], delta[0])) - rot)
        if abs(angle_error) <= self.config.kick_angle_tolerance:
            return encode_discrete_kick_action(kick_index)
        return DISCRETE_ACTION_ROTATE_LEFT if angle_error > 0.0 else DISCRETE_ACTION_ROTATE_RIGHT

    def _min_distance_to_segment(
        self,
        points: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> float:
        """Return the nearest opponent distance to a proposed pass lane.

        This small vectorized geometry helper is the teacher's replacement for Helios'
        pass-checker. It treats the pass as a line segment from carrier to receiver and computes
        how close every opponent is to that segment. A larger value means a safer direct pass.
        The vectorized implementation avoids Python loops in the per-step carrier decision and
        stays cheap enough for video rendering or future warm-start data generation.
        """

        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-6:
            return 0.0
        rel = points - start
        t = np.clip((rel @ segment) / denom, 0.0, 1.0)
        closest = start + t[:, None] * segment
        return float(np.min(np.linalg.norm(points - closest, axis=1)))


def _wrap_angle(angle: float) -> float:
    """Normalize an angle difference to the ``[-pi, pi]`` interval.

    Every movement and kick decision compares a desired target angle with the agent's current
    body rotation. Using one normalization helper avoids subtle left/right bugs at the ``pi``
    boundary, which matter in this discrete action space because one wrong sign can make an
    agent spin away from the ball for many frames.
    """

    return (angle + np.pi) % (2.0 * np.pi) - np.pi
