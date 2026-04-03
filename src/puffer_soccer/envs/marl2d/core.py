"""Python wrappers and discrete action helpers for the native MARL2D soccer env."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
import pufferlib

from .csrc import binding
from .constants import DEFAULT_GAME_LENGTH, DEFAULT_VISION_RANGE
from .renderer import SoccerRenderer


MAX_SIGNED_ENV_SEED = 2**31 - 1
DISCRETE_ACTION_NOOP = 0
DISCRETE_ACTION_MOVE_FORWARD = 1
DISCRETE_ACTION_MOVE_BACKWARD = 2
DISCRETE_ACTION_ROTATE_LEFT = 3
DISCRETE_ACTION_ROTATE_RIGHT = 4
DISCRETE_KICK_ACTION_START = 5
DISCRETE_KICK_STRENGTHS = (
    0.1,
    0.22857143,
    0.35714287,
    0.4857143,
    0.6142857,
    0.74285716,
    0.87142855,
    1.0,
)
DISCRETE_ACTION_COUNT = DISCRETE_KICK_ACTION_START + len(DISCRETE_KICK_STRENGTHS)


def normalize_env_seed(seed: int | None) -> int:
    """Map arbitrary Python integer seeds onto the signed range accepted by the C env.

    The native soccer binding accepts a signed 32-bit integer seed. Long training runs can
    derive evaluation seeds from the epoch counter, and those values eventually exceed that
    range even though they are still perfectly valid Python integers. That used to crash
    evaluation late in training with an `OverflowError` when the reset call forwarded the raw
    value into the binding.

    This helper keeps the environment API ergonomic by accepting any Python integer or `None`
    and folding it deterministically into the legal signed range before the binding sees it.
    Using modulo arithmetic preserves reproducibility while ensuring every caller, including
    old code paths that do not know about the C limit, remains safe.
    """

    if seed is None:
        return 0
    return int(seed) % MAX_SIGNED_ENV_SEED


def encode_discrete_kick_action(kick_strength: int) -> int:
    """Return the discrete action id for one forward kick-strength choice.

    The competition-facing discrete API now allows exactly one intent per step. Kicks occupy a
    contiguous action range after the locomotion actions so agents, tests, and future baseline
    policies can refer to them with a simple helper instead of hand-written offsets. The helper
    validates the kick-strength index against the canonical Python-side lookup table so the
    action contract stays synchronized with the native C decoder.

    The returned value is the public action id that should be sent into the environment. Kick
    index `0` is the weakest dribble-like tap and the last index is the old full-strength kick.
    """

    if kick_strength < 0 or kick_strength >= len(DISCRETE_KICK_STRENGTHS):
        raise ValueError("kick_strength must be in [0, 7]")
    return DISCRETE_KICK_ACTION_START + kick_strength


def _validate_args(players_per_team: int, action_mode: str, reset_setup: str) -> None:
    if players_per_team < 1 or players_per_team > 11:
        raise ValueError("players_per_team must be in [1, 11]")
    if action_mode != "discrete":
        raise ValueError("action_mode must be discrete")
    if reset_setup not in ("position", "random"):
        raise ValueError("reset_setup must be position or random")


def _accumulate_team_episode_returns(
    rewards: np.ndarray,
    terminals: np.ndarray,
    players_per_team: int,
    running_team_returns: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Update per-env team returns and extract the episodes that ended on this step.

    The native simulator's built-in `episode_return` metric sums rewards from both teams
    together. In a zero-sum self-play game that total naturally collapses toward zero, which
    makes it a poor training-progress curve even when PPO is improving. The Python wrappers
    already have access to every step reward and terminal flag, so they can recover a stable
    one-team view without changing PPO itself and without requiring a rebuilt native module.

    This helper adds the current step reward into a running `(blue, red)` return total for each
    live env. When one or more envs terminate on the step, it returns the finished episode
    totals for those envs and clears their running counters so the next episode starts fresh.
    The caller can then attach those finished totals to the next environment log flush.
    """

    rewards_2d = rewards.reshape(running_team_returns.shape[0], players_per_team * 2)
    terminals_2d = terminals.reshape(running_team_returns.shape[0], players_per_team * 2)
    running_team_returns[:, 0] += rewards_2d[:, :players_per_team].sum(axis=1)
    running_team_returns[:, 1] += rewards_2d[:, players_per_team:].sum(axis=1)

    done_mask = terminals_2d.any(axis=1)
    if not np.any(done_mask):
        return np.zeros(2, dtype=np.float32), 0

    finished_team_returns = running_team_returns[done_mask].sum(axis=0, dtype=np.float32)
    running_team_returns[done_mask] = 0.0
    return finished_team_returns.astype(np.float32, copy=False), int(done_mask.sum())


def _merge_team_episode_return_log(
    log: dict[str, float] | None,
    pending_team_returns: np.ndarray,
    pending_episode_count: int,
) -> dict[str, float] | None:
    """Attach Python-tracked per-team episode returns to an env log payload.

    The wrappers accumulate completed blue-team and red-team episode returns between log
    flushes. This function converts that pending sum into the same averaged-per-episode shape
    used by the native logger and merges it into the outgoing dictionary. Existing native
    fields win when they are already present so a future rebuilt extension can provide the same
    keys directly without fighting the Python fallback.
    """

    if log is None and pending_episode_count == 0:
        return None

    merged = {} if log is None else {str(k): float(v) for k, v in log.items()}
    if pending_episode_count > 0:
        merged.setdefault(
            "blue_team_episode_return",
            float(pending_team_returns[0] / pending_episode_count),
        )
        merged.setdefault(
            "red_team_episode_return",
            float(pending_team_returns[1] / pending_episode_count),
        )
        merged.setdefault(
            "current_team_episode_return",
            float(pending_team_returns[0] / pending_episode_count),
        )
    return merged


@dataclass(frozen=True)
class EnvConfig:
    players_per_team: int = 11
    game_length: int = DEFAULT_GAME_LENGTH
    action_mode: str = "discrete"
    do_team_switch: bool = False
    opponents_enabled: bool = True
    vision_range: float = DEFAULT_VISION_RANGE
    reset_setup: str = "position"
    log_interval: int = 128
    render_mode: str | None = None
    seed: int = 0
    buf: dict[str, np.ndarray] | None = None


class MARL2DPufferEnv(pufferlib.PufferEnv):
    """Wrap one native soccer environment and expose the compiled no-opponent mode.

    The training experiments in this project need a clean diagnostic where the blue team plays
    in a truly opponent-free world. That should still use the normal native reset path and the
    same PPO loop as standard training, just with the red team disabled inside the simulator.

    This wrapper therefore forwards `opponents_enabled` directly into the native binding rather
    than trying to emulate the behavior in Python. Keeping that control inside the C simulator
    avoids ghost opponents in observations, keeps post-goal resets consistent, and lets both
    scalar and native vector environments share the same semantics.

    The wrapper also reconstructs per-team episode returns from the step rewards. That gives
    training a meaningful PPO reward curve such as `environment/current_team_episode_return`
    even when the native aggregate `environment/episode_return` stays at zero because blue and
    red rewards cancel in self-play.
    """

    def __init__(
        self,
        players_per_team: int = 11,
        game_length: int = DEFAULT_GAME_LENGTH,
        action_mode: str = "discrete",
        do_team_switch: bool = False,
        opponents_enabled: bool = True,
        vision_range: float = DEFAULT_VISION_RANGE,
        reset_setup: str = "position",
        log_interval: int = 128,
        render_mode: str | None = None,
        buf: dict[str, np.ndarray] | None = None,
        seed: int = 0,
    ):
        _validate_args(players_per_team, action_mode, reset_setup)

        self.render_mode = render_mode
        self.players_per_team = players_per_team
        self.opponents_enabled = opponents_enabled
        self.num_players = players_per_team * 2
        self.num_envs = 1
        self.num_agents = self.num_players
        self.game_length = game_length
        self.log_interval = log_interval

        self.obs_size = 16 + 14 * players_per_team
        self.state_size = 5 + 34 * players_per_team

        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_size,),
            dtype=np.float32,
        )

        self.single_action_space = gymnasium.spaces.Discrete(DISCRETE_ACTION_COUNT)
        self._action_mode_i = 0

        super().__init__(buf)

        self.global_states = np.zeros(
            (self.num_agents, self.state_size), dtype=np.float32
        )
        self._renderer = (
            SoccerRenderer(render_mode=render_mode)
            if render_mode in ("human", "rgb_array")
            else None
        )
        self._running_team_returns = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._pending_team_return_sum = np.zeros(2, dtype=np.float32)
        self._pending_team_return_count = 0
        self._handle = binding.env_init(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminals=self.terminals,
            truncations=self.truncations,
            global_states=self.global_states,
            seed=normalize_env_seed(seed),
            players_per_team=players_per_team,
            game_length=game_length,
            action_mode=self._action_mode_i,
            do_team_switch=int(do_team_switch),
            opponents_enabled=int(opponents_enabled),
            vision_range=float(vision_range),
            reset_setup=0 if reset_setup == "position" else 1,
        )
        self.tick = 0

    def reset(self, seed: int | None = 0):
        binding.env_reset(self._handle, normalize_env_seed(seed))
        self.tick = 0
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        self._running_team_returns.fill(0.0)
        self._pending_team_return_sum.fill(0.0)
        self._pending_team_return_count = 0
        return self.observations, []

    def step(self, actions: np.ndarray):
        self.tick += 1
        self.actions[:] = actions
        binding.env_step(self._handle)
        finished_team_returns, finished_episode_count = _accumulate_team_episode_returns(
            self.rewards,
            self.terminals,
            self.players_per_team,
            self._running_team_returns,
        )
        if finished_episode_count > 0:
            self._pending_team_return_sum += finished_team_returns
            self._pending_team_return_count += finished_episode_count

        infos = []
        if self.tick % self.log_interval == 0:
            log = self.flush_log()
            if log:
                infos.append(log)

        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def set_field_scale(self, scale: float) -> None:
        """Resize the active field while keeping the current episode state valid.

        The no-opponent curriculum for this project is expressed through environment
        geometry rather than reward shaping. Training starts on a smaller field so the ball
        and goal are easier to reach, then the field grows toward full size over training.

        The native binding owns the live gameplay state, so the wrapper simply forwards the
        requested scale. The binding clamps all agents and the ball into the resized field
        and recomputes observations immediately so the next policy step sees a consistent
        state.
        """

        binding.env_set_field_scale(self._handle, float(scale))

    def get_state(self, env_idx: int = 0) -> dict[str, Any]:
        if env_idx != 0:
            raise ValueError("scalar env only has env_idx=0")
        return binding.env_get_state(self._handle)

    def get_last_episode_scores(
        self, env_idx: int = 0, clear: bool = True
    ) -> tuple[int, int] | None:
        if env_idx != 0:
            raise ValueError("scalar env only has env_idx=0")
        scores = binding.env_get_last_scores(self._handle, clear)
        if scores is None:
            return None
        return int(scores[0]), int(scores[1])

    def render(self, env_idx: int = 0):
        if self._renderer is None:
            return None
        if env_idx != 0:
            raise ValueError("scalar env only has env_idx=0")
        return self._renderer.render(self.get_state(0))

    def flush_log(self) -> dict[str, float] | None:
        log = _merge_team_episode_return_log(
            binding.env_log(self._handle),
            self._pending_team_return_sum,
            self._pending_team_return_count,
        )
        self._pending_team_return_sum.fill(0.0)
        self._pending_team_return_count = 0
        return log

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if getattr(self, "_handle", None) is not None:
            binding.env_close(self._handle)
            self._handle = None


class MARL2DNativeVecEnv(pufferlib.PufferEnv):
    """Wrap the compiled native vector environment.

    The native backend is the fastest way to collect rollouts, so the no-opponent baseline
    should use it whenever the compiled extension supports that mode. Forwarding
    `opponents_enabled` here keeps scalar and vector environments behaviorally aligned.
    Like the scalar wrapper, it also reconstructs per-team episode returns in Python so
    training can log a one-team PPO reward curve even when the native zero-sum aggregate is
    uninformative.
    """

    def __init__(
        self,
        num_envs: int = 1,
        players_per_team: int = 11,
        game_length: int = DEFAULT_GAME_LENGTH,
        action_mode: str = "discrete",
        do_team_switch: bool = False,
        opponents_enabled: bool = True,
        vision_range: float = DEFAULT_VISION_RANGE,
        reset_setup: str = "position",
        log_interval: int = 128,
        render_mode: str | None = None,
        buf: dict[str, np.ndarray] | None = None,
        seed: int = 0,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        _validate_args(players_per_team, action_mode, reset_setup)

        self.render_mode = render_mode
        self.players_per_team = players_per_team
        self.opponents_enabled = opponents_enabled
        self.num_players = players_per_team * 2
        self.num_envs = num_envs
        self.num_agents = self.num_players * num_envs
        self.game_length = game_length
        self.log_interval = log_interval

        self.obs_size = 16 + 14 * players_per_team
        self.state_size = 5 + 34 * players_per_team

        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_size,),
            dtype=np.float32,
        )

        self.single_action_space = gymnasium.spaces.Discrete(DISCRETE_ACTION_COUNT)
        self._action_mode_i = 0

        super().__init__(buf)

        self.global_states = np.zeros(
            (self.num_agents, self.state_size), dtype=np.float32
        )
        self._renderer = (
            SoccerRenderer(render_mode=render_mode)
            if render_mode in ("human", "rgb_array")
            else None
        )
        self._running_team_returns = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._pending_team_return_sum = np.zeros(2, dtype=np.float32)
        self._pending_team_return_count = 0
        self._handle = binding.vec_init(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminals=self.terminals,
            truncations=self.truncations,
            global_states=self.global_states,
            num_envs=num_envs,
            seed=normalize_env_seed(seed),
            players_per_team=players_per_team,
            game_length=game_length,
            action_mode=self._action_mode_i,
            do_team_switch=int(do_team_switch),
            opponents_enabled=int(opponents_enabled),
            vision_range=float(vision_range),
            reset_setup=0 if reset_setup == "position" else 1,
        )
        self.tick = 0

    def reset(self, seed: int | None = 0):
        binding.vec_reset(self._handle, normalize_env_seed(seed))
        self.tick = 0
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        self._running_team_returns.fill(0.0)
        self._pending_team_return_sum.fill(0.0)
        self._pending_team_return_count = 0
        return self.observations, []

    def step(self, actions: np.ndarray):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self._handle)
        finished_team_returns, finished_episode_count = _accumulate_team_episode_returns(
            self.rewards,
            self.terminals,
            self.players_per_team,
            self._running_team_returns,
        )
        if finished_episode_count > 0:
            self._pending_team_return_sum += finished_team_returns
            self._pending_team_return_count += finished_episode_count

        infos = []
        if self.tick % self.log_interval == 0:
            log = self.flush_log()
            if log:
                infos.append(log)

        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def set_field_scale(self, scale: float) -> None:
        """Apply one field scale to every native env shard used for training.

        The trainer updates the curriculum once per epoch based on global training progress.
        Native vector environments keep all env instances inside one C allocation, so the
        update can be broadcast cheaply through a single binding call.
        """

        binding.vec_set_field_scale(self._handle, float(scale))

    def get_state(self, env_idx: int = 0) -> dict[str, Any]:
        return binding.vec_get_state(self._handle, env_idx)

    def get_last_episode_scores(
        self, env_idx: int = 0, clear: bool = True
    ) -> tuple[int, int] | None:
        scores = binding.vec_get_last_scores(self._handle, env_idx, clear)
        if scores is None:
            return None
        return int(scores[0]), int(scores[1])

    def render(self, env_idx: int = 0):
        if self._renderer is None:
            return None
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError("invalid env index")
        return self._renderer.render(self.get_state(env_idx))

    def flush_log(self) -> dict[str, float] | None:
        log = _merge_team_episode_return_log(
            binding.vec_log(self._handle),
            self._pending_team_return_sum,
            self._pending_team_return_count,
        )
        self._pending_team_return_sum.fill(0.0)
        self._pending_team_return_count = 0
        return log

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if getattr(self, "_handle", None) is not None:
            binding.vec_close(self._handle)
            self._handle = None


def make_puffer_env(num_envs: int | None = None, **kwargs: Any) -> MARL2DPufferEnv:
    if num_envs not in (None, 1):
        raise ValueError(
            "make_puffer_env now creates exactly one logical env; use "
            "make_native_vec_env or make_soccer_vecenv for multi-env execution"
        )
    if not hasattr(binding, "env_init"):
        return make_native_vec_env(num_envs=1, **kwargs)
    return MARL2DPufferEnv(**kwargs)


def make_native_vec_env(**kwargs: Any) -> MARL2DNativeVecEnv:
    return MARL2DNativeVecEnv(**kwargs)
