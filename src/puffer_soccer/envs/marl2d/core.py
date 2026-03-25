from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np


MAX_SIGNED_ENV_SEED = 2**31 - 1


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


import pufferlib

from .csrc import binding
from .constants import DEFAULT_GAME_LENGTH, DEFAULT_VISION_RANGE
from .renderer import SoccerRenderer


def _validate_args(players_per_team: int, action_mode: str, reset_setup: str) -> None:
    if players_per_team < 1 or players_per_team > 11:
        raise ValueError("players_per_team must be in [1, 11]")
    if action_mode not in ("discrete", "continuous"):
        raise ValueError("action_mode must be discrete or continuous")
    if reset_setup not in ("position", "random"):
        raise ValueError("reset_setup must be position or random")


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

        if action_mode == "discrete":
            self.single_action_space = gymnasium.spaces.Discrete(9)
            self._action_mode_i = 0
        else:
            self.single_action_space = gymnasium.spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            )
            self._action_mode_i = 1

        super().__init__(buf)

        self.global_states = np.zeros(
            (self.num_agents, self.state_size), dtype=np.float32
        )
        self._renderer = (
            SoccerRenderer(render_mode=render_mode)
            if render_mode in ("human", "rgb_array")
            else None
        )
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
        return self.observations, []

    def step(self, actions: np.ndarray):
        self.tick += 1
        self.actions[:] = actions
        binding.env_step(self._handle)

        infos = []
        if self.tick % self.log_interval == 0:
            log = binding.env_log(self._handle)
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
        log = binding.env_log(self._handle)
        if not log:
            return None
        return {str(k): float(v) for k, v in log.items()}

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

        if action_mode == "discrete":
            self.single_action_space = gymnasium.spaces.Discrete(9)
            self._action_mode_i = 0
        else:
            self.single_action_space = gymnasium.spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            )
            self._action_mode_i = 1

        super().__init__(buf)

        self.global_states = np.zeros(
            (self.num_agents, self.state_size), dtype=np.float32
        )
        self._renderer = (
            SoccerRenderer(render_mode=render_mode)
            if render_mode in ("human", "rgb_array")
            else None
        )
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
        return self.observations, []

    def step(self, actions: np.ndarray):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self._handle)

        infos = []
        if self.tick % self.log_interval == 0:
            log = binding.vec_log(self._handle)
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
        log = binding.vec_log(self._handle)
        if not log:
            return None
        return {str(k): float(v) for k, v in log.items()}

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
            "make_puffer_env now creates exactly one logical env; use make_native_vec_env or make_soccer_vecenv for multi-env execution"
        )
    if not hasattr(binding, "env_init"):
        return make_native_vec_env(num_envs=1, **kwargs)
    return MARL2DPufferEnv(**kwargs)


def make_native_vec_env(**kwargs: Any) -> MARL2DNativeVecEnv:
    return MARL2DNativeVecEnv(**kwargs)
