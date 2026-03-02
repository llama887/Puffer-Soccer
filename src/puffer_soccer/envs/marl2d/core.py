from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
import pufferlib

from .csrc import binding
from .constants import DEFAULT_GAME_LENGTH, DEFAULT_VISION_RANGE
from .renderer import SoccerRenderer


@dataclass(frozen=True)
class EnvConfig:
    num_envs: int = 1
    players_per_team: int = 11
    game_length: int = DEFAULT_GAME_LENGTH
    action_mode: str = "discrete"  # discrete or continuous
    do_team_switch: bool = False
    vision_range: float = DEFAULT_VISION_RANGE
    reset_setup: str = "position"  # position or random
    log_interval: int = 128
    render_mode: str | None = None
    seed: int = 0
    buf: dict[str, np.ndarray] | None = None


class MARL2DPufferEnv(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs: int = 1,
        players_per_team: int = 11,
        game_length: int = DEFAULT_GAME_LENGTH,
        action_mode: str = "discrete",
        do_team_switch: bool = False,
        vision_range: float = DEFAULT_VISION_RANGE,
        reset_setup: str = "position",
        log_interval: int = 128,
        render_mode: str | None = None,
        buf: dict[str, np.ndarray] | None = None,
        seed: int = 0,
    ):
        if players_per_team < 1 or players_per_team > 11:
            raise ValueError("players_per_team must be in [1, 11]")
        if action_mode not in ("discrete", "continuous"):
            raise ValueError("action_mode must be discrete or continuous")
        if reset_setup not in ("position", "random"):
            raise ValueError("reset_setup must be position or random")

        self.render_mode = render_mode
        self.players_per_team = players_per_team
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

        self.global_states = np.zeros((self.num_agents, self.state_size), dtype=np.float32)
        self._renderer = SoccerRenderer(render_mode=render_mode) if render_mode in ("human", "rgb_array") else None
        self._handle = binding.vec_init(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminals=self.terminals,
            truncations=self.truncations,
            global_states=self.global_states,
            num_envs=num_envs,
            seed=seed,
            players_per_team=players_per_team,
            game_length=game_length,
            action_mode=self._action_mode_i,
            do_team_switch=int(do_team_switch),
            vision_range=float(vision_range),
            reset_setup=0 if reset_setup == "position" else 1,
        )
        self.tick = 0

    def reset(self, seed: int | None = 0):
        if seed is None:
            seed = 0
        binding.vec_reset(self._handle, int(seed))
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

    def get_state(self, env_idx: int = 0) -> dict[str, Any]:
        return binding.vec_get_state(self._handle, env_idx)

    def get_last_episode_scores(self, env_idx: int = 0, clear: bool = True) -> tuple[int, int] | None:
        scores = binding.vec_get_last_scores(self._handle, env_idx, clear)
        if scores is None:
            return None
        return int(scores[0]), int(scores[1])

    def render(self, env_idx: int = 0):
        if self._renderer is None:
            return None
        if env_idx < 0 or env_idx >= self.num_envs:
            raise ValueError("invalid env index")
        state = self.get_state(env_idx)
        return self._renderer.render(state)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if getattr(self, "_handle", None) is not None:
            binding.vec_close(self._handle)
            self._handle = None


def make_puffer_env(**kwargs: Any) -> MARL2DPufferEnv:
    return MARL2DPufferEnv(**kwargs)
