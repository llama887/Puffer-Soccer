from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from .core import MARL2DPufferEnv, unflatten_obs
from .renderer import SoccerRenderer


class MARL2DParallelEnv(ParallelEnv):
    metadata = {"name": "marl2d_puffer", "render_modes": ["human", "rgb_array", "none"]}

    def __init__(
        self,
        players_per_team: int = 11,
        game_length: int = 400,
        action_mode: str = "discrete",
        do_team_switch: bool = False,
        vision_range: float = np.pi,
        reset_setup: str = "position",
        render_mode: str | None = None,
        seed: int = 0,
    ):
        self.players_per_team = players_per_team
        self.num_players = players_per_team * 2
        self.possible_agents = [f"agent_{i}" for i in range(self.num_players)]
        self.agents = self.possible_agents[:]

        self._puffer = MARL2DPufferEnv(
            num_envs=1,
            players_per_team=players_per_team,
            game_length=game_length,
            action_mode=action_mode,
            do_team_switch=do_team_switch,
            vision_range=vision_range,
            reset_setup=reset_setup,
            render_mode=render_mode,
            seed=seed,
        )
        self._action_mode = action_mode

        self._obs_space = spaces.Dict(
            {
                "self": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
                "ball": spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
                "teammates": spaces.Box(-1.0, 1.0, shape=(players_per_team - 1, 7), dtype=np.float32),
                "opponents": spaces.Box(-1.0, 1.0, shape=(players_per_team, 7), dtype=np.float32),
                "time_left": spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
                "one_hot_id": spaces.Box(0.0, 1.0, shape=(11,), dtype=np.float32),
            }
        )

        if action_mode == "discrete":
            self._act_space = spaces.Discrete(9)
        else:
            self._act_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                shape=(2,),
                dtype=np.float32,
            )

        self._renderer = SoccerRenderer(render_mode=render_mode) if render_mode in ("human", "rgb_array") else None

    def observation_space(self, agent: str):
        return self._obs_space

    def action_space(self, agent: str):
        return self._act_space

    def _obs_dict(self):
        obs = {}
        for i, agent in enumerate(self.possible_agents):
            obs[agent] = unflatten_obs(self._puffer.observations[i], self.players_per_team)
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        self._puffer.reset(seed=0 if seed is None else seed)
        self.agents = self.possible_agents[:]
        obs = self._obs_dict()
        infos = {
            agent: {"global_state": self._puffer.global_states[i].copy()}
            for i, agent in enumerate(self.possible_agents)
        }
        return obs, infos

    def step(self, actions: dict[str, Any]):
        if not self.agents:
            raise RuntimeError("step called on terminated environment; call reset")

        if self._action_mode == "discrete":
            action_arr = np.zeros((self.num_players,), dtype=np.int32)
            for i, agent in enumerate(self.possible_agents):
                action_arr[i] = int(actions.get(agent, 0))
        else:
            action_arr = np.zeros((self.num_players, 2), dtype=np.float32)
            for i, agent in enumerate(self.possible_agents):
                action_arr[i] = np.asarray(actions.get(agent, np.zeros(2)), dtype=np.float32)

        self._puffer.step(action_arr)

        obs = self._obs_dict()
        rewards = {agent: float(self._puffer.rewards[i]) for i, agent in enumerate(self.possible_agents)}
        terminations = {agent: bool(self._puffer.terminals[i]) for i, agent in enumerate(self.possible_agents)}
        truncations = {agent: bool(self._puffer.truncations[i]) for i, agent in enumerate(self.possible_agents)}
        infos = {
            agent: {"global_state": self._puffer.global_states[i].copy()}
            for i, agent in enumerate(self.possible_agents)
        }

        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def render(self):
        if self._renderer is None:
            return None
        state = self._puffer.get_state(0)
        return self._renderer.render(state)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        self._puffer.close()


def make_parallel_env(**kwargs: Any) -> MARL2DParallelEnv:
    return MARL2DParallelEnv(**kwargs)
