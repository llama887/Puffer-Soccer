from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import dm_env
import numpy as np


def _load_third_party():
    repo_root = Path(__file__).resolve().parents[4]
    third_party = repo_root / "third-party" / "MARL2DFootball"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    from custom_football_env import FootballEnv

    return FootballEnv


@dataclass
class ReferenceStep:
    obs: np.ndarray
    state: np.ndarray
    rewards: np.ndarray
    done: bool


class ReferenceEnvAdapter:
    def __init__(
        self,
        players_per_team: int,
        action_mode: str = "continuous",
        game_length: int = 400,
        reset_setup: str = "position",
    ):
        FootballEnv = _load_third_party()
        self.players_per_team = players_per_team
        self.num_players = players_per_team * 2

        self.env = FootballEnv(
            render_game=False,
            game_setting="ppo_attention_state",
            players_per_team=[players_per_team, players_per_team],
            do_team_switch=False,
            include_wait=False,
            game_length=game_length,
            game_diff=1.0,
            vision_range=np.pi,
            action_space=action_mode,
            reset_setup=reset_setup,
        )

    def _to_array(self, timestep, extras):
        obs = []
        state = []
        rewards = []

        for i in range(self.num_players):
            key = f"agent_{i}"
            pieces = [np.asarray(x, dtype=np.float32).reshape(-1) for x in timestep.observation[key].observation]
            obs.append(np.concatenate(pieces, axis=0))

            s = extras["env_states"][key]
            if s is None:
                state.append(np.zeros_like(state[0]))
            else:
                state_pieces = [np.asarray(x, dtype=np.float32).reshape(-1) for x in s]
                state.append(np.concatenate(state_pieces, axis=0))

            rewards.append(float(timestep.reward[key]) if timestep.reward is not None else 0.0)

        return np.stack(obs), np.stack(state), np.asarray(rewards, dtype=np.float32)

    def reset(self) -> ReferenceStep:
        timestep, extras = self.env.reset()
        obs, state, rew = self._to_array(timestep, extras)
        return ReferenceStep(obs=obs, state=state, rewards=rew, done=False)

    def step(self, actions: np.ndarray) -> ReferenceStep:
        dict_actions = {f"agent_{i}": actions[i] for i in range(self.num_players)}
        timestep, extras = self.env.step(dict_actions)
        obs, state, rew = self._to_array(timestep, extras)
        done = timestep.step_type == dm_env.StepType.LAST
        return ReferenceStep(obs=obs, state=state, rewards=rew, done=bool(done))
