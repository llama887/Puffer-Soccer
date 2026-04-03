"""Tests for league training helpers and trainer-side league wiring.

These tests focus on the new MARLadona-inspired opponent-pool behavior. The goal is to keep
the league bookkeeping and wrapper contracts stable without having to run a full PPO training
job in every test.
"""

# pylint: disable=wrong-import-position,too-many-instance-attributes

from __future__ import annotations

from pathlib import Path
import sys

import gymnasium
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_pufferl


class FakeLeagueEnv:
    """Minimal vector-env stub that exposes the buffers used by the league wrapper.

    The wrapper only needs a small subset of the full vector-environment surface: fixed-size
    flat agent buffers, simple reset/step methods, and observation/action space metadata. This
    fake env provides exactly that so the tests can exercise the hidden-opponent wiring
    directly.
    """

    def __init__(self, num_envs: int = 2, players_per_team: int = 2) -> None:
        self.num_envs = num_envs
        self.players_per_team = players_per_team
        self.num_agents = num_envs * players_per_team * 2
        self.opponents_enabled = True
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(3)
        self.observations = np.zeros(
            (self.num_agents, self.single_observation_space.shape[0]),
            dtype=np.float32,
        )
        self.rewards = np.zeros((self.num_agents,), dtype=np.float32)
        self.terminals = np.zeros((self.num_agents,), dtype=bool)
        self.truncations = np.zeros((self.num_agents,), dtype=bool)
        self.masks = np.ones((self.num_agents,), dtype=bool)
        self.actions = np.zeros((self.num_agents,), dtype=np.int32)
        self.last_full_actions = np.zeros_like(self.actions)

    def reset(self, seed: int | None = None):
        """Reset the fake env and return one deterministic observation batch."""

        del seed
        self.observations.fill(1.0)
        self.rewards.fill(0.0)
        self.terminals.fill(False)
        self.truncations.fill(False)
        return self.observations, {}

    def step(self, actions: np.ndarray):
        """Record the full action array and terminate all environments immediately."""

        self.last_full_actions = actions.copy()
        self.actions[:] = actions
        self.observations.fill(2.0)
        self.rewards.fill(0.25)
        self.terminals.fill(True)
        self.truncations.fill(False)
        return self.observations, self.rewards, self.terminals, self.truncations, {}

    def close(self) -> None:
        """Close the fake env without external side effects."""

        return None


def test_league_manager_caps_pool_and_promotes_on_threshold() -> None:
    """Verify league promotion appends snapshots and evicts the oldest when full.

    The capped pool is the core mechanism we are adding. This test keeps the FIFO eviction
    rule and paper-style win-rate promotion gate explicit.
    """

    config = train_pufferl.LeagueConfig(
        rl_alg="league",
        max_size=2,
        promotion_win_rate_threshold=0.75,
        standardized_eval_ratio=0.0,
        standardized_eval_enabled=False,
    )
    manager = train_pufferl.LeagueManager(config, seed=0)
    manager.bootstrap({"a": 1}, label="bootstrap", source_epoch=0)
    manager.append_snapshot({"a": 2}, label="epoch_1", source_epoch=1)
    result = manager.maybe_promote(
        aggregate_win_rate=0.80,
        aggregate_score_diff=0.1,
        snapshot_state_dict={"a": 3},
        source_epoch=2,
        label="epoch_2",
    )

    assert result.promoted is True
    assert manager.size() == 2
    assert [entry.source_epoch for entry in manager.entries] == [1, 2]


def test_league_manager_sampling_returns_known_entry_ids() -> None:
    """Verify uniform sampling draws only from the active retained league ids.

    Training assigns one frozen opponent per environment. The wrapper relies on these ids
    being stable and resolvable back into retained snapshots.
    """

    config = train_pufferl.LeagueConfig(
        rl_alg="league",
        max_size=8,
        promotion_win_rate_threshold=0.75,
        standardized_eval_ratio=0.0,
        standardized_eval_enabled=False,
    )
    manager = train_pufferl.LeagueManager(config, seed=123)
    manager.bootstrap({"a": 1}, label="bootstrap", source_epoch=0)
    manager.append_snapshot({"a": 2}, label="epoch_1", source_epoch=1)

    sampled = manager.sample_entry_ids(16)

    assert set(sampled).issubset({entry.entry_id for entry in manager.entries})
    assert len(sampled) == 16


def test_league_training_wrapper_exposes_only_learner_rows_and_tracks_assignments() -> None:
    """Verify the wrapper slices learner rows and keeps a non-empty assignment histogram.

    This is the key rollout contract for league training: PPO should see only learner rows,
    while frozen league policies control the hidden opponent rows internally.
    """

    env = FakeLeagueEnv()
    policy = train_pufferl.Policy(env)
    config = train_pufferl.LeagueConfig(
        rl_alg="league",
        max_size=8,
        promotion_win_rate_threshold=0.75,
        standardized_eval_ratio=0.0,
        standardized_eval_enabled=False,
    )
    manager = train_pufferl.LeagueManager(config, seed=0)
    manager.bootstrap(
        train_pufferl.snapshot_policy_state(policy),
        label="bootstrap",
        source_epoch=0,
    )
    wrapper = train_pufferl.LeagueTrainingWrapper(
        env,
        players_per_team=env.players_per_team,
        device="cpu",
        league_manager=manager,
    )

    obs, _ = wrapper.reset(seed=0)
    _, rewards, terminals, truncations, _ = wrapper.step(
        np.zeros((wrapper.num_agents,), dtype=np.int32)
    )

    assert obs.shape[0] == env.num_envs * env.players_per_team
    assert rewards.shape[0] == wrapper.num_agents
    assert terminals.shape[0] == wrapper.num_agents
    assert truncations.shape[0] == wrapper.num_agents
    assert wrapper.latest_opponent_histogram()
    assert env.last_full_actions.shape[0] == env.num_agents
