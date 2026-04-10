"""Head-to-head policy evaluation for the leaderboard server.

Loads TorchScript policy modules and runs them against each other using the
native soccer environment. Results feed into the ELO rating system.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from puffer_soccer.policy_bundle_runner import load_policy_module, forward_policy_module
from puffer_soccer.vector_env import VecEnvConfig, make_soccer_vecenv

from server.config import PLAYERS_PER_TEAM, GAME_LENGTH, GAMES_PER_MATCHUP, EVAL_NUM_ENVS
from server.elo import update_elo

logger = logging.getLogger(__name__)


def _make_side_assignment(num_envs: int) -> np.ndarray:
    """Split envs so policy A plays blue in the first half, red in the second."""
    current_on_blue = np.zeros((num_envs,), dtype=bool)
    current_on_blue[: (num_envs + 1) // 2] = True
    return current_on_blue


def _score_from_perspective(
    goals_blue: int, goals_red: int, current_on_blue: bool
) -> tuple[float, float]:
    """Return (score_diff, win_score) from the current policy's perspective."""
    score_diff = (
        float(goals_blue - goals_red)
        if current_on_blue
        else float(goals_red - goals_blue)
    )
    if score_diff > 0:
        win = 1.0
    elif score_diff < 0:
        win = 0.0
    else:
        win = 0.5
    return score_diff, win


class MatchEvaluator:
    """Run head-to-head soccer matches between two TorchScript policy modules."""

    def __init__(
        self,
        players_per_team: int = PLAYERS_PER_TEAM,
        game_length: int = GAME_LENGTH,
        num_envs: int = EVAL_NUM_ENVS,
        device: str = "cpu",
    ):
        self.device = device
        self.players_per_team = players_per_team
        self.num_players = players_per_team * 2
        self.num_envs = num_envs

        self.env = make_soccer_vecenv(
            players_per_team=players_per_team,
            vec=VecEnvConfig(backend="native", shard_num_envs=num_envs, num_shards=1),
            action_mode="discrete",
            game_length=game_length,
            render_mode=None,
            seed=0,
            log_interval=1,
            opponents_enabled=True,
        )

        self.a_on_blue = _make_side_assignment(num_envs)

        # Build agent index masks: which flat agent indices belong to policy A vs B
        a_mask = np.zeros((self.env.num_agents,), dtype=bool)
        for env_idx in range(num_envs):
            start = env_idx * self.num_players
            split = start + players_per_team
            end = start + self.num_players
            if self.a_on_blue[env_idx]:
                a_mask[start:split] = True
            else:
                a_mask[split:end] = True

        self.a_indices = torch.as_tensor(np.flatnonzero(a_mask), dtype=torch.long, device=device)
        self.b_indices = torch.as_tensor(np.flatnonzero(~a_mask), dtype=torch.long, device=device)
        self.action_buf = np.zeros((self.env.num_agents,), dtype=np.int32)

    def run_match(
        self,
        policy_a: torch.jit.ScriptModule,
        policy_b: torch.jit.ScriptModule,
        num_games: int = GAMES_PER_MATCHUP,
        seed: int = 42,
    ) -> dict:
        """Play num_games between policy_a and policy_b, return match results.

        Returns dict with: games_played, wins_a, wins_b, draws, win_rate_a, score_diff
        """
        obs, _ = self.env.reset(seed=seed)

        completed = 0
        wins_a = 0
        wins_b = 0
        draws = 0
        score_diffs: list[float] = []
        win_scores: list[float] = []

        with torch.no_grad():
            while completed < num_games:
                obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

                logits_a, _ = forward_policy_module(policy_a, obs_t.index_select(0, self.a_indices))
                logits_b, _ = forward_policy_module(policy_b, obs_t.index_select(0, self.b_indices))

                acts_a = torch.argmax(logits_a, dim=-1).cpu().numpy().astype(np.int32)
                acts_b = torch.argmax(logits_b, dim=-1).cpu().numpy().astype(np.int32)

                self.action_buf[self.a_indices.cpu().numpy()] = acts_a
                self.action_buf[self.b_indices.cpu().numpy()] = acts_b

                _, _, terminals, truncations, _ = self.env.step(self.action_buf)
                done_envs = np.flatnonzero(
                    terminals.reshape(self.num_envs, self.num_players).all(axis=1)
                    | truncations.reshape(self.num_envs, self.num_players).all(axis=1)
                )

                for env_idx in done_envs:
                    if completed >= num_games:
                        break
                    final_goals = self.env.get_last_episode_scores(int(env_idx))
                    if final_goals is None:
                        continue
                    sd, ws = _score_from_perspective(
                        final_goals[0], final_goals[1], bool(self.a_on_blue[env_idx])
                    )
                    score_diffs.append(sd)
                    win_scores.append(ws)
                    if ws == 1.0:
                        wins_a += 1
                    elif ws == 0.0:
                        wins_b += 1
                    else:
                        draws += 1
                    completed += 1

                obs = self.env.observations

        games = len(win_scores)
        win_rate_a = float(np.mean(win_scores)) if games > 0 else 0.5

        return {
            "games_played": games,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "win_rate_a": win_rate_a,
            "score_diff": float(np.mean(score_diffs)) if games > 0 else 0.0,
        }

    def close(self):
        self.env.close()


def run_round_robin(
    new_submission_id: str,
    new_policy_path: str,
    opponents: list[dict],
    db,
    k_factor: float = 32.0,
    games_per_matchup: int = GAMES_PER_MATCHUP,
    device: str = "cpu",
) -> float:
    """Run a new submission against all existing active submissions.

    Updates ELO ratings for all involved submissions and records matches.
    Returns the new submission's final ELO.
    """
    evaluator = MatchEvaluator(device=device)
    try:
        new_policy = load_policy_module(new_policy_path, device=device)
        new_elo = 1000.0

        for opp in opponents:
            opp_policy = load_policy_module(opp["policy_path"], device=device)
            opp_elo = opp["elo_rating"]

            result = evaluator.run_match(new_policy, opp_policy, num_games=games_per_matchup)

            logger.info(
                "Match %s vs %s: win_rate=%.2f score_diff=%.2f",
                new_submission_id,
                opp["id"],
                result["win_rate_a"],
                result["score_diff"],
            )

            # Update ELO
            new_elo, opp_elo = update_elo(new_elo, opp_elo, result["win_rate_a"], k=k_factor)
            db.update_submission_elo(new_submission_id, new_elo)
            db.update_submission_elo(opp["id"], opp_elo)

            # Record match
            db.record_match(
                sub_a_id=new_submission_id,
                sub_b_id=opp["id"],
                games_played=result["games_played"],
                wins_a=result["wins_a"],
                wins_b=result["wins_b"],
                draws=result["draws"],
                win_rate_a=result["win_rate_a"],
                score_diff=result["score_diff"],
            )

        return new_elo
    finally:
        evaluator.close()
