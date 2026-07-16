"""Evaluate a saved policy checkpoint against the Helios-inspired teacher."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any

import imageio.v2 as imageio
import numpy as np

from puffer_soccer.envs.marl2d import make_puffer_env
from puffer_soccer.envs.marl2d.core import (
    DISCRETE_ACTION_MOVE_FORWARD,
    DISCRETE_ACTION_ROTATE_LEFT,
    DISCRETE_ACTION_ROTATE_RIGHT,
    DISCRETE_KICK_ACTION_START,
)
from puffer_soccer.helios_teacher import HeliosTeacher, HeliosTeacherConfig
from puffer_soccer.vector_env import VecEnvConfig, make_soccer_vecenv


def _load_train_module() -> Any:
    """Load ``train_pufferl.py`` so evaluation uses the repo's policy helpers.

    The best checked-in bot is a raw PyTorch checkpoint, not a standalone policy bundle. The
    training script already knows how to rebuild the correct feed-forward or recurrent module
    from checkpoint keys, run recurrent eval inference, resolve the active device, and compute
    side-balanced win scores. Importing those helpers avoids duplicating fragile checkpoint
    loading logic in this one-off teacher evaluator.
    """

    script_path = Path(__file__).resolve().with_name("train_pufferl.py")
    spec = importlib.util.spec_from_file_location("train_pufferl_helios_eval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_pufferl = _load_train_module()
torch = train_pufferl.torch


class PolicyVsHeliosEvaluator:  # pylint: disable=too-many-instance-attributes
    """Run side-balanced matches between one torch policy and the scripted teacher.

    Existing head-to-head evaluation assumes both sides are torch policies that consume batched
    observations. The Helios teacher is different: it consumes each environment's full public
    state and emits actions for all players. This evaluator keeps the same side-balancing and
    score summary conventions as the training evaluator, but fills the opponent side from
    ``HeliosTeacher`` actions instead of policy logits.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        players_per_team: int,
        game_length: int,
        eval_envs: int,
        device: str,
        teacher_config: HeliosTeacherConfig | None = None,
    ) -> None:
        self.players_per_team = int(players_per_team)
        self.num_players = self.players_per_team * 2
        self.device = device
        self.eval_env = make_soccer_vecenv(
            players_per_team=self.players_per_team,
            action_mode="discrete",
            game_length=game_length,
            render_mode=None,
            seed=0,
            opponents_enabled=True,
            log_interval=1,
            vec=VecEnvConfig(backend="native", shard_num_envs=eval_envs, num_shards=1),
        )
        self.num_envs = int(self.eval_env.num_envs)
        self.current_on_blue = train_pufferl.make_side_assignment(self.num_envs)
        current_mask = np.zeros((self.eval_env.num_agents,), dtype=bool)
        for env_idx in range(self.num_envs):
            start = env_idx * self.num_players
            split = start + self.players_per_team
            end = start + self.num_players
            if self.current_on_blue[env_idx]:
                current_mask[start:split] = True
            else:
                current_mask[split:end] = True

        self.current_indices_np = np.flatnonzero(current_mask).astype(np.int64, copy=False)
        self.current_indices = torch.as_tensor(
            self.current_indices_np, dtype=torch.long, device=device
        )
        self.action_buf = np.zeros((self.eval_env.num_agents,), dtype=np.int32)
        self.teacher = HeliosTeacher(self.players_per_team, config=teacher_config)
        self.helios_action_counts = {
            "move_forward": 0,
            "rotate": 0,
            "kick": 0,
            "total": 0,
        }

    # pylint: disable=too-many-locals
    def evaluate(
        self,
        policy: Any,
        *,
        games: int,
        seed: int,
    ) -> dict[str, float]:
        """Return win rate and score difference from the policy's point of view.

        Draws count as half a win, matching the rest of this repo's evaluator. The policy plays
        both field orientations because ``current_on_blue`` alternates the controlled side
        across vector envs. Helios actions are generated per env from the same pre-step state and
        then only the opponent half is copied into the native action buffer.
        """

        obs, _ = self.eval_env.reset(seed=seed)
        self.eval_env.flush_log()
        for key in self.helios_action_counts:
            self.helios_action_counts[key] = 0
        completed_games = 0
        win_scores: list[float] = []
        score_diffs: list[float] = []

        was_training = policy.training
        policy.eval()
        recurrent_state = train_pufferl.initial_recurrent_eval_state(policy)
        with torch.no_grad():
            while completed_games < games:
                self._fill_teacher_actions()
                obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                logits, _ = train_pufferl.forward_policy_eval(
                    policy,
                    obs_tensor.index_select(0, self.current_indices),
                    recurrent_state,
                )
                policy_actions = (
                    torch.argmax(logits, dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False)
                )
                self.action_buf[self.current_indices_np] = policy_actions

                _, _, terminals, truncations, _ = self.eval_env.step(self.action_buf)
                done_envs = np.flatnonzero(
                    terminals.reshape(self.num_envs, self.num_players).all(axis=1)
                    | truncations.reshape(self.num_envs, self.num_players).all(axis=1)
                )
                for env_idx in done_envs:
                    if completed_games >= games:
                        break
                    final_goals = self.eval_env.get_last_episode_scores(int(env_idx))
                    if final_goals is None:
                        continue
                    score_diff, win_score = train_pufferl.score_metrics_from_perspective(
                        final_goals[0],
                        final_goals[1],
                        bool(self.current_on_blue[env_idx]),
                    )
                    score_diffs.append(score_diff)
                    win_scores.append(win_score)
                    completed_games += 1
                obs = self.eval_env.observations

        if was_training:
            policy.train()
        metrics = train_pufferl.summarize_match_results(win_scores, score_diffs)
        total = max(1, self.helios_action_counts["total"])
        metrics.update(
            {
                "helios_move_forward_frac": self.helios_action_counts["move_forward"] / total,
                "helios_rotate_frac": self.helios_action_counts["rotate"] / total,
                "helios_kick_frac": self.helios_action_counts["kick"] / total,
            }
        )
        return metrics

    def _fill_teacher_actions(self) -> None:
        """Fill the native action buffer with Helios actions for every opponent side.

        The teacher returns actions for both teams because it reasons from the complete public
        state. During policy-vs-teacher evaluation, only the side not controlled by the policy is
        kept. The policy side is overwritten afterward by neural-network actions, so this method
        can use the same dense action buffer without extra temporary arrays.
        """

        for env_idx in range(self.num_envs):
            full_actions = self.teacher.actions(self.eval_env.get_state(env_idx))
            start = env_idx * self.num_players
            split = start + self.players_per_team
            end = start + self.num_players
            if self.current_on_blue[env_idx]:
                helios_actions = full_actions[self.players_per_team :]
                self.action_buf[split:end] = helios_actions
            else:
                helios_actions = full_actions[: self.players_per_team]
                self.action_buf[start:split] = helios_actions
            self._accumulate_helios_action_counts(helios_actions)

    def _accumulate_helios_action_counts(self, helios_actions: np.ndarray) -> None:
        """Track the scripted opponent's basic action mix during numeric evaluation.

        The Helios teacher can become worse in several different ways: it can spin too much,
        move without ever touching the ball, or kick constantly without enough control. These
        counters are intentionally simple fractions over all scripted opponent actions. They
        make each evaluation run measurable beyond the final win rate, while avoiding expensive
        possession reconstruction inside the hot rollout loop.
        """

        self.helios_action_counts["move_forward"] += int(
            np.count_nonzero(helios_actions == DISCRETE_ACTION_MOVE_FORWARD)
        )
        self.helios_action_counts["rotate"] += int(
            np.count_nonzero(
                (helios_actions == DISCRETE_ACTION_ROTATE_LEFT)
                | (helios_actions == DISCRETE_ACTION_ROTATE_RIGHT)
            )
        )
        self.helios_action_counts["kick"] += int(
            np.count_nonzero(helios_actions >= DISCRETE_KICK_ACTION_START)
        )
        self.helios_action_counts["total"] += int(helios_actions.size)

    def close(self) -> None:
        """Release the native vector environment used for evaluation."""

        self.eval_env.close()


def load_checkpoint_policy(
    *,
    checkpoint_path: Path,
    players_per_team: int,
    device: str,
) -> Any:
    """Load a raw policy checkpoint into the matching current in-repo module."""

    env = make_soccer_vecenv(
        players_per_team=players_per_team,
        action_mode="discrete",
        game_length=400,
        render_mode=None,
        seed=0,
        opponents_enabled=True,
        vec=VecEnvConfig(backend="native", shard_num_envs=1, num_shards=1),
    )
    try:
        state_dict = train_pufferl.load_checkpoint_state_dict(
            train_pufferl.resolve_checkpoint_file(checkpoint_path)
        )
        policy = train_pufferl.build_policy_for_state(env, state_dict).to(device)
        policy.load_state_dict(state_dict, strict=True)
        return policy
    finally:
        env.close()


def record_policy_vs_helios_video(  # pylint: disable=too-many-arguments,too-many-locals
    policy: Any,
    *,
    players_per_team: int,
    game_length: int,
    seed: int,
    device: str,
    output_path: Path,
    video_fps: int,
    policy_on_blue: bool,
) -> tuple[int, int] | None:
    """Render one policy-vs-Helios game to an mp4 and return its final score.

    The scalar renderer is easier to inspect than the native vector evaluator because it draws
    exactly one match from start to finish. The policy controls one full team and the Helios
    teacher controls the other team from the same public environment state used during numeric
    evaluation. This makes the replay a direct visual check of the reported matchup rather than
    a separate self-play or policy-vs-policy video path.

    Recurrent policy state is carried across frames through ``forward_policy_eval`` just like in
    the numeric evaluator. The only difference is that observations are sliced from one scalar
    env instead of from a vector batch, and rendered frames are saved through ``imageio``.
    """

    env = make_puffer_env(
        players_per_team=players_per_team,
        action_mode="discrete",
        game_length=game_length,
        render_mode="rgb_array",
        seed=seed,
        opponents_enabled=True,
    )
    teacher = HeliosTeacher(players_per_team)
    frames: list[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    policy_indices = (
        np.arange(players_per_team, dtype=np.int64)
        if policy_on_blue
        else np.arange(players_per_team, players_per_team * 2, dtype=np.int64)
    )
    action_buf = np.zeros((players_per_team * 2,), dtype=np.int32)

    was_training = policy.training
    policy.eval()
    recurrent_state = train_pufferl.initial_recurrent_eval_state(policy)
    try:
        with torch.no_grad():
            for _ in range(game_length):
                frame = env.render()
                if frame is not None:
                    frames.append(frame.astype(np.uint8, copy=False))

                full_teacher_actions = teacher.actions(env.get_state())
                if policy_on_blue:
                    action_buf[players_per_team:] = full_teacher_actions[players_per_team:]
                else:
                    action_buf[:players_per_team] = full_teacher_actions[:players_per_team]

                obs_tensor = torch.as_tensor(
                    obs[policy_indices],
                    device=device,
                    dtype=torch.float32,
                )
                logits, _ = train_pufferl.forward_policy_eval(
                    policy,
                    obs_tensor,
                    recurrent_state,
                )
                action_buf[policy_indices] = (
                    torch.argmax(logits, dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False)
                )
                obs, _, terminals, truncations, _ = env.step(action_buf)
                if bool(np.all(terminals) or np.all(truncations)):
                    break

            frame = env.render()
            if frame is not None:
                frames.append(frame.astype(np.uint8, copy=False))
    finally:
        score = env.get_last_episode_scores(clear=False)
        env.close()
        if was_training:
            policy.train()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=video_fps)  # type: ignore[arg-type]
    return score


def parse_args() -> argparse.Namespace:
    """Parse CLI options for policy-vs-Helios evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("experiments/lhasdbcl/model_100000.pt"),
    )
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--game-length", type=int, default=400)
    parser.add_argument("--eval-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--video-output", type=Path, default=None)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--policy-side", choices=["blue", "red"], default="blue")
    return parser.parse_args()


def main() -> None:
    """Run the policy-vs-teacher evaluation and print a compact result line."""

    args = parse_args()
    device = train_pufferl.resolve_device(args.device)
    policy = load_checkpoint_policy(
        checkpoint_path=args.checkpoint_path,
        players_per_team=args.players_per_team,
        device=device,
    )
    evaluator = PolicyVsHeliosEvaluator(
        players_per_team=args.players_per_team,
        game_length=args.game_length,
        eval_envs=args.eval_envs,
        device=device,
    )
    try:
        metrics = evaluator.evaluate(policy, games=args.games, seed=args.seed)
    finally:
        evaluator.close()
    print(
        "Policy vs Helios evaluation: "
        f"checkpoint={args.checkpoint_path}, "
        f"games={int(metrics['games'])}, "
        f"win_rate={metrics['win_rate']:.3f}, "
        f"score_diff={metrics['score_diff']:.3f}, "
        f"helios_move_forward_frac={metrics['helios_move_forward_frac']:.3f}, "
        f"helios_rotate_frac={metrics['helios_rotate_frac']:.3f}, "
        f"helios_kick_frac={metrics['helios_kick_frac']:.3f}"
    )
    if args.video_output is not None:
        score = record_policy_vs_helios_video(
            policy,
            players_per_team=args.players_per_team,
            game_length=args.game_length,
            seed=args.seed,
            device=device,
            output_path=args.video_output,
            video_fps=args.video_fps,
            policy_on_blue=args.policy_side == "blue",
        )
        print(f"video={args.video_output}")
        print(f"video_score={score}")


if __name__ == "__main__":
    main()
