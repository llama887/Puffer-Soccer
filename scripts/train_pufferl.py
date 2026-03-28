from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import sys
import tempfile
import time
from typing import TYPE_CHECKING, Any, cast

import imageio.v2 as imageio
import numpy as np

import pufferlib
import pufferlib.pufferl as pufferl
import pufferlib.pytorch

from puffer_soccer import policy_bundle
from puffer_soccer.autotune import (
    BenchmarkResult,
    autotune_vecenv,
    format_benchmark_result,
    vec_config_from_benchmark,
)
from puffer_soccer.envs.marl2d import make_puffer_env
from puffer_soccer.torch_loader import import_torch
from puffer_soccer.vector_env import (
    VecEnvConfig,
    make_soccer_vecenv,
    physical_cpu_count,
    total_sim_envs,
)

if TYPE_CHECKING:
    import torch

torch = import_torch()


STANDARD_HYPERPARAMETERS_PATH = Path("experiments/autoload_hyperparameters.json")
STANDARD_HYPERPARAMETER_KEYS = (
    "learning_rate",
    "gamma",
    "gae_lambda",
    "update_epochs",
    "clip_coef",
    "vf_coef",
    "vf_clip_coef",
    "max_grad_norm",
    "ent_coef",
    "prio_alpha",
    "prio_beta0",
    "train_batch_size",
    "bptt_horizon",
    "minibatch_size",
    "no_opponent_map_scale_ladder",
    "no_opponent_map_scale_start",
    "no_opponent_map_scale_end",
    "no_opponent_map_scale_power",
    "no_opponent_map_scale_full_progress",
)
STANDARD_VECENV_KEYS = (
    "vec_backend",
    "num_envs",
    "vec_num_shards",
    "vec_batch_size",
)
SELF_PLAY_RESETTABLE_AUTOLOAD_KEYS = (
    "learning_rate",
    "gamma",
    "gae_lambda",
    "update_epochs",
    "clip_coef",
    "vf_coef",
    "vf_clip_coef",
    "max_grad_norm",
    "ent_coef",
    "prio_alpha",
    "prio_beta0",
    "train_batch_size",
    "bptt_horizon",
    "minibatch_size",
)
EXPLICIT_ARG_FLAGS = {
    "train_batch_size": "--train-batch-size",
    "bptt_horizon": "--bptt-horizon",
    "minibatch_size": "--minibatch-size",
    "update_epochs": "--update-epochs",
    "learning_rate": "--learning-rate",
    "gamma": "--gamma",
    "gae_lambda": "--gae-lambda",
    "clip_coef": "--clip-coef",
    "vf_coef": "--vf-coef",
    "vf_clip_coef": "--vf-clip-coef",
    "max_grad_norm": "--max-grad-norm",
    "ent_coef": "--ent-coef",
    "prio_alpha": "--prio-alpha",
    "prio_beta0": "--prio-beta0",
    "video_output": "--video-output",
    "best_checkpoint_video_output": "--best-checkpoint-video-output",
}
DEFAULT_VIDEO_OUTPUT_PATH = Path("experiments/self_play.mp4")
DEFAULT_BEST_CHECKPOINT_VIDEO_OUTPUT_PATH = Path("experiments/best_checkpoint.mp4")
RUN_VIDEO_FILENAMES = {
    "self_play": "self_play.mp4",
    "no_opponent": "self_play_no_opponent.mp4",
    "best_checkpoint": "best_checkpoint.mp4",
}


class Policy(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = env.single_observation_space.shape[0]
        if hasattr(env.single_action_space, "n"):
            self.discrete = True
            act_dim = env.single_action_space.n
            self.action_head = torch.nn.Linear(256, act_dim)
        else:
            self.discrete = False
            act_dim = env.single_action_space.shape[0]
            self.action_head = torch.nn.Linear(256, act_dim)

        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(obs_dim, 256)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(256, 256)),
            torch.nn.ReLU(),
        )
        self.value_head = torch.nn.Linear(256, 1)

    def forward(self, observations, state=None):
        hidden = self.net(observations)
        logits = self.action_head(hidden)
        values = self.value_head(hidden)
        return logits, values

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state=state)


def forward_policy_eval(
    policy_runner: Any,
    observations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run eval inference against either a live policy module or a frozen bundle module.

    The project now supports two opponent formats during evaluation:
    - a live in-repo ``Policy`` object, usually loaded from a raw state dict
    - a TorchScript module revived from a saved policy bundle

    Keeping the dispatch in one helper lets evaluators and replay exporters share the same
    compatibility layer instead of each path re-implementing slightly different assumptions
    about how a policy object exposes its eval forward pass.
    """

    if hasattr(policy_runner, "forward_eval"):
        output = policy_runner.forward_eval(observations)
    else:
        output = policy_runner(observations)
    if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError("Expected policy runner to return a (logits, values) tuple")
    logits, values = output
    if not isinstance(logits, torch.Tensor) or not isinstance(values, torch.Tensor):
        raise ValueError("Policy runner returned non-tensor outputs")
    return logits, values


def policy_example_observation(
    policy: torch.nn.Module,
    observation_shape: tuple[int, ...],
) -> torch.Tensor:
    """Create one example observation tensor for TorchScript bundle export.

    Exporting a traced evaluation module requires one representative input tensor. The
    observation shape comes from the active soccer environment, while the device comes from
    the live policy so the tracing call uses the same placement the model is already using.
    Keeping this helper separate avoids hard-coding tensor construction details in the bundle
    export path.
    """

    try:
        device = next(policy.parameters()).device
    except StopIteration as err:
        raise ValueError("policy does not expose any parameters") from err
    return torch.zeros((1, *observation_shape), dtype=torch.float32, device=device)

class RegularizedPuffeRL(pufferl.PuffeRL):
    def __init__(
        self,
        config,
        vecenv,
        policy,
        *,
        logger=None,
        regularization_enabled: bool = True,
        past_kl_coef: float = 0.1,
        uniform_kl_base_coef: float = 0.05,
        uniform_kl_power: float = 0.3,
    ):
        super().__init__(config, vecenv, policy, logger=logger)
        self.regularization_enabled = regularization_enabled
        self.past_kl_coef = float(past_kl_coef) if regularization_enabled else 0.0
        self.uniform_kl_base_coef = (
            float(uniform_kl_base_coef) if regularization_enabled else 0.0
        )
        self.uniform_kl_power = float(uniform_kl_power)
        if not hasattr(vecenv.single_action_space, "n"):
            raise ValueError(
                "RegularizedPuffeRL currently supports discrete action spaces only"
            )

        self.uniform_log_prob = -math.log(float(vecenv.single_action_space.n))
        self.past_policy = copy.deepcopy(self.uncompiled_policy).to(config["device"])
        self.past_policy.eval()
        for param in self.past_policy.parameters():
            param.requires_grad_(False)

    def _sync_past_policy(self) -> None:
        self.past_policy.load_state_dict(
            self.uncompiled_policy.state_dict(), strict=True
        )

    @pufferl.record
    def train(self):
        profile = self.profile
        epoch = self.epoch
        profile("train", epoch)
        losses = defaultdict(float)
        config = self.config
        device = config["device"]
        self._sync_past_policy()

        b0 = config["prio_beta0"]
        a = config["prio_alpha"]
        clip_coef = config["clip_coef"]
        vf_clip = config["vf_clip_coef"]
        anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
        self.ratio[:] = 1

        for mb in range(self.total_minibatches):
            profile("train_misc", epoch, nest=True)
            self.amp_context.__enter__()

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = pufferl.compute_puff_advantage(
                self.values,
                self.rewards,
                self.terminals,
                self.ratio,
                advantages,
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )

            profile("train_copy", epoch)
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments * prio_probs[idx, None]) ** -anneal_beta
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_rewards = self.rewards[idx]
            mb_terminals = self.terminals[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            profile("train_forward", epoch)
            if not config["use_rnn"]:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
            )

            logits, newvalue = self.policy(mb_obs, state)
            _, newlogprob, entropy = pufferlib.pytorch.sample_logits(
                logits, action=mb_actions
            )

            profile("train_misc", epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config["clip_coef"]).float().mean()

            adv = advantages[idx]
            adv = pufferl.compute_puff_advantage(
                mb_values,
                mb_rewards,
                mb_terminals,
                ratio,
                adv,
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )
            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            entropy_loss = entropy.mean()

            if not isinstance(logits, torch.Tensor):
                raise ValueError(
                    "RegularizedPuffeRL expects tensor logits for discrete actions"
                )

            if self.regularization_enabled:
                with torch.no_grad():
                    old_logits, _ = self.past_policy(mb_obs, state)
                old_log_probs = torch.log_softmax(old_logits, dim=-1)
                new_log_probs = torch.log_softmax(logits, dim=-1)
                new_probs = torch.softmax(logits, dim=-1)
                past_kl = torch.sum(
                    new_probs * (new_log_probs - old_log_probs), dim=-1
                ).mean()
                iter_number = max(1, self.epoch + 1)
                uniform_kl_coef = self.uniform_kl_base_coef / (
                    iter_number**self.uniform_kl_power
                )
                uniform_kl = torch.sum(
                    new_probs * (new_log_probs - self.uniform_log_prob),
                    dim=-1,
                ).mean()
            else:
                uniform_kl_coef = 0.0
                past_kl = torch.zeros((), device=device)
                uniform_kl = torch.zeros((), device=device)

            loss = (
                pg_loss
                + config["vf_coef"] * v_loss
                - config["ent_coef"] * entropy_loss
                + self.past_kl_coef * past_kl
                + uniform_kl_coef * uniform_kl
            )
            self.amp_context.__enter__()  # TODO: AMP needs some debugging

            self.values[idx] = newvalue.detach().float()

            profile("train_misc", epoch)
            losses["policy_loss"] += pg_loss.item() / self.total_minibatches
            losses["value_loss"] += v_loss.item() / self.total_minibatches
            losses["entropy"] += entropy_loss.item() / self.total_minibatches
            losses["old_approx_kl"] += old_approx_kl.item() / self.total_minibatches
            losses["approx_kl"] += approx_kl.item() / self.total_minibatches
            losses["clipfrac"] += clipfrac.item() / self.total_minibatches
            losses["importance"] += ratio.mean().item() / self.total_minibatches
            losses["past_kl"] += past_kl.item() / self.total_minibatches
            losses["past_kl_coef"] += self.past_kl_coef / self.total_minibatches
            losses["past_kl_term"] += (
                self.past_kl_coef * past_kl
            ).item() / self.total_minibatches
            losses["uniform_kl"] += uniform_kl.item() / self.total_minibatches
            losses["uniform_kl_coef"] += float(uniform_kl_coef) / self.total_minibatches
            losses["uniform_kl_term"] += (
                uniform_kl_coef * uniform_kl
            ).item() / self.total_minibatches
            losses["regularization_term"] += (
                self.past_kl_coef * past_kl + uniform_kl_coef * uniform_kl
            ).item() / self.total_minibatches

            profile("learn", epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), config["max_grad_norm"]
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

        profile("train_misc", epoch)
        if config["anneal_lr"]:
            self.scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
        losses["explained_variance"] = explained_var.item()

        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config["total_timesteps"]
        if (
            done_training
            or self.global_step == 0
            or time.time() > self.last_log_time + 0.25
        ):
            self.losses = losses
            logs = self.mean_and_log()
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config["checkpoint_interval"] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f"Checkpoint saved at update {self.epoch}"

        return logs


@dataclass
class PromotionStats:
    """Track repeated promotion batches until the best-checkpoint test is decisive.

    Promotion is intentionally more strict than the regular plotting eval. We keep a
    running estimate of the per-game win score, where wins count as 1.0, draws count as
    0.5, and losses count as 0.0. That score matches the existing evaluation metric, so
    the confidence gate is easy to explain and directly comparable to the logged plots.

    The class stores only aggregate moments instead of every individual result. That keeps
    the promotion loop cheap even when we need many parallel batches to separate two close
    checkpoints.
    """

    games: int = 0
    batches: int = 0
    win_score_sum: float = 0.0
    win_score_sq_sum: float = 0.0
    score_diff_sum: float = 0.0

    def update(self, win_scores: list[float], score_diffs: list[float]) -> None:
        """Merge one finished promotion batch into the running summary.

        Each batch is produced by a fully parallel head-to-head evaluation over the
        autotuned vector layout. The batch may contain any number of finished games, so
        this method accepts lists rather than assuming a fixed batch size.
        """

        if len(win_scores) != len(score_diffs):
            raise ValueError("win_scores and score_diffs must have matching lengths")
        if not win_scores:
            return

        self.games += len(win_scores)
        self.batches += 1
        self.win_score_sum += float(sum(win_scores))
        self.win_score_sq_sum += float(sum(score * score for score in win_scores))
        self.score_diff_sum += float(sum(score_diffs))

    @property
    def mean_win_rate(self) -> float:
        """Return the average win score from the current policy's perspective."""

        if self.games <= 0:
            return 0.0
        return self.win_score_sum / float(self.games)

    @property
    def mean_score_diff(self) -> float:
        """Return the average goal difference from the current policy's perspective."""

        if self.games <= 0:
            return 0.0
        return self.score_diff_sum / float(self.games)

    def lower_confidence_bound(self, confidence: float) -> float:
        """Return a one-sided lower confidence bound for the running win score.

        We use a normal approximation on the sample mean because the per-game score is
        bounded and the promotion loop only makes decisions after several parallel batches.
        Returning negative infinity for tiny samples prevents accidental promotion before we
        have enough evidence to justify replacing the best checkpoint.
        """

        if not 0.5 < confidence < 1.0:
            raise ValueError("confidence must be strictly between 0.5 and 1.0")
        if self.games < 2:
            return float("-inf")

        mean = self.mean_win_rate
        variance_numerator = self.win_score_sq_sum - float(self.games) * mean * mean
        sample_variance = max(0.0, variance_numerator / float(self.games - 1))
        standard_error = math.sqrt(sample_variance / float(self.games))
        z_score = NormalDist().inv_cdf(confidence)
        return mean - z_score * standard_error


class HeadToHeadEvaluator:
    """Run repeated current-vs-opponent soccer matches over a reusable vector env.

    Training now evaluates against both the immediately previous iterate and the best
    checkpoint stored in W&B. Reusing one evaluation environment avoids repeatedly creating
    vector workers during training, which matters when the autotuner has selected a large
    parallel layout for the current machine.

    The evaluator always reports results from the current policy's point of view. To avoid
    side bias, half the envs place the current policy on blue and the other half place it on
    red. That mirrors the existing past-iterate evaluation behavior while making the helper
    generic enough for best-checkpoint promotion tests as well.
    """

    def __init__(
        self,
        *,
        players_per_team: int,
        game_length: int,
        vec_config: VecEnvConfig,
        device: str,
    ):
        """Create the reusable evaluation environment and agent index maps."""

        self.device = device
        self.eval_env = make_soccer_vecenv(
            players_per_team=players_per_team,
            vec=vec_config,
            action_mode="discrete",
            game_length=game_length,
            render_mode=None,
            seed=0,
            log_interval=1,
            opponents_enabled=True,
        )
        self.opponent_policy = Policy(self.eval_env).to(device)
        self.num_envs = self.eval_env.num_envs
        self.num_players = players_per_team * 2
        self.current_on_blue = make_side_assignment(self.num_envs)
        current_agent_mask = np.zeros((self.eval_env.num_agents,), dtype=bool)
        for env_idx in range(self.num_envs):
            start = env_idx * self.num_players
            split = start + players_per_team
            end = start + self.num_players
            if self.current_on_blue[env_idx]:
                current_agent_mask[start:split] = True
            else:
                current_agent_mask[split:end] = True

        self.current_indices = torch.as_tensor(
            np.flatnonzero(current_agent_mask), dtype=torch.long, device=device
        )
        self.opponent_indices = torch.as_tensor(
            np.flatnonzero(~current_agent_mask), dtype=torch.long, device=device
        )
        self.action_buf = np.zeros((self.eval_env.num_agents,), dtype=np.int32)

    def run_games(
        self,
        current_policy: Policy,
        opponent: Mapping[str, torch.Tensor] | Any,
        num_games: int,
        seed: int,
    ) -> tuple[list[float], list[float]]:
        """Play a fixed number of games and return per-game win scores and score diffs.

        Returning raw per-game outcomes keeps the helper flexible. The regular training
        plots can average the results directly, while the promotion loop can keep batching
        them until its confidence target is met.
        """

        if isinstance(opponent, Mapping):
            self.opponent_policy.load_state_dict(opponent, strict=True)
            opponent_runner: Any = self.opponent_policy
        else:
            opponent_runner = opponent
        obs, _ = self.eval_env.reset(seed=seed)
        self.eval_env.flush_log()

        was_training_current = current_policy.training
        current_policy.eval()
        if hasattr(opponent_runner, "eval"):
            opponent_runner.eval()

        completed_games = 0
        score_diffs: list[float] = []
        win_scores: list[float] = []
        with torch.no_grad():
            while completed_games < num_games:
                obs_tensor = torch.as_tensor(
                    obs, device=self.device, dtype=torch.float32
                )

                current_logits, _ = forward_policy_eval(
                    current_policy,
                    obs_tensor.index_select(0, self.current_indices)
                )
                opponent_logits, _ = forward_policy_eval(
                    opponent_runner,
                    obs_tensor.index_select(0, self.opponent_indices)
                )
                current_actions = (
                    torch.argmax(current_logits, dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False)
                )
                opponent_actions = (
                    torch.argmax(opponent_logits, dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False)
                )

                self.action_buf[self.current_indices.cpu().numpy()] = current_actions
                self.action_buf[self.opponent_indices.cpu().numpy()] = opponent_actions

                _, _, terminals, truncations, _ = self.eval_env.step(self.action_buf)
                done_envs = np.flatnonzero(
                    terminals.reshape(self.num_envs, self.num_players).all(axis=1)
                    | truncations.reshape(self.num_envs, self.num_players).all(axis=1)
                )
                for env_idx in done_envs:
                    if completed_games >= num_games:
                        break
                    final_goals = self.eval_env.get_last_episode_scores(int(env_idx))
                    if final_goals is None:
                        continue
                    score_diff, win_score = score_metrics_from_perspective(
                        final_goals[0],
                        final_goals[1],
                        bool(self.current_on_blue[env_idx]),
                    )
                    score_diffs.append(score_diff)
                    win_scores.append(win_score)
                    completed_games += 1
                obs = self.eval_env.observations

        if was_training_current:
            current_policy.train()

        return win_scores, score_diffs

    def evaluate(
        self,
        current_policy: Policy,
        opponent: Mapping[str, torch.Tensor] | Any,
        num_games: int,
        seed: int,
    ) -> dict[str, float]:
        """Return mean head-to-head metrics for a fixed number of evaluation games."""

        win_scores, score_diffs = self.run_games(
            current_policy, opponent, num_games=num_games, seed=seed
        )
        return summarize_match_results(win_scores, score_diffs)

    def close(self) -> None:
        """Release the reusable evaluation environment."""

        self.eval_env.close()


class BlueTeamNoOpponentWrapper(pufferlib.PufferEnv):
    """Expose only blue-team trajectories while preserving the native no-opponent simulator.

    The native no-opponent env disables red gameplay internally, but it still exposes red agent
    slots to PPO. Those red slots never move and never receive reward, so half of every warm-
    start rollout becomes dead experience that only adds noise to the sparse scoring task.

    This wrapper keeps the simulator untouched and instead fixes the training interface. PPO
    only sees the blue agents that can act and learn, while the wrapped env still owns the full
    game state, the scoreboard, rendering, and field-scaling behavior. Blue observations keep
    the same feature shape as self-play, including zero-filled opponent slots, so warm-start and
    self-play still train the same policy architecture.
    """

    def __init__(self, env: pufferlib.PufferEnv, players_per_team: int) -> None:
        """Wrap one no-opponent env and expose only the controllable blue agents.

        The wrapper relies on the soccer env's fixed ordering where each env shard lists the
        blue team first and the red team second. That gives us one static index map for both
        slicing outputs and re-expanding blue actions back into the full native action array.
        """

        if getattr(env, "opponents_enabled", True):
            raise ValueError(
                "BlueTeamNoOpponentWrapper requires opponents_enabled=False"
            )
        if players_per_team < 1:
            raise ValueError("players_per_team must be positive")

        self.env = env
        self.players_per_team = players_per_team
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.opponents_enabled = False
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space
        self.num_agents = self.num_envs * self.players_per_team
        self._blue_indices = self._build_blue_indices()
        self._full_action_template = np.zeros_like(env.actions)

        super().__init__()

    def _build_blue_indices(self) -> np.ndarray:
        """Return the flat wrapped-env indices that correspond to blue players only.

        Building the index once keeps the hot rollout path allocation-free and makes the
        wrapper's semantics explicit: we are not inventing new agent identities, only filtering
        the native ordering down to the blue team in each environment shard.
        """

        full_agents = int(getattr(self.env, "num_agents", 0))
        full_agents_per_env = self.players_per_team * 2
        expected_full_agents = self.num_envs * full_agents_per_env
        if full_agents != expected_full_agents:
            raise ValueError(
                "wrapped env does not match the expected blue-then-red layout: "
                f"full_agents={full_agents}, expected_full_agents={expected_full_agents}"
            )

        blue_indices = np.empty((self.num_agents,), dtype=np.int32)
        write_ptr = 0
        for env_idx in range(self.num_envs):
            env_start = env_idx * full_agents_per_env
            env_stop = env_start + self.players_per_team
            blue_indices[write_ptr : write_ptr + self.players_per_team] = np.arange(
                env_start, env_stop, dtype=np.int32
            )
            write_ptr += self.players_per_team
        return blue_indices

    def _copy_blue_outputs(self) -> None:
        """Refresh the wrapper buffers from the wrapped env after each transition.

        The native env remains the authoritative owner of observations and episode state. The
        wrapper simply mirrors the blue rows into its own PPO-facing buffers so training only
        consumes trajectories for the agents that can actually influence the warm-start task.
        """

        self.observations[:] = self.env.observations[self._blue_indices]
        self.rewards[:] = self.env.rewards[self._blue_indices]
        self.terminals[:] = self.env.terminals[self._blue_indices]
        self.truncations[:] = self.env.truncations[self._blue_indices]
        self.masks[:] = self.env.masks[self._blue_indices]

    def _expand_blue_actions(self, blue_actions: np.ndarray) -> np.ndarray:
        """Expand blue-only actions into the full native action array expected by the env.

        The wrapped no-opponent env ignores red actions entirely, so zero is a safe filler for
        all hidden red slots. Reusing one persistent full-size buffer avoids repeated allocation
        inside the rollout loop.
        """

        expanded_actions = self._full_action_template
        expanded_actions.fill(0)
        expanded_actions[self._blue_indices] = blue_actions
        return expanded_actions

    def reset(self, seed: int | None = 0):
        """Reset the wrapped env and surface only the blue-team observation rows."""

        self.env.reset(seed=seed)
        self._copy_blue_outputs()
        return self.observations, []

    def step(self, actions: np.ndarray):
        """Step the wrapped env using only blue actions and return only blue outputs.

        This is the critical warm-start fix. PPO provides actions for controllable blue agents
        only; the wrapper expands them to the full native layout, steps the simulator, and then
        slices the resulting transition back down to blue trajectories before returning it.
        """

        expanded_actions = self._expand_blue_actions(actions)
        _, _, _, _, infos = self.env.step(expanded_actions)
        self._copy_blue_outputs()
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def set_field_scale(self, scale: float) -> None:
        """Forward map-curriculum updates and refresh the blue-team PPO view immediately."""

        setter = getattr(self.env, "set_field_scale", None)
        if setter is None:
            raise RuntimeError(
                "wrapped environment does not support set_field_scale()"
            )
        setter(scale)
        self._copy_blue_outputs()

    def get_state(self, env_idx: int = 0) -> dict[str, Any]:
        """Return the full wrapped-env state for scoring diagnostics and rendering."""

        getter = getattr(self.env, "get_state", None)
        if getter is None:
            raise RuntimeError("wrapped environment does not expose get_state()")
        return getter(env_idx)

    def get_last_episode_scores(
        self, env_idx: int = 0, clear: bool = True
    ) -> tuple[int, int] | None:
        """Forward final score queries to the wrapped env without changing semantics."""

        getter = getattr(self.env, "get_last_episode_scores", None)
        if getter is None:
            raise RuntimeError(
                "wrapped environment does not expose get_last_episode_scores()"
            )
        return getter(env_idx, clear)

    def render(self, env_idx: int = 0):
        """Render through the wrapped env so warm-start videos still show the full field."""

        renderer = getattr(self.env, "render", None)
        if renderer is None:
            return None
        return renderer(env_idx)

    def flush_log(self) -> dict[str, float] | None:
        """Forward env-log flushing so dashboards still reflect the native simulator log."""

        flusher = getattr(self.env, "flush_log", None)
        if flusher is None:
            return None
        return flusher()

    def close(self) -> None:
        """Close the wrapped env when the blue-only training view is finished."""

        self.env.close()


def parse_no_opponent_scale_ladder(raw_value: str) -> tuple[float, ...]:
    """Parse one comma-separated warm-start ladder into immutable field-scale stages.

    The no-opponent warm-start now uses a solved-stage ladder rather than a time-based growth
    curve. Representing the ladder as a single CLI string keeps batch-script overrides simple,
    while parsing it once here ensures the rest of the training code can work with a validated,
    immutable tuple of floats instead of repeatedly splitting raw strings.

    Empty stages are rejected because they usually come from accidental trailing commas and can
    silently turn a simple curriculum typo into confusing runtime behavior later in training.
    """

    stages = [part.strip() for part in raw_value.split(",")]
    if not stages or any(not stage for stage in stages):
        raise ValueError("no-opponent-map-scale-ladder must list one or more scales")
    return tuple(float(stage) for stage in stages)


@dataclass(frozen=True)
class NoOpponentCurriculumConfig:
    """Describe the discrete map ladder used for no-opponent warm-start training.

    The old warm-start grew the map on a fixed time schedule. That made the task harder even
    when the policy had not yet learned to score on the current map. The new curriculum is
    stage-based instead: the trainer stays on one explicit scale until evaluation clears the
    configured goal-rate threshold, then advances to the next rung in the ladder.

    Keeping the ladder in one frozen config object makes validation, logging, and stage lookup
    simple and keeps the main training loop focused on the promotion logic instead of on raw
    argparse plumbing.
    """

    stage_scales: tuple[float, ...]

    @classmethod
    def from_args(cls, args) -> "NoOpponentCurriculumConfig":
        """Build the active stage ladder from the parsed training CLI namespace."""

        return cls(
            stage_scales=parse_no_opponent_scale_ladder(
                args.no_opponent_map_scale_ladder
            )
        )

    def validate(self) -> None:
        """Reject invalid ladder settings with clear user-facing errors.

        A stage ladder is only useful when each rung is explicit and well ordered. The trainer
        therefore validates that the ladder is non-empty, that every scale lies inside the
        native env's legal range, and that the ladder is strictly increasing so stage-advance
        logs always correspond to a real change in task difficulty.
        """

        if not self.stage_scales:
            raise ValueError("no-opponent-map-scale-ladder must contain at least one stage")

        previous = 0.0
        for index, scale in enumerate(self.stage_scales):
            if not 0.1 <= scale <= 1.0:
                raise ValueError("every no-opponent map scale must be in [0.1, 1.0]")
            if index > 0 and scale <= previous:
                raise ValueError(
                    "no-opponent-map-scale-ladder must be strictly increasing"
                )
            previous = scale

    def enabled(self) -> bool:
        """Return whether the ladder changes the map at least once during warm-start."""

        return len(self.stage_scales) > 1 or self.stage_scales[0] != 1.0

    def initial_scale(self) -> float:
        """Return the first map scale used at the start of warm-start training."""

        return float(self.stage_scales[0])

    def stage_count(self) -> int:
        """Return how many explicit stages the warm-start ladder contains."""

        return len(self.stage_scales)

    def field_scale(self, stage_index: int) -> float:
        """Return the field scale for one explicit ladder stage.

        The curriculum now advances through fixed stages rather than through a continuous
        progress function. Using a stage index here keeps the training loop honest about that
        change and prevents old time-based terminology from creeping back into the code.
        """

        if stage_index < 0 or stage_index >= len(self.stage_scales):
            raise ValueError("invalid no-opponent curriculum stage index")
        return float(self.stage_scales[stage_index])


@dataclass(frozen=True)
class NoOpponentPhaseConfig:
    """Describe the short no-opponent warm-start phase that precedes self-play.

    The user wants every normal training run to begin with a brief sparse-reward drill:
    remove the opponent team, grow the field with the existing curriculum, and stop that
    warm-start once the policy can score repeatedly with high reliability. Keeping the
    stopping rules in one immutable object makes the phase easy to validate, log, and pass
    around without scattering raw argparse fields through the training loop.

    This config is intentionally about *phase control*, not environment geometry. The map
    scaling itself still lives in ``NoOpponentCurriculumConfig`` so the same field-growth
    logic can stay focused on one job while this class answers a different question:
    how much of the total PPO budget should be spent on the warm-start, and when is the
    policy good enough to move on to full self-play?
    """

    min_iterations: int
    max_iterations: int
    eval_interval: int
    goal_rate_threshold: float
    multi_goal_rate_threshold: float

    @classmethod
    def from_args(cls, args) -> "NoOpponentPhaseConfig":
        """Build the no-opponent phase controls from the parsed training CLI.

        This helper exists so ``main`` can ask for one coherent phase policy instead of
        repeatedly reaching into ``args`` for several related fields. That keeps the phase
        setup readable and gives tests one stable constructor path to exercise.
        """

        return cls(
            min_iterations=int(args.no_opponent_phase_min_iterations),
            max_iterations=int(args.no_opponent_phase_max_iterations),
            eval_interval=int(args.no_opponent_phase_eval_interval),
            goal_rate_threshold=float(args.no_opponent_phase_goal_rate_threshold),
            multi_goal_rate_threshold=float(
                args.no_opponent_phase_multi_goal_rate_threshold
            ),
        )

    def validate(self, total_ppo_iterations: int) -> None:
        """Reject invalid warm-start settings with clear, phase-specific errors.

        The warm-start is supposed to be fast and predictable. Invalid settings such as a
        negative iteration cap or thresholds outside ``[0, 1]`` tend to fail later in the
        run in much less obvious ways, so we validate them up front while the CLI context is
        still fresh and easy to understand.
        """

        if total_ppo_iterations < 1:
            raise ValueError("ppo-iterations must be positive")
        if self.min_iterations < 0:
            raise ValueError("no-opponent-phase-min-iterations must be non-negative")
        if self.max_iterations < 0:
            raise ValueError("no-opponent-phase-max-iterations must be non-negative")
        if self.max_iterations > 0 and self.min_iterations > self.max_iterations:
            raise ValueError(
                "no-opponent-phase-min-iterations must be less than or equal to "
                "no-opponent-phase-max-iterations"
            )
        if self.eval_interval < 1:
            raise ValueError("no-opponent-phase-eval-interval must be positive")
        if not 0.0 <= self.goal_rate_threshold <= 1.0:
            raise ValueError(
                "no-opponent-phase-goal-rate-threshold must be in [0.0, 1.0]"
            )
        if not 0.0 <= self.multi_goal_rate_threshold <= 1.0:
            raise ValueError(
                "no-opponent-phase-multi-goal-rate-threshold must be in [0.0, 1.0]"
            )
        if self.max_iterations > total_ppo_iterations:
            raise ValueError(
                "no-opponent-phase-max-iterations cannot exceed ppo-iterations"
            )

    def enabled(self) -> bool:
        """Return whether the training run should execute a warm-start phase at all.

        Using ``max_iterations`` as the on/off switch keeps the interface simple: the phase
        is active when it has any budget at all, and completely disabled when that cap is
        set to zero. That lets callers skip the warm-start without introducing a second,
        redundant boolean flag.
        """

        return self.max_iterations > 0

    def completion_reached(self, metrics: Mapping[str, float]) -> bool:
        """Return whether no-opponent evaluation is strong enough to exit the warm-start.

        The warm-start exit rule is intentionally configurable. Some experiments only need a
        reliable first goal before moving into harder play, while others may still require
        repeated scoring after post-goal resets. Keeping both thresholds in one helper lets
        the training loop ask one clear question regardless of which regime is active.
        """

        return (
            float(metrics["goal_rate"]) >= self.goal_rate_threshold
            and float(metrics["multi_goal_rate"]) >= self.multi_goal_rate_threshold
        )


def split_phase_iterations(
    total_ppo_iterations: int, no_opponent_phase: NoOpponentPhaseConfig
) -> tuple[int, int]:
    """Split the total PPO-iteration budget into warm-start and self-play segments.

    The requested workflow measures phase length in PPO iterations, not environment steps.
    That keeps the budget easy to reason about even when the warm-start uses a native env
    and self-play later switches to an autotuned multiprocessing layout with a different
    batch size. The returned pair always sums to the user-requested total iteration budget.
    """

    if total_ppo_iterations < 1:
        raise ValueError("total_ppo_iterations must be positive")
    if not no_opponent_phase.enabled():
        return 0, total_ppo_iterations

    warm_start_iterations = min(no_opponent_phase.max_iterations, total_ppo_iterations)
    self_play_iterations = max(0, total_ppo_iterations - warm_start_iterations)
    return warm_start_iterations, self_play_iterations


def phase_total_timestep_budget(
    requested_total_timesteps: int | None,
    *,
    total_ppo_iterations: int,
    phase_ppo_iterations: int,
) -> int | None:
    """Project an optional whole-run timestep budget onto one training phase.

    Most runs in this repo budget training in PPO iterations, but the trainer also supports
    an explicit total-timestep cap for fair comparisons. When the run now has two phases, we
    need a deterministic way to carve that global step budget into a warm-start portion and
    a self-play portion before each phase computes its own batch-aligned total.

    The split is intentionally proportional to the requested PPO-iteration allocation. Each
    phase later rounds its own share down to full update batches, which matches the existing
    whole-run behavior of never silently training past the requested step budget.
    """

    if requested_total_timesteps is None:
        return None
    if total_ppo_iterations < 1:
        raise ValueError("total_ppo_iterations must be positive")
    if phase_ppo_iterations < 0:
        raise ValueError("phase_ppo_iterations must be non-negative")
    if phase_ppo_iterations == 0:
        return 0
    return (requested_total_timesteps * phase_ppo_iterations) // total_ppo_iterations


def resolve_no_opponent_game_length(no_opponent_eval_max_steps: int) -> int:
    """Return the shared episode horizon used by every no-opponent task variant.

    Warm-start training, greedy no-opponent evaluation, and no-opponent replay export are
    all supposed to refer to the same sparse scoring task. If those paths use different
    episode lengths, evaluation can count goals that the trainer never had enough time to
    reach, which makes the warm-start gate fail for the wrong reason.

    This helper keeps the task definition aligned without adding another CLI flag. The
    no-opponent task should preserve the historical 400-step floor, but it should also grow
    to match any larger evaluation horizon so every no-opponent code path measures the same
    effective problem.
    """

    if no_opponent_eval_max_steps < 1:
        raise ValueError("no-opponent-eval-max-steps must be positive")
    return max(400, int(no_opponent_eval_max_steps))


def run_no_opponent_rollouts(
    policy: Policy,
    *,
    players_per_team: int,
    seed: int,
    device: str,
    num_games: int,
    max_steps: int,
    field_scale: float = 1.0,
) -> list[dict[str, float]]:
    """Run greedy no-opponent episodes in one native vector batch and summarize each game.

    Warm-start evaluation is now expected to run many games per check, so the old scalar loop
    would waste most of the native env's speed. This helper therefore builds one native vector
    no-opponent env with one shard per requested game, runs the whole batch in parallel, and
    then emits one summary per shard.

    `field_scale` is part of the task definition now. Promotion through the warm-start ladder
    depends on solving the current stage, so evaluation must run on the same active map size as
    training rather than silently grading the policy on a different field.
    """

    no_opponent_game_length = resolve_no_opponent_game_length(max_steps)
    eval_vec_config = VecEnvConfig(
        backend="native",
        shard_num_envs=max(1, int(num_games)),
        num_shards=1,
    )
    base_env = make_soccer_vecenv(
        players_per_team=players_per_team,
        vec=eval_vec_config,
        action_mode="discrete",
        game_length=no_opponent_game_length,
        render_mode=None,
        seed=seed,
        opponents_enabled=False,
    )
    env = BlueTeamNoOpponentWrapper(base_env, players_per_team)
    results: list[dict[str, float]] = []
    policy_was_training = policy.training
    policy.eval()

    try:
        if field_scale != 1.0:
            maybe_set_training_field_scale(env, field_scale)
        with torch.no_grad():
            obs, _ = env.reset(seed=seed)
            num_envs = env.num_envs
            first_goal_step = np.full((num_envs,), float(max_steps + 1), dtype=np.float32)
            blue_scored = np.zeros((num_envs,), dtype=bool)
            red_scored_first = np.zeros((num_envs,), dtype=bool)
            blue_goals = np.zeros((num_envs,), dtype=np.int32)
            red_goals = np.zeros((num_envs,), dtype=np.int32)

            for step_idx in range(1, max_steps + 1):
                obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
                logits, _ = policy.forward_eval(obs_tensor)
                actions = (
                    torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int32, copy=False)
                )
                obs, _, terminals, truncations, _ = env.step(actions)
                for env_idx in range(num_envs):
                    goals_blue, goals_red = env.get_state(env_idx)["goals"]
                    blue_goals[env_idx] = int(goals_blue)
                    red_goals[env_idx] = int(goals_red)
                    if blue_goals[env_idx] > 0 and not blue_scored[env_idx]:
                        blue_scored[env_idx] = True
                        first_goal_step[env_idx] = float(step_idx)
                    if red_goals[env_idx] > 0 and not blue_scored[env_idx]:
                        red_scored_first[env_idx] = True
                if bool(terminals.all() or truncations.all()):
                    break

            for env_idx in range(num_envs):
                results.append(
                    {
                        "goal_rate": float(int(blue_scored[env_idx])),
                        "multi_goal_rate": float(int(blue_goals[env_idx] >= 2)),
                        "mean_goals_scored": float(blue_goals[env_idx]),
                        "own_goal_rate": float(int(red_scored_first[env_idx])),
                        "mean_first_goal_step": float(first_goal_step[env_idx]),
                        "blue_goals": float(blue_goals[env_idx]),
                        "red_goals": float(red_goals[env_idx]),
                    }
                )
    finally:
        env.close()
        if policy_was_training:
            policy.train()

    return results


def evaluate_no_opponent_policy(
    policy: Policy,
    *,
    players_per_team: int,
    seed: int,
    device: str,
    num_games: int,
    max_steps: int,
    field_scale: float = 1.0,
) -> dict[str, float]:
    """Measure whether a policy solves the no-opponent scoring task repeatedly.

    The clean diagnostic the user wants is stricter than "can score once." In a world with
    no opponents, the learned team should keep finding the ball and scoring again after the
    native post-goal field reset. This evaluator therefore runs greedy rollouts and reports
    both first-goal speed and repeated-scoring throughput.

    Returned metrics:
    - `goal_rate`: fraction of games with at least one blue goal before the cutoff
    - `multi_goal_rate`: fraction of games with at least two blue goals before the cutoff
    - `mean_goals_scored`: average number of blue goals scored per game
    - `own_goal_rate`: fraction of games where the red side scores first
    - `mean_first_goal_step`: steps until the first blue goal, or `max_steps + 1` on failure
    """

    results = run_no_opponent_rollouts(
        policy,
        players_per_team=players_per_team,
        seed=seed,
        device=device,
        num_games=num_games,
        max_steps=max_steps,
        field_scale=field_scale,
    )

    return {
        "goal_rate": float(np.mean([result["goal_rate"] for result in results]))
        if results
        else 0.0,
        "multi_goal_rate": float(np.mean([result["multi_goal_rate"] for result in results]))
        if results
        else 0.0,
        "mean_goals_scored": float(np.mean([result["mean_goals_scored"] for result in results]))
        if results
        else 0.0,
        "own_goal_rate": float(np.mean([result["own_goal_rate"] for result in results]))
        if results
        else 0.0,
        "mean_first_goal_step": float(
            np.mean([result["mean_first_goal_step"] for result in results])
        )
        if results
        else float(max_steps + 1),
        "games": float(num_games),
    }


def maybe_set_training_field_scale(env: object, scale: float) -> None:
    """Apply one training-map scale when the active env implementation supports it.

    Map-size curriculum is now shared by both the no-opponent drill and normal self-play.
    Native scalar and native vector envs expose `set_field_scale`, while the Python vector
    wrappers do not. Keeping the capability check here makes the training loop itself easy
    to read and turns an unsupported backend choice into one clear runtime error instead of
    a silent no-op.
    """

    setter = getattr(env, "set_field_scale", None)
    if setter is None:
        raise RuntimeError(
            "training field curriculum requires an environment with set_field_scale()"
        )
    setter(scale)


def load_env_file(path: str = ".env") -> None:
    """Load simple `.env` assignments into the current process environment.

    Training often runs from shells, batch scripts, or remote workers that do not already have
    every secret exported. This helper lets the trainer pick up values such as
    ``WANDB_API_KEY`` directly from a local `.env` file before W&B is initialized.

    The parser intentionally stays lightweight, but it supports the two assignment styles most
    likely to appear in this repo:
    - ``KEY=value``
    - ``export KEY=value``

    Existing environment variables still win so callers can override local defaults from the
    outer shell when needed.
    """

    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def clone_state_dict(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone a policy state dict onto CPU tensors for safe reuse across eval calls.

    Evaluation opponents and promotion checks should not share live parameter storage with
    the training policy. Returning detached CPU clones keeps the snapshots stable even while
    training continues to update the active model on another device.
    """

    return {
        key: value.detach().cpu().clone() for key, value in policy.state_dict().items()
    }


def compute_eval_interval_epochs(total_epochs: int, fractions: int) -> int:
    """Convert an evaluation frequency expressed in fractions of training into epochs.

    The caller specifies how many evenly spaced checkpoints should be evaluated over the
    whole run. Clamping to one epoch keeps short smoke tests and tiny experiments from
    ending up with a zero interval.
    """

    return max(1, total_epochs // max(1, fractions))


def should_run_periodic_training_event(
    epoch: int, total_epochs: int, interval_epochs: int
) -> bool:
    """Return whether a shared periodic training event should run at this epoch.

    Training now uses one common cadence for three pieces of bookkeeping that must stay in
    lockstep: past-iterate evaluation, periodic self-play video logging, and refreshing the
    retained past-iterate snapshot used at the next comparison point. Keeping this decision in
    one helper avoids subtle drift where one call site treats the final epoch specially and
    another does not.

    The final epoch always returns ``True`` even when the total epoch count is not an exact
    multiple of the interval. That preserves the existing "always evaluate at the end" behavior
    and extends the same guarantee to periodic self-play video logging.
    """

    if epoch <= 0:
        return False
    if interval_epochs <= 0:
        raise ValueError("interval_epochs must be positive")
    return epoch % interval_epochs == 0 or epoch == total_epochs


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def compute_train_sizes(total_agents: int, horizon: int = 64) -> tuple[int, int]:
    """Return the PuffeRL batch and minibatch sizes for the active env layout.

    The training loop operates over all live agents in the vector environment at once,
    so the total number of agents directly determines how much experience we collect per
    rollout. This helper keeps that sizing logic in one place and enforces the horizon
    divisibility constraint that PuffeRL expects for truncated backpropagation.

    Keeping this function separate matters because the number of parallel environments may
    now come from the autotuner instead of a fixed CLI flag. The rest of training should
    be able to ask for stable train sizes without caring how the vector layout was chosen.
    """
    if total_agents < 1:
        raise ValueError("total_agents must be positive")

    batch_size = total_agents * horizon
    # PuffeRL requires minibatches to divide evenly into the BPTT horizon.
    minibatch_size = round_up_to_multiple(max(total_agents * 16, horizon), horizon)
    minibatch_size = min(minibatch_size, batch_size)
    return batch_size, minibatch_size


def choose_valid_minibatch_size(
    batch_size: int,
    horizon: int,
    requested_size: int,
    max_minibatch_size: int | None = None,
) -> int:
    """Project a requested minibatch size onto the nearest legal PuffeRL value.

    PuffeRL accepts only minibatches that line up with two structural constraints:
    they must be divisible by the rollout horizon, and they should divide the full
    training batch cleanly so every update consumes a whole number of sequence segments.
    The trainer can also impose a hard upper bound through ``max_minibatch_size``. That
    cap matters for large self-play layouts, where a tuned rollout ratio from a smaller
    run can scale into an otherwise legal minibatch that PuffeRL still rejects at runtime.

    This helper keeps all of those rules in one place. It takes the caller's requested
    size, clamps it into the feasible search region, and then returns the nearest legal
    candidate. Sharing the logic between CLI runs, autoloaded hyperparameters, and tuning
    code keeps rollout-size coercion consistent across every training entrypoint.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if batch_size % horizon != 0:
        raise ValueError("batch_size must be divisible by horizon")
    if max_minibatch_size is not None and max_minibatch_size < horizon:
        raise ValueError("max_minibatch_size must be at least one horizon")

    upper_bound = batch_size
    if max_minibatch_size is not None:
        upper_bound = min(upper_bound, max_minibatch_size)

    clamped = min(max(requested_size, horizon), upper_bound)
    candidates = [
        candidate
        for candidate in range(horizon, upper_bound + 1, horizon)
        if batch_size % candidate == 0 and candidate <= upper_bound
    ]
    if not candidates:
        raise ValueError(
            "expected at least one minibatch candidate divisible by batch and horizon "
            "under the configured minibatch cap"
        )

    return min(
        candidates,
        key=lambda candidate: (
            abs(candidate - clamped),
            candidate > clamped,
            candidate,
        ),
    )


def resolve_requested_train_sizes(
    total_agents: int,
    *,
    horizon: int,
    requested_batch_size: int | None,
    requested_minibatch_size: int | None,
    max_minibatch_size: int | None = None,
) -> tuple[int, int]:
    """Resolve legal rollout sizes for the active vector layout and trainer limits.

    This project now has three different sources of rollout-size requests: hand-written
    CLI flags, autoloaded defaults from earlier sweeps, and the old implicit "derive it
    from the vecenv" behavior. The active environment can also change between the short
    warm-start phase and the larger self-play phase, which means a previously tuned ratio
    may scale into a minibatch that PuffeRL refuses to train with.

    This helper keeps the resolution policy simple and stable. It preserves the existing
    default batch behavior, rounds requested batch sizes up to a whole-horizon multiple,
    and then projects the minibatch onto the nearest legal value that satisfies all active
    constraints: divisible by the horizon, a clean divisor of the batch, and optionally at
    or below PuffeRL's ``max_minibatch_size``. Returning a fully legal pair here keeps the
    later trainer construction path free of rollout-size surprises.
    """

    if horizon < 1:
        raise ValueError("horizon must be positive")

    default_batch_size, default_minibatch_size = compute_train_sizes(
        total_agents, horizon
    )
    if requested_batch_size is None and requested_minibatch_size is None:
        return default_batch_size, default_minibatch_size

    minimum_batch_size = total_agents * horizon
    batch_size = (
        default_batch_size if requested_batch_size is None else requested_batch_size
    )
    batch_size = round_up_to_multiple(max(batch_size, minimum_batch_size), horizon)

    requested_minibatch = (
        default_minibatch_size
        if requested_minibatch_size is None
        else requested_minibatch_size
    )
    minibatch_size = choose_valid_minibatch_size(
        batch_size=batch_size,
        horizon=horizon,
        requested_size=requested_minibatch,
        max_minibatch_size=max_minibatch_size,
    )
    return batch_size, minibatch_size


def resolve_total_timesteps(
    requested_total_timesteps: int | None, ppo_iterations: int, batch_size: int
) -> int:
    """Return the exact training budget that the current run should use.

    Standard training still supports the old `ppo_iterations * batch_size` behavior, but
    RL sweeps need a fixed sample budget so rollout-size knobs can be compared fairly.
    PuffeRL trains in whole-batch epochs, so a fixed step budget must be rounded to a
    whole number of batches. Rounding down keeps the budget from drifting upward for large
    batches and keeps cross-trial comparisons honest.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    if requested_total_timesteps is None:
        return ppo_iterations * batch_size

    if requested_total_timesteps < batch_size:
        raise ValueError(
            "requested_total_timesteps must be at least one full training batch"
        )

    full_batches = requested_total_timesteps // batch_size
    if full_batches < 1:
        raise ValueError("requested_total_timesteps did not leave any full batches")
    return full_batches * batch_size


def load_base_train_config() -> dict[str, Any]:
    """Load PufferLib's default train config without letting it parse this script's CLI.

    `pufferl.load_config()` inspects `sys.argv`, which would otherwise interpret this
    project's custom arguments as native PufferLib flags. Centralizing the temporary argv
    swap keeps that compatibility quirk in one place and makes both the normal trainer and
    the new hyperparameter tuning path share the exact same starting defaults.
    """

    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        cfg = pufferl.load_config("default")
    finally:
        sys.argv = argv
    return dict(cfg["train"])


def build_phase_train_config(
    args,
    vecenv,
    device: str,
    *,
    ppo_iterations: int,
    total_timesteps: int | None,
    opponents_enabled: bool,
) -> dict[str, Any]:
    """Assemble the concrete PuffeRL train config for one specific training phase.

    Training can now happen in two segments that share policy weights but not necessarily
    the same environment layout: a short native no-opponent warm-start followed by the main
    self-play run, which may use an autotuned multiprocessing vectorizer. Both phases still
    need the same rollout-size coercion and optimizer defaults, so this helper centralizes
    that work while accepting the phase-local budget explicitly.

    Passing the budget in rather than reading it directly from ``args`` is what makes the
    phase split reliable. The caller decides how much of the overall run belongs to the
    warm-start versus self-play, and this helper turns that decision into the exact
    batch-aligned PuffeRL config for the active environment.
    """

    phase_args = phase_args_for_training(args, opponents_enabled=opponents_enabled)
    config = load_base_train_config()
    max_minibatch_size = config.get("max_minibatch_size")
    requested_batch_size = phase_args.train_batch_size
    requested_minibatch_size = phase_args.minibatch_size
    if (
        phase_args._autoload_source_num_agents is not None
        and phase_args._autoload_batch_multiple is not None
        and phase_args._autoload_minibatch_divisor is not None
        and vecenv.num_agents != phase_args._autoload_source_num_agents
        and not phase_args._explicit_train_batch_size
        and not phase_args._explicit_minibatch_size
    ):
        requested_batch_size = (
            vecenv.num_agents
            * phase_args.bptt_horizon
            * phase_args._autoload_batch_multiple
        )
        requested_minibatch_size = max(
            phase_args.bptt_horizon,
            requested_batch_size // phase_args._autoload_minibatch_divisor,
        )
        print(
            "Adjusted autoloaded rollout sizes for the current agent count: "
            f"source_num_agents={phase_args._autoload_source_num_agents}, "
            f"current_num_agents={vecenv.num_agents}, "
            f"batch_multiple={phase_args._autoload_batch_multiple}, "
            f"minibatch_divisor={phase_args._autoload_minibatch_divisor}, "
            f"resolved_train_batch_size={requested_batch_size}, "
            f"resolved_minibatch_size={requested_minibatch_size}"
        )

    batch_size, minibatch_size = resolve_requested_train_sizes(
        vecenv.num_agents,
        horizon=phase_args.bptt_horizon,
        requested_batch_size=requested_batch_size,
        requested_minibatch_size=requested_minibatch_size,
        max_minibatch_size=max_minibatch_size,
    )
    if (
        requested_minibatch_size is not None
        and minibatch_size != requested_minibatch_size
        and max_minibatch_size is not None
        and requested_minibatch_size > max_minibatch_size
    ):
        print(
            "Clamped requested minibatch size to satisfy PuffeRL limits: "
            f"requested_minibatch_size={requested_minibatch_size}, "
            f"resolved_minibatch_size={minibatch_size}, "
            f"max_minibatch_size={max_minibatch_size}"
        )
    total_timesteps = resolve_total_timesteps(
        total_timesteps,
        ppo_iterations,
        batch_size,
    )
    config["batch_size"] = batch_size
    config["bptt_horizon"] = phase_args.bptt_horizon
    config["minibatch_size"] = minibatch_size
    config["total_timesteps"] = total_timesteps
    config["learning_rate"] = phase_args.learning_rate
    config["update_epochs"] = phase_args.update_epochs
    config["ent_coef"] = phase_args.ent_coef
    config["gamma"] = phase_args.gamma
    config["gae_lambda"] = phase_args.gae_lambda
    config["clip_coef"] = phase_args.clip_coef
    config["vf_coef"] = phase_args.vf_coef
    config["vf_clip_coef"] = phase_args.vf_clip_coef
    config["max_grad_norm"] = phase_args.max_grad_norm
    config["prio_alpha"] = phase_args.prio_alpha
    config["prio_beta0"] = phase_args.prio_beta0
    config["device"] = device
    config["seed"] = phase_args.seed
    config["env"] = "puffer_soccer_marl2d"
    if phase_args.checkpoint_interval is not None:
        config["checkpoint_interval"] = phase_args.checkpoint_interval
    return config


def build_train_config(args, vecenv, device: str) -> dict[str, Any]:
    """Assemble the concrete PuffeRL train config for a single-phase training run.

    Most unit tests and a few manual call sites still reason about training as one
    uninterrupted run. This thin wrapper preserves that interface while delegating the real
    work to ``build_phase_train_config``, which is now the canonical implementation used by
    both the warm-start phase and the main self-play phase.
    """

    return build_phase_train_config(
        args,
        vecenv,
        device,
        ppo_iterations=int(args.ppo_iterations),
        total_timesteps=args.total_timesteps,
        opponents_enabled=True,
    )


def summarize_effective_hyperparameters(
    train_config: Mapping[str, Any], args
) -> dict[str, int | float | bool | str | None]:
    """Capture the learning settings that define one finished training run.

    Hyperparameter tuning only helps if we can later explain which exact values won. This
    summary intentionally records the effective values after all coercion and defaulting,
    not just the raw CLI arguments. That makes the generated summary files reliable inputs
    for follow-up confirmation runs and for manual inspection after a long sweep.

    The no-opponent warm-start now depends on task-definition details in addition to PPO
    values, especially the shared episode horizon and whether the field curriculum is active
    by default. Recording those values here makes failed warm-start runs much easier to
    diagnose because the summary captures the actual task the policy saw during training.
    """

    return {
        "learning_rate": float(train_config["learning_rate"]),
        "gamma": float(train_config["gamma"]),
        "gae_lambda": float(train_config["gae_lambda"]),
        "update_epochs": int(train_config["update_epochs"]),
        "clip_coef": float(train_config["clip_coef"]),
        "vf_coef": float(train_config["vf_coef"]),
        "vf_clip_coef": float(train_config["vf_clip_coef"]),
        "max_grad_norm": float(train_config["max_grad_norm"]),
        "ent_coef": float(train_config["ent_coef"]),
        "prio_alpha": float(train_config["prio_alpha"]),
        "prio_beta0": float(train_config["prio_beta0"]),
        "train_batch_size": int(train_config["batch_size"]),
        "bptt_horizon": int(train_config["bptt_horizon"]),
        "minibatch_size": int(train_config["minibatch_size"]),
        "total_timesteps": int(train_config["total_timesteps"]),
        "regularization_enabled": not args.no_regularization,
        "past_kl_coef": float(args.past_kl_coef),
        "uniform_kl_base_coef": float(args.uniform_kl_base_coef),
        "uniform_kl_power": float(args.uniform_kl_power),
        "past_iterate_eval_fractions": int(args.past_iterate_eval_fractions),
        "no_opponent_phase_min_iterations": int(args.no_opponent_phase_min_iterations),
        "no_opponent_phase_max_iterations": int(args.no_opponent_phase_max_iterations),
        "no_opponent_phase_eval_interval": int(args.no_opponent_phase_eval_interval),
        "no_opponent_phase_goal_rate_threshold": float(
            args.no_opponent_phase_goal_rate_threshold
        ),
        "no_opponent_phase_multi_goal_rate_threshold": float(
            args.no_opponent_phase_multi_goal_rate_threshold
        ),
        "no_opponent_num_envs": int(args.no_opponent_num_envs),
        "no_opponent_map_scale_ladder": str(args.no_opponent_map_scale_ladder),
        "no_opponent_map_scale_start": float(args.no_opponent_map_scale_start),
        "no_opponent_map_scale_end": float(args.no_opponent_map_scale_end),
        "no_opponent_map_scale_power": float(args.no_opponent_map_scale_power),
        "no_opponent_map_scale_full_progress": float(
            args.no_opponent_map_scale_full_progress
        ),
        "no_opponent_field_curriculum_enabled": bool(field_curriculum_enabled(args)),
        "no_opponent_eval_games": int(args.no_opponent_eval_games),
        "no_opponent_eval_max_steps": int(args.no_opponent_eval_max_steps),
        "no_opponent_training_game_length": int(
            resolve_no_opponent_game_length(args.no_opponent_eval_max_steps)
        ),
    }


def standardized_hyperparameter_defaults(
    effective_hyperparameters: Mapping[str, object],
) -> dict[str, object]:
    """Extract the stable training defaults that should be reusable across future runs.

    The tuner evaluates one very specific training setup, but the follow-up runs we care
    about often change the mode around those learned defaults. For example, a no-opponent
    sweep should be able to seed a later self-play run without silently forcing
    `--no-opponent-team` or overwriting the caller's training budget. This helper keeps the
    reusable part narrow: PPO settings plus the map curriculum values.

    Returning a filtered mapping here gives both the tuner and the trainer one shared
    definition of "autoloadable hyperparameters". That avoids the common drift where the
    sweep saves one set of keys while the training script expects another.
    """

    defaults: dict[str, object] = {}
    for key in STANDARD_HYPERPARAMETER_KEYS:
        if key in effective_hyperparameters:
            defaults[key] = effective_hyperparameters[key]
    return defaults


def standardized_vecenv_defaults(
    vecenv_defaults: Mapping[str, object],
) -> dict[str, object]:
    """Extract the reusable vecenv defaults that are safe to autoload later.

    Runtime tuning for SPS is much more hardware-specific than PPO tuning, so we keep this
    payload deliberately narrow: only the parser fields that define the training vector
    layout itself. That lets a one-time benchmark save the exact backend and worker layout
    for a fixed Slurm machine without accidentally freezing unrelated settings such as the
    training budget, device selection, or team size.

    Keeping the filter in one helper matters for long-lived batch workflows. The pretune
    script, the autoload parser, and any future maintenance tools all need one shared
    definition of which vecenv keys belong in the standardized JSON record.
    """

    defaults: dict[str, object] = {}
    for key in STANDARD_VECENV_KEYS:
        if key in vecenv_defaults:
            defaults[key] = vecenv_defaults[key]
    return defaults


def vecenv_defaults_from_benchmark(result: BenchmarkResult) -> dict[str, object]:
    """Translate one benchmark winner into the exact CLI defaults training understands.

    The autotuner measures vector layouts in terms of shards and per-shard environment
    counts, while the human-facing training CLI still speaks in `--num-envs`,
    `--vec-num-shards`, and `--vec-batch-size`. This adapter keeps that impedance mismatch
    out of the batch scripts by converting the measured winner into the same parser fields
    that normal training already uses.

    The returned mapping intentionally includes `None` for shard- and batch-specific fields
    when the winner is native. Persisting those nulls is important because it clears stale
    multiprocessing settings from earlier tuning runs instead of silently carrying them
    forward into a now-native configuration.
    """

    return standardized_vecenv_defaults(
        {
            "vec_backend": result.backend,
            "num_envs": result.total_envs,
            "vec_num_shards": None if result.backend == "native" else result.num_shards,
            "vec_batch_size": None if result.backend == "native" else result.batch_size,
        }
    )


def benchmark_record(result: BenchmarkResult) -> dict[str, object]:
    """Serialize one SPS benchmark result into compact JSON-safe metadata.

    The saved autoload file should explain not only which layout won, but also why we
    should trust that choice later. Recording the measured SPS and CPU utilization next to
    the chosen defaults gives us a lightweight audit trail that is easy to inspect after a
    long cluster run and easy to compare against later confirmation runs.

    The structure is intentionally flat and made only of JSON-native values so helper
    scripts can merge it into the standardized autoload file without any custom encoder.
    """

    return {
        "backend": result.backend,
        "num_envs": int(result.total_envs),
        "shard_num_envs": int(result.shard_num_envs),
        "num_shards": int(result.num_shards),
        "batch_size": (
            None if result.batch_size is None else int(result.batch_size)
        ),
        "players_per_team": int(result.players_per_team),
        "action_mode": str(result.action_mode),
        "sps": int(result.sps),
        "cpu_avg": None if result.cpu_avg is None else float(result.cpu_avg),
        "cpu_peak": None if result.cpu_peak is None else float(result.cpu_peak),
    }


def load_standardized_hyperparameter_defaults(path: Path) -> dict[str, object]:
    """Read reusable train and vecenv defaults from disk when a standardized file exists.

    Automatic loading should make everyday training easier, not more fragile. The file is
    therefore optional: if it does not exist, training falls back to the script defaults and
    command-line flags exactly as before. When the file does exist, only the curated
    `train_defaults` and `vecenv_defaults` payloads are applied so that metadata and
    experiment-specific fields do not leak into argparse.

    The loader intentionally accepts both the new standardized record and the older plain
    dictionary layout. That backward-compatibility lets existing experiments keep working
    while the repo converges on one canonical format.
    """

    if not path.exists():
        return {}

    payload = read_json_record(path)
    if payload is None:
        return {}
    defaults = payload.get("train_defaults", payload)
    if not isinstance(defaults, Mapping):
        raise TypeError(f"expected mapping in hyperparameter file: {path}")
    merged_defaults = standardized_hyperparameter_defaults(defaults)
    merged_defaults.update(standardized_vecenv_defaults(payload))
    vecenv_defaults = payload.get("vecenv_defaults")
    if isinstance(vecenv_defaults, Mapping):
        merged_defaults.update(standardized_vecenv_defaults(vecenv_defaults))
    rollout_defaults = payload.get("rollout_defaults")
    if isinstance(rollout_defaults, Mapping):
        source_num_agents = rollout_defaults.get("source_num_agents")
        batch_multiple = rollout_defaults.get("batch_multiple")
        minibatch_divisor = rollout_defaults.get("minibatch_divisor")
        if source_num_agents is not None:
            merged_defaults["_autoload_source_num_agents"] = int(source_num_agents)
        if batch_multiple is not None:
            merged_defaults["_autoload_batch_multiple"] = int(batch_multiple)
        if minibatch_divisor is not None:
            merged_defaults["_autoload_minibatch_divisor"] = int(minibatch_divisor)
    source = payload.get("source")
    if isinstance(source, Mapping):
        source_label = source.get("label")
        source_path = source.get("path")
        if source_label is not None:
            merged_defaults["_autoload_source_label"] = str(source_label)
        if source_path is not None:
            merged_defaults["_autoload_source_path"] = str(source_path)
    return merged_defaults


def base_training_arg_defaults() -> dict[str, object]:
    """Return the built-in argparse defaults for the training CLI.

    Self-play occasionally needs to discard autoloaded no-opponent PPO values while still
    preserving the curriculum defaults from the same file. Reconstructing the parser
    defaults through one helper avoids hard-coding a second copy of those baseline values.
    """

    parser = build_training_parser(default_overrides=None)
    return {
        key: parser.get_default(flag.removeprefix("--").replace("-", "_"))
        for key, flag in EXPLICIT_ARG_FLAGS.items()
    }


def autoload_source_is_no_opponent(args) -> bool:
    """Return whether the active autoload file came from no-opponent tuning.

    The no-opponent sweep solves a different problem than the later self-play phase. Its
    curriculum defaults remain useful for warm-start, but its PPO defaults proved much too
    aggressive once opponents are enabled. The standardized autoload file already records
    provenance metadata, so this helper turns that metadata into one clear phase decision.
    """

    for field_name in ("_autoload_source_label", "_autoload_source_path"):
        value = getattr(args, field_name, None)
        if isinstance(value, str):
            normalized = value.lower().replace("-", "_")
            if "no_opponent" in normalized:
                return True
    return False


def phase_args_for_training(args, *, opponents_enabled: bool):
    """Return the effective argparse namespace for one specific training phase.

    The intended workflow is asymmetric: the no-opponent phase should keep the no-opponent
    sweep defaults, while the self-play phase should only inherit those PPO settings when a
    human asked for them explicitly on the CLI. Returning a shallow copy lets the caller
    adjust one phase cleanly without mutating the shared parsed namespace.
    """

    phase_args = copy.copy(args)
    if not opponents_enabled or not autoload_source_is_no_opponent(args):
        return phase_args

    parser_defaults = base_training_arg_defaults()
    reset_keys: list[str] = []
    for key in SELF_PLAY_RESETTABLE_AUTOLOAD_KEYS:
        if getattr(args, f"_explicit_{key}", False):
            continue
        setattr(phase_args, key, parser_defaults[key])
        reset_keys.append(key)

    if reset_keys:
        phase_args._autoload_source_num_agents = None
        phase_args._autoload_batch_multiple = None
        phase_args._autoload_minibatch_divisor = None
        print(
            "Self-play phase ignored no-opponent autoloaded PPO defaults and restored "
            f"parser defaults for: {', '.join(reset_keys)}"
        )
    return phase_args


def write_standardized_hyperparameters(
    *,
    path: Path,
    effective_hyperparameters: Mapping[str, object],
    source_path: Path | None = None,
    source_label: str | None = None,
    source_num_agents: int | None = None,
    vecenv_defaults: Mapping[str, object] | None = None,
    vecenv_benchmark: Mapping[str, object] | None = None,
) -> None:
    """Write the canonical reusable hyperparameter file consumed by normal training.

    The point of this file is operational simplicity. After a sweep finishes, later
    training jobs should not need to copy a long list of CLI flags by hand. The record we
    write here is intentionally small and explicit: versioned metadata plus the filtered
    `train_defaults` block that `train_pufferl.py` knows how to autoload.

    The optional source metadata is included because long RL jobs are expensive. When a
    training run later picks up this file, we need an easy way to trace which sweep output
    produced the defaults without digging through commit history or logs. The optional
    vecenv payload serves the same operational goal for SPS pretuning: once a fixed Slurm
    machine has been benchmarked, later jobs should be able to reuse that winner without
    rerunning autotune or editing the batch script by hand.
    """

    source_payload: dict[str, object] = {}
    if source_path is not None:
        source_payload["path"] = str(source_path)
    if source_label is not None:
        source_payload["label"] = source_label

    rollout_defaults: dict[str, object] = {}
    if source_num_agents is not None:
        horizon_value = effective_hyperparameters["bptt_horizon"]
        batch_size_value = effective_hyperparameters["train_batch_size"]
        minibatch_size_value = effective_hyperparameters["minibatch_size"]
        for key, value in (
            ("bptt_horizon", horizon_value),
            ("train_batch_size", batch_size_value),
            ("minibatch_size", minibatch_size_value),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise TypeError(f"expected integer-like value for {key}")
        horizon = int(cast(int | float | str, horizon_value))
        batch_size = int(cast(int | float | str, batch_size_value))
        minibatch_size = int(cast(int | float | str, minibatch_size_value))
        denom = max(1, source_num_agents * horizon)
        batch_multiple = max(1, batch_size // denom)
        minibatch_divisor = max(1, batch_size // max(1, minibatch_size))
        rollout_defaults = {
            "source_num_agents": int(source_num_agents),
            "batch_multiple": int(batch_multiple),
            "minibatch_divisor": int(minibatch_divisor),
        }

    write_json_record(
        path,
        {
            "format_version": 1,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": source_payload,
            "train_defaults": standardized_hyperparameter_defaults(
                effective_hyperparameters
            ),
            "rollout_defaults": rollout_defaults,
            "vecenv_defaults": standardized_vecenv_defaults(vecenv_defaults or {}),
            "vecenv_benchmark": dict(vecenv_benchmark or {}),
        },
    )


def maybe_write_run_summary(path: Path | None, payload: Mapping[str, object]) -> None:
    """Persist one run summary when the caller asked for machine-readable output.

    The tuning driver launches many independent training subprocesses. Writing a compact
    JSON summary at the end of each run avoids fragile log parsing and gives the sweep a
    simple contract: if the file exists, the trial finished cleanly and recorded its final
    objective metrics.
    """

    if path is None:
        return
    write_json_record(path, payload)


def field_curriculum_enabled(args) -> bool:
    """Return whether the active CLI settings request a multi-stage warm-start ladder.

    The no-opponent curriculum is stage-based now, so the natural enablement question is
    whether the parsed ladder changes the field at least once. Keeping this helper centralized
    avoids duplicating raw ladder parsing across training, summaries, and tests.
    """

    ladder_text = getattr(args, "no_opponent_map_scale_ladder", None)
    if ladder_text is not None:
        stages = parse_no_opponent_scale_ladder(str(ladder_text))
        return len(stages) > 1 or stages[0] != 1.0

    return (
        float(getattr(args, "no_opponent_map_scale_start"))
        != float(getattr(args, "no_opponent_map_scale_end"))
    )


def resolve_no_opponent_num_envs(args) -> int:
    """Return the native env count used by the warm-start-only no-opponent phase.

    The warm-start drill is intentionally different from self-play. It benefits from a wide,
    simple native rollout that collects many blue-only scoring attempts quickly, even when the
    later self-play phase uses fewer envs or an autotuned multiprocessing layout. Keeping that
    decision in one helper makes the warm-start data budget explicit in logs and avoids tying it
    accidentally to the self-play vecenv configuration.
    """

    requested = int(args.no_opponent_num_envs)
    if requested < 1:
        raise ValueError("no-opponent-num-envs must be positive")
    return requested


def resolve_vec_config(args) -> VecEnvConfig:
    """Build a manual vector-environment configuration from explicit CLI settings.

    This path is only used when the caller selects a concrete backend such as native,
    serial, or multiprocessing. It preserves the existing direct-control behavior where
    the user chooses the exact environment layout up front rather than asking the runtime
    to benchmark several layouts first.

    The autotuned `auto` backend intentionally does not go through this helper because it
    needs to search over many candidate layouts before deciding which configuration should
    be used for training.
    """
    if args.vec_backend == "native":
        return VecEnvConfig(
            backend="native",
            shard_num_envs=args.num_envs,
            num_shards=1,
        )

    num_shards = args.vec_num_shards or min(args.num_envs, physical_cpu_count())
    num_shards = max(1, min(num_shards, args.num_envs))
    if args.num_envs % num_shards != 0:
        raise ValueError(
            "num_envs must be divisible by vec_num_shards for non-native backends"
        )
    batch_size = args.vec_batch_size or num_shards
    shard_num_envs = args.num_envs // num_shards
    return VecEnvConfig(
        backend=args.vec_backend,
        shard_num_envs=shard_num_envs,
        num_shards=num_shards,
        num_workers=num_shards if args.vec_backend == "multiprocessing" else None,
        batch_size=batch_size,
    )


def autotune_training_vec_config(args) -> tuple[VecEnvConfig, BenchmarkResult]:
    """Benchmark self-play vector layouts before training and return the chosen config.

    The auto backend is intended to optimize the long self-play portion of training, which
    is where almost all PPO iterations are spent. The short no-opponent warm-start may still
    use a native environment for curriculum control, but that should not distort the vector
    layout chosen for the main run. This helper therefore always tunes for the self-play
    stage rather than inheriting temporary restrictions from the warm-start phase.

    Returning both the concrete ``VecEnvConfig`` and the benchmark that produced it keeps the
    eventual runtime choice transparent in logs and summaries, which is especially important
    when autotune selects multiprocessing on one machine and native on another.
    """
    requested_backend = "auto"
    if args.vec_backend != "auto":
        requested_backend = args.vec_backend

    outcome = autotune_vecenv(
        players_per_team=args.players_per_team,
        seconds=args.autotune_seconds,
        action_mode="discrete",
        backend=requested_backend,
        max_num_envs=args.autotune_max_num_envs,
        max_num_shards=args.autotune_max_num_shards,
        reporter=print,
    )
    return vec_config_from_benchmark(outcome.best), outcome.best


def resolve_training_vec_config(args) -> tuple[VecEnvConfig, BenchmarkResult | None]:
    """Resolve the vector layout that training should use.

    Training supports two modes. Manual backends use the CLI values exactly, while the
    `auto` backend runs a short benchmark sweep before any training work starts and then
    commits to the best discovered layout for the remainder of the run.

    Returning the optional benchmark result makes the autotuned decision visible to the
    caller without forcing the manual path to invent placeholder benchmark data.
    """
    if args.vec_backend == "auto":
        return autotune_training_vec_config(args)
    return resolve_vec_config(args), None


def build_training_parser(
    default_overrides: Mapping[str, object] | None = None,
) -> argparse.ArgumentParser:
    """Build the full training CLI, optionally starting from sweep-produced defaults.

    The trainer now has three sources of configuration: built-in defaults, an optional
    standardized hyperparameter file written by the sweep, and the explicit command line for
    the current run. Building the parser in one helper lets us apply that precedence cleanly:
    base defaults first, autoloaded defaults second, and final CLI flags last.

    `default_overrides` only changes argparse defaults. It does not bypass normal parsing or
    validation, which keeps the resulting behavior easy to reason about and consistent with
    manual CLI use.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--autoload-hyperparameters",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hyperparameters-path",
        type=str,
        default=str(STANDARD_HYPERPARAMETERS_PATH),
    )
    parser.add_argument(
        "--_autoload-source-num-agents",
        dest="_autoload_source_num_agents",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_autoload-batch-multiple",
        dest="_autoload_batch_multiple",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_autoload-minibatch-divisor",
        dest="_autoload_minibatch_divisor",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_autoload-source-label",
        dest="_autoload_source_label",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_autoload-source-path",
        dest="_autoload_source_path",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--no-opponent-num-envs", type=int, default=64)
    parser.add_argument(
        "--vec-backend",
        type=str,
        default="native",
        choices=["native", "serial", "multiprocessing", "auto"],
    )
    parser.add_argument("--vec-num-shards", type=int, default=None)
    parser.add_argument("--vec-batch-size", type=int, default=None)
    parser.add_argument("--autotune-max-num-envs", type=int, default=None)
    parser.add_argument("--autotune-max-num-shards", type=int, default=None)
    parser.add_argument("--autotune-seconds", type=float, default=0.75)
    parser.add_argument("--ppo-iterations", type=int, default=1000)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--bptt-horizon", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.90)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=2.0)
    parser.add_argument("--vf-clip-coef", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--prio-alpha", type=float, default=0.8)
    parser.add_argument("--prio-beta0", type=float, default=0.2)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--no-regularization", action="store_true")
    parser.add_argument("--no-opponent-phase-min-iterations", type=int, default=8)
    parser.add_argument("--no-opponent-phase-max-iterations", type=int, default=128)
    parser.add_argument("--no-opponent-phase-eval-interval", type=int, default=4)
    parser.add_argument(
        "--no-opponent-phase-goal-rate-threshold",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--no-opponent-phase-multi-goal-rate-threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--no-opponent-map-scale-ladder",
        type=str,
        default="0.2,0.4,0.6,0.8,1.0",
    )
    parser.add_argument("--no-opponent-map-scale-start", type=float, default=0.45)
    parser.add_argument("--no-opponent-map-scale-end", type=float, default=1.0)
    parser.add_argument("--no-opponent-map-scale-power", type=float, default=3.0)
    parser.add_argument(
        "--no-opponent-map-scale-full-progress",
        type=float,
        default=0.6,
    )
    parser.add_argument("--past-kl-coef", type=float, default=0.1)
    parser.add_argument("--uniform-kl-base-coef", type=float, default=0.05)
    parser.add_argument("--uniform-kl-power", type=float, default=0.3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", type=str, default="robot-soccer")
    parser.add_argument("--wandb-group", type=str, default="puffer-default")
    parser.add_argument("--wandb-tag", type=str, default=None)
    parser.add_argument("--wandb-video-key", type=str, default="self_play_video")
    parser.add_argument("--video-output", type=str, default=str(DEFAULT_VIDEO_OUTPUT_PATH))
    parser.add_argument(
        "--best-checkpoint-video-key",
        type=str,
        default="best_checkpoint_video",
    )
    parser.add_argument(
        "--best-checkpoint-video-output",
        type=str,
        default=str(DEFAULT_BEST_CHECKPOINT_VIDEO_OUTPUT_PATH),
    )
    parser.add_argument(
        "--export-videos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-max-steps", type=int, default=600)
    parser.add_argument(
        "--past-iterate-eval", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--past-iterate-eval-fractions", type=int, default=20)
    parser.add_argument("--past-iterate-eval-envs", type=int, default=None)
    parser.add_argument("--past-iterate-eval-games", type=int, default=64)
    parser.add_argument("--past-iterate-eval-game-length", type=int, default=400)
    parser.add_argument("--no-opponent-eval-games", type=int, default=100)
    parser.add_argument("--no-opponent-eval-max-steps", type=int, default=600)
    parser.add_argument("--final-best-eval-games", type=int, default=0)
    parser.add_argument(
        "--fixed-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--run-summary-path", type=str, default=None)
    parser.add_argument(
        "--best-checkpoint-config-path",
        type=str,
        default="experiments/best_checkpoint.json",
    )
    parser.add_argument(
        "--best-checkpoint-history-path",
        type=str,
        default="experiments/best_checkpoint_history.jsonl",
    )
    parser.add_argument(
        "--best-checkpoint-promotion-confidence",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--best-checkpoint-promotion-min-batches",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--best-checkpoint-promotion-max-batches",
        type=int,
        default=64,
    )
    if default_overrides:
        parser.set_defaults(**default_overrides)
    return parser


def parse_training_args(argv: list[str] | None = None):
    """Parse the training CLI after optionally loading standardized sweep defaults.

    This is a two-stage parse on purpose. We first read only the autoload controls so the
    caller can point at a different hyperparameter file or disable autoload entirely.
    After that, we build the full parser with any loaded defaults applied and parse the real
    command line normally.

    This approach keeps the everyday interface short: a batch script can stay minimal, while
    power users can still override any individual value on the CLI with the usual argparse
    precedence.
    """

    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--autoload-hyperparameters",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    bootstrap.add_argument(
        "--hyperparameters-path",
        type=str,
        default=str(STANDARD_HYPERPARAMETERS_PATH),
    )
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    default_overrides = (
        load_standardized_hyperparameter_defaults(
            Path(bootstrap_args.hyperparameters_path)
        )
        if bootstrap_args.autoload_hyperparameters
        else {}
    )
    parser = build_training_parser(default_overrides=default_overrides)
    args = parser.parse_args(argv)
    raw_args = sys.argv[1:] if argv is None else argv
    for key, flag in EXPLICIT_ARG_FLAGS.items():
        setattr(args, f"_explicit_{key}", flag in raw_args)
    if args.autoload_hyperparameters and default_overrides:
        print(
            "Loaded hyperparameter defaults from "
            f"{Path(args.hyperparameters_path)}"
        )
    return args


def make_side_assignment(num_envs: int) -> np.ndarray:
    """Split environments so the current policy plays both sides equally often.

    Head-to-head evaluation should not accidentally reward whichever team starts on the
    easier side of the field. This helper deterministically assigns the current policy to
    blue for the first half of the envs and to red for the remainder.
    """

    current_on_blue = np.zeros((num_envs,), dtype=bool)
    current_on_blue[: (num_envs + 1) // 2] = True
    return current_on_blue


def score_metrics_from_perspective(
    goals_blue: int, goals_red: int, current_on_blue: bool
) -> tuple[float, float]:
    """Convert final scores into goal difference and win score for one policy.

    The training plots already treat draws as half a win. Reusing that definition here keeps
    the regular eval curves and the promotion confidence test on the same scale.
    """

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


def summarize_match_results(
    win_scores: list[float], score_diffs: list[float]
) -> dict[str, float]:
    """Average per-game match outcomes into the scalar metrics logged to W&B.

    Evaluation code paths share the same summary format so the caller can log past-iterate
    results, best-checkpoint results, and promotion batches without each call site
    duplicating the same mean-and-empty-case handling.
    """

    games = min(len(win_scores), len(score_diffs))
    if games <= 0:
        return {"win_rate": 0.0, "score_diff": 0.0, "games": 0.0}
    return {
        "win_rate": float(np.mean(win_scores[:games])),
        "score_diff": float(np.mean(score_diffs[:games])),
        "games": float(games),
    }


def resolve_eval_vec_config(
    training_vec_config: VecEnvConfig, eval_envs_override: int | None
) -> VecEnvConfig:
    """Choose the vector layout used for head-to-head evaluation.

    By default we want eval to use as much safe parallelism as training. However, the current
    head-to-head evaluator relies on native env methods that expose final per-env score data.
    The Python multiprocessing backend does not surface those methods through the vector wrapper,
    so the automatic path falls back to a native eval env sized to the same total number of
    parallel environments. That keeps eval large and fast while preserving the score signals we
    need for promotion decisions and W&B plots.

    A native override is still supported for debugging and tests. Keeping it native makes
    the override easy to reason about because the user is specifying a direct environment
    count instead of a full shard layout.
    """

    if eval_envs_override is None:
        return VecEnvConfig(
            backend="native",
            shard_num_envs=total_sim_envs(training_vec_config),
            num_shards=1,
        )
    if eval_envs_override < 1:
        raise ValueError("eval_envs_override must be positive when provided")
    return VecEnvConfig(
        backend="native",
        shard_num_envs=eval_envs_override,
        num_shards=1,
    )


def evaluate_against_past_iterate(
    current_policy: Policy,
    previous_state_dict: dict[str, torch.Tensor],
    evaluator: HeadToHeadEvaluator,
    games: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate the current policy against the immediately previous policy snapshot.

    The wrapper keeps the main training loop readable while routing all of the actual match
    execution through the reusable evaluator. That lets past-iterate eval and best-checkpoint
    eval share the same environment setup and side-balancing logic.
    """

    return evaluator.evaluate(
        current_policy,
        previous_state_dict,
        num_games=games,
        seed=seed,
    )


def should_attempt_promotion(metrics: Mapping[str, float]) -> bool:
    """Return whether a lightweight best-checkpoint eval is strong enough to verify.

    Promotion matches are intentionally more expensive than the regular logging eval. We only
    launch them when the current policy already looks better than the stored best checkpoint,
    using score difference as a tie-break when the win score is exactly even.
    """

    win_rate = float(metrics.get("win_rate", 0.0))
    score_diff = float(metrics.get("score_diff", 0.0))
    return win_rate > 0.5 or (math.isclose(win_rate, 0.5) and score_diff > 0.0)


def run_promotion_evaluation(
    current_policy: Policy,
    best_opponent: Mapping[str, torch.Tensor] | Any,
    evaluator: HeadToHeadEvaluator,
    *,
    confidence: float,
    min_batches: int,
    max_batches: int,
    seed: int,
) -> dict[str, float]:
    """Batch promotion matches until the confidence gate fires or the cap is reached.

    The user requested that only promotion use a confidence-based stopping rule. Regular eval
    therefore stays fixed-size for easier-to-compare plots, while this helper keeps batching
    games over the fully parallel eval layout until we are confident enough to replace the
    current best checkpoint.
    """

    if min_batches < 1 or max_batches < min_batches:
        raise ValueError(
            "promotion batch limits must satisfy 1 <= min_batches <= max_batches"
        )

    stats = PromotionStats()
    games_per_batch = max(1, evaluator.num_envs)
    promoted = False
    lcb = float("-inf")
    for batch_idx in range(max_batches):
        win_scores, score_diffs = evaluator.run_games(
            current_policy,
            best_opponent,
            num_games=games_per_batch,
            seed=seed + batch_idx,
        )
        stats.update(win_scores, score_diffs)
        lcb = stats.lower_confidence_bound(confidence)
        if stats.batches >= min_batches and lcb > 0.5:
            promoted = True
            break

    return {
        "promoted": 1.0 if promoted else 0.0,
        "confidence": confidence,
        "games": float(stats.games),
        "batches": float(stats.batches),
        "games_per_batch": float(games_per_batch),
        "win_rate": stats.mean_win_rate,
        "score_diff": stats.mean_score_diff,
        "win_rate_lcb": lcb,
    }


def _json_default(value):
    """Convert simple path values before writing metadata files to disk."""

    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json_record(path: Path) -> dict[str, object] | None:
    """Read one JSON metadata record from disk if it exists.

    The best-checkpoint pointer lives in a local file so later runs know which W&B artifact
    currently represents the best model. Returning ``None`` for a missing file keeps startup
    simple for the very first run on a new machine.
    """

    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def write_json_record(path: Path, payload: Mapping[str, object]) -> None:
    """Persist one JSON metadata record atomically.

    Checkpoint metadata is consulted at the start of future training runs, so partial writes
    would be painful. Writing to a temporary file first avoids corrupting the pointer if the
    process is interrupted while updating the current best checkpoint.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    )
    os.replace(tmp_path, path)


def append_jsonl_record(path: Path, payload: Mapping[str, object]) -> None:
    """Append one immutable JSON Lines history record.

    The current-best file tells us what to use now, while the history file preserves every
    promotion event over time. JSON Lines keeps the format append-friendly and easy to inspect
    by hand after long training runs.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default))
        handle.write("\n")


def serialize_vec_config(vec_config: VecEnvConfig) -> dict[str, object]:
    """Convert a vector-layout choice into JSON-safe metadata."""

    return {
        "backend": vec_config.backend,
        "shard_num_envs": vec_config.shard_num_envs,
        "num_shards": vec_config.num_shards,
        "num_workers": vec_config.num_workers,
        "batch_size": vec_config.batch_size,
        "zero_copy": vec_config.zero_copy,
        "overwork": vec_config.overwork,
    }


def build_best_artifact_name(run_id: str, epoch: int) -> str:
    """Create a unique W&B artifact name for one promoted checkpoint snapshot."""

    return f"best-checkpoint-{run_id}-epoch-{epoch:06d}"


def build_artifact_ref(logger, artifact_name: str, alias: str = "best") -> str:
    """Build a fully qualified artifact reference for later download."""

    run = logger.wandb.run
    entity = getattr(run, "entity", None)
    project = getattr(run, "project", None)
    if entity:
        return f"{entity}/{project}/{artifact_name}:{alias}"
    return f"{project}/{artifact_name}:{alias}"


def upload_best_checkpoint_artifact(
    logger,
    checkpoint_path: Path,
    *,
    artifact_name: str,
    metadata: Mapping[str, object],
) -> str | None:
    """Upload one promoted checkpoint snapshot as a W&B model artifact."""

    if logger is None or not hasattr(logger, "wandb"):
        return None
    epoch_value = metadata.get("epoch", 0)
    epoch_alias = epoch_value if isinstance(epoch_value, int) else 0
    artifact = logger.wandb.Artifact(
        artifact_name,
        type="model",
        metadata=dict(metadata),
    )
    artifact.add_file(str(checkpoint_path), name="model.pt")
    logger.wandb.run.log_artifact(
        artifact,
        aliases=["best", f"epoch-{epoch_alias:06d}"],
    )
    return build_artifact_ref(logger, artifact_name, alias="best")


def resolve_checkpoint_file(path: Path) -> Path:
    """Locate the actual model checkpoint file inside a local path or artifact download."""

    if path.is_file():
        return path
    preferred = path / "model.pt"
    if preferred.exists():
        return preferred
    candidates = sorted(
        candidate
        for candidate in path.rglob("*.pt")
        if candidate.name != "trainer_state.pt"
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found under {path}")
    return candidates[0]


def download_artifact_checkpoint(logger, artifact_ref: str, cache_dir: Path) -> Path:
    """Download a W&B checkpoint artifact into the local cache directory."""

    if logger is None or not hasattr(logger, "wandb"):
        raise RuntimeError(
            "Cannot download a W&B artifact without an active wandb logger"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = artifact_ref.replace("/", "_").replace(":", "_").replace("\\", "_")
    download_root = cache_dir / safe_name
    artifact = logger.wandb.use_artifact(artifact_ref)
    downloaded_dir = Path(artifact.download(root=str(download_root)))
    return resolve_checkpoint_file(downloaded_dir)


def load_checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a policy state dict from a checkpoint file onto CPU memory."""

    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state dict checkpoint in {path}")
    return {str(name): tensor for name, tensor in state.items()}


def load_best_checkpoint_state(
    best_record: Mapping[str, object],
    *,
    logger,
    cache_dir: Path,
) -> tuple[dict[str, torch.Tensor], Path]:
    """Load the current best checkpoint using the local record plus W&B as needed."""

    artifact_ref = best_record.get("artifact_ref")
    if (
        isinstance(artifact_ref, str)
        and artifact_ref
        and logger is not None
        and hasattr(logger, "wandb")
    ):
        checkpoint_path = download_artifact_checkpoint(logger, artifact_ref, cache_dir)
        return load_checkpoint_state_dict(checkpoint_path), checkpoint_path

    cached_path = best_record.get("cached_checkpoint_path")
    if not isinstance(cached_path, str) or not cached_path:
        raise ValueError(
            "Best checkpoint record does not include a usable artifact ref or cache path"
        )
    checkpoint_path = resolve_checkpoint_file(Path(cached_path))
    return load_checkpoint_state_dict(checkpoint_path), checkpoint_path


def load_best_policy_bundle(
    best_record: Mapping[str, object] | None,
    *,
    device: str,
) -> tuple[torch.jit.ScriptModule, dict[str, object], Path] | None:
    """Load the self-contained best-model bundle when the pointer record exposes one.

    Newer best-checkpoint records carry a bundle directory that contains a TorchScript policy
    module. That module should be preferred over the raw checkpoint because it remains usable
    even if the live repo policy architecture changes later. Returning ``None`` keeps older
    records on the legacy checkpoint fallback path.
    """

    if not isinstance(best_record, Mapping):
        return None
    bundle_dir = policy_bundle.bundle_dir_from_record(dict(best_record))
    if bundle_dir is None:
        return None
    bundle_module, manifest = policy_bundle.load_policy_module_from_bundle(
        bundle_dir,
        device=device,
    )
    return bundle_module, manifest, bundle_dir


def current_timestamp() -> str:
    """Return a simple UTC timestamp string for metadata files."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def best_bundle_archive_dir(best_config_path: Path, artifact_name: str) -> Path:
    """Return the immutable archive directory for one saved best-model bundle.

    The canonical current-best path moves over time, but history records should still point at
    the exact bundle created for one promotion event. An artifact-scoped archive directory keeps
    that history durable while the separate ``current_best`` directory can continue acting as the
    stable pointer for later eval runs.
    """

    return best_config_path.parent / "baselines" / "archive" / artifact_name


def best_bundle_current_dir(best_config_path: Path) -> Path:
    """Return the canonical directory that always mirrors the active current best bundle.

    Users asked for one stable local folder they can evaluate against. This helper defines that
    folder in one place so training, export tools, and tests all share the same path contract.
    """

    return best_config_path.parent / "baselines" / "current_best"


def build_bundle_export_metadata(
    *,
    args,
    run_id: str,
    epoch: int,
    global_step: int,
    event: str,
    checkpoint_path: Path,
    previous_best: Mapping[str, object] | None,
    train_config: Mapping[str, Any] | None,
    objective_metrics: Mapping[str, float] | None,
    promotion_metrics: Mapping[str, float] | None,
) -> dict[str, object]:
    """Build the manifest payload written into one exported best-model bundle.

    The bundle should be understandable on its own long after the original run has finished.
    Recording the train args, eval settings, source checkpoint, git commit, and promotion data
    directly in the manifest makes the saved baseline far easier to inspect and trust later.
    """

    return {
        "run_id": run_id,
        "epoch": epoch,
        "global_step": global_step,
        "event": event,
        "git_commit": policy_bundle.current_git_commit(Path.cwd()),
        "original_checkpoint_path": str(checkpoint_path),
        "previous_best_artifact_ref": None
        if previous_best is None
        else previous_best.get("artifact_ref"),
        "train_args": {
            key: value
            for key, value in vars(args).items()
            if not key.startswith("_")
        },
        "effective_train_config": None if train_config is None else dict(train_config),
        "eval_settings": {
            "past_iterate_eval_games": int(args.past_iterate_eval_games),
            "past_iterate_eval_game_length": int(args.past_iterate_eval_game_length),
            "final_best_eval_games": int(args.final_best_eval_games),
            "promotion_confidence": float(args.best_checkpoint_promotion_confidence),
            "promotion_min_batches": int(args.best_checkpoint_promotion_min_batches),
            "promotion_max_batches": int(args.best_checkpoint_promotion_max_batches),
        },
        "objective_metrics": None
        if objective_metrics is None
        else dict(objective_metrics),
        "promotion_metrics": None
        if promotion_metrics is None
        else dict(promotion_metrics),
    }


def export_best_policy_bundle(
    *,
    policy: torch.nn.Module,
    checkpoint_state: dict[str, torch.Tensor],
    best_config_path: Path,
    artifact_name: str,
    observation_shape: tuple[int, ...],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Export both the immutable archive bundle and the canonical current-best bundle.

    The archive copy gives history records stable paths, while the canonical ``current_best``
    directory gives every future evaluation run one fixed location to load. Both bundles carry
    the same policy contents so the canonical pointer is just a movable mirror of the archive.
    """

    example_observation = policy_example_observation(policy, observation_shape)
    archive_result = policy_bundle.export_policy_bundle(
        policy=policy,
        checkpoint_state=checkpoint_state,
        bundle_dir=best_bundle_archive_dir(best_config_path, artifact_name),
        example_observation=example_observation,
        metadata=dict(metadata),
    )
    current_metadata = dict(metadata)
    current_metadata["archived_bundle_dir"] = archive_result["bundle_dir"]
    current_result = policy_bundle.export_policy_bundle(
        policy=policy,
        checkpoint_state=checkpoint_state,
        bundle_dir=best_bundle_current_dir(best_config_path),
        example_observation=example_observation,
        metadata=current_metadata,
    )
    return {
        "archive_bundle_dir": archive_result["bundle_dir"],
        "archive_bundle_manifest_path": archive_result["bundle_manifest_path"],
        "archive_bundle_policy_module_path": archive_result[
            "bundle_policy_module_path"
        ],
        "bundle_dir": current_result["bundle_dir"],
        "bundle_manifest_path": current_result["bundle_manifest_path"],
        "bundle_policy_module_path": current_result["bundle_policy_module_path"],
        "bundle_schema_version": current_result["bundle_schema_version"],
    }


def register_best_checkpoint(
    *,
    logger,
    checkpoint_path: Path,
    best_config_path: Path,
    best_history_path: Path,
    previous_best: Mapping[str, object] | None,
    vec_config: VecEnvConfig,
    run_id: str,
    epoch: int,
    global_step: int,
    event: str,
    promotion_metrics: Mapping[str, float] | None,
    policy: torch.nn.Module | None = None,
    observation_shape: tuple[int, ...] | None = None,
    bundle_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Persist and optionally upload a newly established best checkpoint.

    The record still carries the raw checkpoint metadata for backward compatibility, but when
    a live policy snapshot is available the function also exports a self-contained policy bundle
    and records its paths. That bundle is what later evals should prefer because it survives
    policy-architecture changes in the main repo.
    """

    artifact_name = build_best_artifact_name(run_id, epoch)
    metadata = {
        "run_id": run_id,
        "epoch": epoch,
        "global_step": global_step,
        "event": event,
        "previous_best_artifact_ref": None
        if previous_best is None
        else previous_best.get("artifact_ref"),
        "promotion_games": None
        if promotion_metrics is None
        else promotion_metrics.get("games"),
        "promotion_win_rate_vs_previous_best": None
        if promotion_metrics is None
        else promotion_metrics.get("win_rate"),
        "promotion_score_diff_vs_previous_best": None
        if promotion_metrics is None
        else promotion_metrics.get("score_diff"),
        "promotion_win_rate_lcb": None
        if promotion_metrics is None
        else promotion_metrics.get("win_rate_lcb"),
    }
    artifact_ref = upload_best_checkpoint_artifact(
        logger,
        checkpoint_path,
        artifact_name=artifact_name,
        metadata=metadata,
    )
    record = {
        "artifact_name": artifact_name,
        "artifact_ref": artifact_ref,
        "cached_checkpoint_path": str(checkpoint_path),
        "created_at": current_timestamp(),
        "epoch": epoch,
        "event": event,
        "global_step": global_step,
        "previous_best_artifact_ref": None
        if previous_best is None
        else previous_best.get("artifact_ref"),
        "promotion_batches": None
        if promotion_metrics is None
        else promotion_metrics.get("batches"),
        "promotion_confidence": None
        if promotion_metrics is None
        else promotion_metrics.get("confidence"),
        "promotion_games": None
        if promotion_metrics is None
        else promotion_metrics.get("games"),
        "promotion_win_rate_lcb": None
        if promotion_metrics is None
        else promotion_metrics.get("win_rate_lcb"),
        "promotion_win_rate_vs_previous_best": None
        if promotion_metrics is None
        else promotion_metrics.get("win_rate"),
        "promotion_score_diff_vs_previous_best": None
        if promotion_metrics is None
        else promotion_metrics.get("score_diff"),
        "run_id": run_id,
        "vec_config": serialize_vec_config(vec_config),
    }
    history_record = dict(record)
    if (
        policy is not None
        and observation_shape is not None
        and bundle_metadata is not None
    ):
        bundle_result = export_best_policy_bundle(
            policy=policy,
            checkpoint_state=snapshot_policy_state(policy),
            best_config_path=best_config_path,
            artifact_name=artifact_name,
            observation_shape=observation_shape,
            metadata=dict(bundle_metadata),
        )
        record.update(
            {
                "bundle_dir": bundle_result["bundle_dir"],
                "bundle_manifest_path": bundle_result["bundle_manifest_path"],
                "bundle_policy_module_path": bundle_result["bundle_policy_module_path"],
                "bundle_schema_version": bundle_result["bundle_schema_version"],
            }
        )
        history_record.update(
            {
                "bundle_dir": bundle_result["archive_bundle_dir"],
                "bundle_manifest_path": bundle_result[
                    "archive_bundle_manifest_path"
                ],
                "bundle_policy_module_path": bundle_result[
                    "archive_bundle_policy_module_path"
                ],
                "bundle_schema_version": bundle_result["bundle_schema_version"],
            }
        )
    else:
        record.update(
            {
                "bundle_dir": None,
                "bundle_manifest_path": None,
                "bundle_policy_module_path": None,
                "bundle_schema_version": None,
            }
        )
        history_record.update(
            {
                "bundle_dir": None,
                "bundle_manifest_path": None,
                "bundle_policy_module_path": None,
                "bundle_schema_version": None,
            }
        )
    write_json_record(best_config_path, record)
    append_jsonl_record(best_history_path, history_record)
    return record


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return name


def snapshot_policy_state(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot a policy state dict for later evaluation or promotion bookkeeping.

    This helper mirrors ``clone_state_dict`` but stays separate because call sites use it to
    describe intention more clearly: one path snapshots the current best checkpoint state, and
    another path snapshots the policy right before a training update.
    """

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.state_dict().items()
    }


def _configure_iterate_metrics(logger) -> None:
    """Tell W&B to chart periodic evaluation and video metrics against stable progress steps.

    The previous configuration still referenced an old metric namespace. Defining shared step
    metrics here keeps periodic evaluation plots and periodic self-play video metadata aligned on
    the same x-axis, which matters when later inspecting whether a behavioral shift first showed
    up in numeric evaluation or in the replay video.
    """

    if logger is None or not hasattr(logger, "wandb"):
        return
    logger.wandb.define_metric("evaluation/progress_step")
    logger.wandb.define_metric("evaluation/*", step_metric="evaluation/progress_step")
    logger.wandb.define_metric("video/progress_step")
    logger.wandb.define_metric("video/*", step_metric="video/progress_step")


def _is_permission_like_error(err: Exception) -> bool:
    message = str(err).lower()
    return (
        isinstance(err, PermissionError)
        or "permission denied" in message
        or "broken pipe" in message
    )


def _is_writable_target(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    try:
        if path.exists():
            with path.open("ab"):
                pass
        else:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".write_test_", delete=True
            ):
                pass
        return True
    except OSError:
        return False


def _unique_path(path: Path) -> Path:
    """Return the first unused sibling path by appending an increasing numeric suffix.

    Video export fallback paths should never fail just because a directory already contains a
    long history of older files. The previous fixed attempt cap turned a bookkeeping detail into
    a training crash once a shared artifact directory filled up. This helper now keeps scanning
    until it finds the next open slot so collision handling remains best-effort and non-fatal.
    """

    if not path.exists():
        return path

    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _resolve_writable_output_path(path: Path) -> Path:
    """Return a writable path for video export, falling back to sibling or temp paths.

    Training video logging is helpful for diagnosis, but it must not be brittle. This helper
    first tries the requested path directly, then a unique sibling when the original location is
    blocked, and finally a temp-directory path when the whole target directory is unavailable.
    That layered fallback keeps W&B uploads working even when the preferred local artifact path
    is unwritable.
    """

    expanded = path.expanduser()
    if _is_writable_target(expanded):
        return expanded

    candidate = _unique_path(expanded)
    if _is_writable_target(candidate):
        print(f"Video output not writable at {expanded}; using {candidate}")
        return candidate

    tmp_base = Path(tempfile.gettempdir()) / "puffer-soccer-videos" / expanded.name
    tmp_candidate = _unique_path(tmp_base)
    if _is_writable_target(tmp_candidate):
        print(
            f"Video output not writable at {expanded}; using temp path {tmp_candidate}"
        )
        return tmp_candidate

    raise PermissionError(
        f"No writable output path available for video export (requested: {expanded})"
    )


def build_run_video_tag(run_start_time: float, run_id: str | None = None) -> str:
    """Build the per-run artifact tag used for default local video output directories.

    The default video layout now groups artifacts by training run instead of dropping every
    replay into one shared `experiments/` directory. The tag combines the run start timestamp
    with the W&B run id when one exists, which keeps folders easy to read by eye while still
    separating repeated launches that happen on the same day.
    """

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(run_start_time))
    if run_id is None:
        return timestamp
    normalized_run_id = str(run_id).strip()
    return timestamp if not normalized_run_id else f"{timestamp}_{normalized_run_id}"


def run_video_directory(run_tag: str) -> Path:
    """Return the default per-run directory that stores locally written training videos.

    Only video artifacts move into the run-scoped layout for now. Keeping that policy in one
    helper makes the default directory structure explicit and avoids scattering stringly-typed
    `experiments/<tag>/video` assembly across the training and test code.
    """

    return Path("experiments") / run_tag / "video"


def canonical_run_video_path(run_tag: str, video_kind: str) -> Path:
    """Return the canonical default local path for one named video artifact kind.

    Periodic videos should have stable filenames inside a run so the latest replay overwrites the
    prior one rather than growing an unbounded local file list. Mapping the supported artifact
    kinds here keeps that naming consistent for self-play, no-opponent, and best-checkpoint
    exports.
    """

    filename = RUN_VIDEO_FILENAMES[video_kind]
    return run_video_directory(run_tag) / filename


def configure_run_video_outputs(args, *, logger, run_start_time: float) -> None:
    """Resolve default video outputs for this run while preserving explicit CLI overrides.

    The trainer still exposes `--video-output` and `--best-checkpoint-video-output` so manual
    workflows can pin artifacts to custom locations. When those flags are left at their defaults,
    though, we now redirect them into `experiments/<run-tag>/video/` using one timestamped run
    folder. Mutating `args` in place keeps the rest of the video pipeline simple because every
    downstream helper can continue reading the resolved paths from the same argparse fields.
    """

    run_id = None if logger is None else getattr(logger, "run_id", None)
    run_tag = build_run_video_tag(run_start_time, run_id=run_id)
    setattr(args, "run_video_tag", run_tag)

    if not getattr(args, "_explicit_video_output", False):
        args.video_output = str(canonical_run_video_path(run_tag, "self_play"))
    if not getattr(args, "_explicit_best_checkpoint_video_output", False):
        args.best_checkpoint_video_output = str(
            canonical_run_video_path(run_tag, "best_checkpoint")
        )


def print_match_summary(label: str, epoch: int, metrics: Mapping[str, float]) -> None:
    """Print one compact evaluation summary line for the active training epoch."""

    print(
        f"{label} (epoch={epoch}, games={int(metrics['games'])}): "
        f"win_rate={metrics['win_rate']:.3f}, score_diff={metrics['score_diff']:.3f}"
    )


def log_video_artifact(
    logger,
    video_key: str,
    video_path: Path,
    fps: int,
    step: int,
    extra_payload: Mapping[str, float] | None = None,
) -> None:
    """Upload one generated video to W&B together with any cadence metadata.

    Video artifacts are most useful when they can be lined up with the exact training progress
    point that produced them. Accepting an optional payload lets callers attach the shared
    periodic-step metadata used by numeric evaluation without duplicating the W&B logging logic
    at every call site.
    """

    if logger is None or not hasattr(logger, "wandb"):
        return
    video_format = "gif" if video_path.suffix.lower() == ".gif" else "mp4"
    payload: dict[str, Any] = {}
    if extra_payload is not None:
        payload.update(extra_payload)
    payload[video_key] = logger.wandb.Video(
        str(video_path),
        fps=fps,
        format=video_format,
    )
    logger.wandb.log(payload, step=step)


def log_periodic_self_play_video(
    policy: Policy,
    args,
    *,
    logger,
    epoch: int,
    global_step: int,
    eval_interval_epochs: int,
    baseline_epoch: int,
) -> Path | None:
    """Render and log the shared-cadence self-play video for one training checkpoint.

    Periodic self-play videos are intended to be inspected side by side with past-iterate
    evaluation metrics. This helper keeps the render, W&B upload, and cadence metadata tied
    together so callers cannot forget to log the video at the same progress point as the
    matching numeric evaluation.
    """

    if not args.export_videos:
        return None

    video_path = save_self_play_video(policy, args, overwrite_existing=True)
    if video_path is None:
        return None

    log_video_artifact(
        logger,
        args.wandb_video_key,
        video_path,
        args.video_fps,
        global_step,
        extra_payload={
            "video/progress_step": float(global_step),
            "video/self_play/current_epoch": float(epoch),
            "video/self_play/baseline_epoch": float(baseline_epoch),
            "video/self_play/eval_epochs_interval": float(eval_interval_epochs),
        },
    )
    return video_path


def log_periodic_no_opponent_video(
    policy: Policy,
    args,
    *,
    logger,
    epoch: int,
    global_step: int,
    field_scale: float = 1.0,
) -> Path | None:
    """Render and log one warm-start replay while the no-opponent phase is active.

    The warm-start is intentionally short, which means it is easy to lose visibility into
    whether the curriculum is actually teaching the intended behavior before self-play takes
    over. Logging a replay from that phase gives a direct qualitative check that complements
    the scoring metrics used for early exit. By default the video lands in the active run's
    `video/` directory and overwrites the previous warm-start replay from that same run so the
    trainer keeps one stable local artifact for W&B upload.
    """

    if not args.export_videos:
        return None

    no_opponent_path = canonical_run_video_path(
        getattr(args, "run_video_tag", build_run_video_tag(time.time())),
        "no_opponent",
    )
    if getattr(args, "_explicit_video_output", False):
        requested_path = Path(str(args.video_output))
        no_opponent_path = requested_path.with_name(
            f"{requested_path.stem}_no_opponent{requested_path.suffix}"
        )
    video_path = save_match_video(
        policy,
        args,
        output_path=no_opponent_path,
        label="no-opponent video",
        opponents_enabled=False,
        no_opponent_field_scale=field_scale,
        overwrite_existing=True,
    )
    if video_path is None:
        return None

    log_video_artifact(
        logger,
        args.wandb_video_key,
        video_path,
        args.video_fps,
        global_step,
        extra_payload={
            "video/progress_step": float(global_step),
            "video/no_opponent/current_epoch": float(epoch),
        },
    )
    return video_path


def _build_run_summary(
    *,
    args,
    trainer,
    train_config: Mapping[str, Any],
    eval_interval_epochs: int,
    vec_config: VecEnvConfig,
    eval_vec_config: VecEnvConfig | None,
    best_record: Mapping[str, object] | None,
    latest_best_metrics: Mapping[str, float] | None,
    final_best_metrics: Mapping[str, float] | None,
    latest_no_opponent_metrics: Mapping[str, float] | None,
    final_no_opponent_metrics: Mapping[str, float] | None,
    model_path: Path,
) -> dict[str, object]:
    """Build the machine-readable summary emitted at the end of one training run.

    The RL tuning workflow launches the trainer as a subprocess so each trial gets a fresh
    random seed and isolated resources. Returning Python objects alone would not be enough,
    because the parent process needs a stable file format that survives crashes and can be
    inspected later by hand. This helper gathers the final objective metrics and the exact
    effective hyperparameters into one compact JSON-ready payload.
    """

    objective_metrics = (
        dict(final_best_metrics)
        if final_best_metrics is not None
        else None
        if latest_best_metrics is None
        else dict(latest_best_metrics)
    )
    if objective_metrics is None:
        objective_metrics = (
            dict(final_no_opponent_metrics)
            if final_no_opponent_metrics is not None
            else None
            if latest_no_opponent_metrics is None
            else dict(latest_no_opponent_metrics)
        )
    no_opponent_stage_scales = list(
        parse_no_opponent_scale_ladder(str(args.no_opponent_map_scale_ladder))
    )
    return {
        "seed": int(args.seed),
        "runtime_seconds": float(max(0.0, time.time() - trainer.start_time)),
        "global_step": int(trainer.global_step),
        "epoch": int(trainer.epoch),
        "model_path": str(model_path),
        "fixed_best_checkpoint": bool(args.fixed_best_checkpoint),
        "best_checkpoint_record": None if best_record is None else dict(best_record),
        "effective_hyperparameters": summarize_effective_hyperparameters(
            train_config,
            args,
        ),
        "past_iterate_eval_interval_epochs": int(eval_interval_epochs),
        "objective_metrics": objective_metrics,
        "latest_best_checkpoint_metrics": None
        if latest_best_metrics is None
        else dict(latest_best_metrics),
        "final_best_checkpoint_metrics": None
        if final_best_metrics is None
        else dict(final_best_metrics),
        "latest_no_opponent_metrics": None
        if latest_no_opponent_metrics is None
        else dict(latest_no_opponent_metrics),
        "final_no_opponent_metrics": None
        if final_no_opponent_metrics is None
        else dict(final_no_opponent_metrics),
        "no_opponent_task_config": {
            "training_game_length": int(
                resolve_no_opponent_game_length(args.no_opponent_eval_max_steps)
            ),
            "eval_max_steps": int(args.no_opponent_eval_max_steps),
            "field_curriculum_enabled": bool(field_curriculum_enabled(args)),
            "goal_rate_threshold": float(args.no_opponent_phase_goal_rate_threshold),
            "multi_goal_rate_threshold": float(
                args.no_opponent_phase_multi_goal_rate_threshold
            ),
            "map_scale_ladder": no_opponent_stage_scales,
            "stage_count": len(no_opponent_stage_scales),
        },
        "vec_config": serialize_vec_config(vec_config),
        "eval_vec_config": None
        if eval_vec_config is None
        else serialize_vec_config(eval_vec_config),
    }


def main():
    """Run one training job with optional autoloaded defaults and map curriculum.

    This entrypoint now supports the normal day-to-day workflow for the repo: start from
    reasonable built-in defaults, optionally layer in the latest sweep-selected PPO and
    curriculum settings, and then let the explicit CLI for the current run override anything
    that needs to change. Keeping that precedence here means batch scripts can stay short
    without hiding where the final configuration came from.
    """
    args = parse_training_args()
    load_env_file(".env")
    run_start_time = time.time()

    no_opponent_curriculum = NoOpponentCurriculumConfig.from_args(args)
    no_opponent_curriculum.validate()
    no_opponent_phase = NoOpponentPhaseConfig.from_args(args)
    no_opponent_phase.validate(int(args.ppo_iterations))
    no_opponent_game_length = resolve_no_opponent_game_length(
        int(args.no_opponent_eval_max_steps)
    )

    device = resolve_device(args.device)
    best_config_path = Path(args.best_checkpoint_config_path)
    best_history_path = Path(args.best_checkpoint_history_path)
    best_cache_dir = best_config_path.parent / "wandb_artifacts"
    run_summary_path = (
        None if args.run_summary_path is None else Path(args.run_summary_path)
    )

    logger = None
    if args.wandb:
        logger_args = {
            "wandb_project": args.wandb_project,
            "wandb_group": args.wandb_group,
            "tag": args.wandb_tag,
        }
        logger = pufferl.WandbLogger(logger_args)
        _configure_iterate_metrics(logger)

    configure_run_video_outputs(args, logger=logger, run_start_time=run_start_time)

    best_record = read_json_record(best_config_path)
    best_bundle_module: Any | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_opponent: Mapping[str, torch.Tensor] | Any | None = None
    latest_best_metrics: dict[str, float] | None = None
    final_best_metrics: dict[str, float] | None = None
    latest_no_opponent_metrics: dict[str, float] | None = None
    final_no_opponent_metrics: dict[str, float] | None = None
    effective_train_config: dict[str, Any] | None = None
    effective_vec_config: VecEnvConfig | None = None
    eval_vec_config: VecEnvConfig | None = None
    eval_interval_epochs = 1
    model_path: Path | None = None
    final_trainer = None
    current_policy: Policy | None = None
    self_play_initial_state: dict[str, torch.Tensor] | None = None
    remaining_total_timesteps = args.total_timesteps

    warm_start_budget, _ = split_phase_iterations(
        int(args.ppo_iterations), no_opponent_phase
    )
    if warm_start_budget > 0:
        warm_start_num_envs = resolve_no_opponent_num_envs(args)
        warm_start_vec_config = VecEnvConfig(
            backend="native",
            shard_num_envs=warm_start_num_envs,
            num_shards=1,
        )
        warm_start_base_vecenv = make_soccer_vecenv(
            players_per_team=args.players_per_team,
            action_mode="discrete",
            game_length=no_opponent_game_length,
            render_mode=None,
            seed=args.seed,
            opponents_enabled=False,
            vec=warm_start_vec_config,
        )
        warm_start_vecenv = BlueTeamNoOpponentWrapper(
            warm_start_base_vecenv, args.players_per_team
        )
        print(
            "No-opponent vecenv config: "
            f"backend={warm_start_vec_config.backend}, "
            f"shard_num_envs={warm_start_vec_config.shard_num_envs}, "
            f"num_shards={warm_start_vec_config.num_shards}, "
            f"num_workers={warm_start_vec_config.num_workers}, "
            f"batch_size={warm_start_vec_config.batch_size}, "
            f"total_sim_envs={total_sim_envs(warm_start_vec_config)}, "
            f"full_num_agents={warm_start_base_vecenv.num_agents}, "
            f"controlled_num_agents={warm_start_vecenv.num_agents}"
        )
        initial_no_opponent_scale = 1.0
        if no_opponent_curriculum.enabled():
            initial_no_opponent_scale = no_opponent_curriculum.initial_scale()
            maybe_set_training_field_scale(
                warm_start_vecenv, initial_no_opponent_scale
            )
        print(
            "No-opponent task config: "
            f"game_length={no_opponent_game_length}, "
            f"eval_max_steps={int(args.no_opponent_eval_max_steps)}, "
            f"field_curriculum_enabled={no_opponent_curriculum.enabled()}, "
            "stage_scales="
            f"{','.join(f'{scale:.3f}' for scale in no_opponent_curriculum.stage_scales)}, "
            f"initial_applied_scale={initial_no_opponent_scale:.3f}"
        )
        warm_start_train_config = build_phase_train_config(
            args,
            warm_start_vecenv,
            device,
            ppo_iterations=warm_start_budget,
            total_timesteps=phase_total_timestep_budget(
                args.total_timesteps,
                total_ppo_iterations=int(args.ppo_iterations),
                phase_ppo_iterations=warm_start_budget,
            ),
            opponents_enabled=False,
        )
        print(
            "No-opponent train config: "
            f"controlled_num_agents={warm_start_vecenv.num_agents}, "
            f"batch_size={warm_start_train_config['batch_size']}, "
            f"bptt_horizon={warm_start_train_config['bptt_horizon']}, "
            f"minibatch_size={warm_start_train_config['minibatch_size']}, "
            f"total_timesteps={warm_start_train_config['total_timesteps']}, "
            f"learning_rate={warm_start_train_config['learning_rate']:.6g}, "
            f"update_epochs={warm_start_train_config['update_epochs']}, "
            f"ent_coef={warm_start_train_config['ent_coef']:.6g}"
        )

        current_policy = Policy(warm_start_vecenv).to(device)
        warm_start_trainer = RegularizedPuffeRL(
            warm_start_train_config,
            warm_start_vecenv,
            current_policy,
            logger=logger,
            regularization_enabled=False,
            past_kl_coef=args.past_kl_coef,
            uniform_kl_base_coef=args.uniform_kl_base_coef,
            uniform_kl_power=args.uniform_kl_power,
        )
        warm_start_eval_interval = max(
            1, min(no_opponent_phase.eval_interval, warm_start_trainer.total_epochs)
        )
        no_opponent_stage_index = 0

        while warm_start_trainer.epoch < warm_start_trainer.total_epochs:
            warm_start_trainer.evaluate()
            warm_start_trainer.train()
            should_run_periodic_event = should_run_periodic_training_event(
                warm_start_trainer.epoch,
                warm_start_trainer.total_epochs,
                warm_start_eval_interval,
            )
            if not should_run_periodic_event:
                continue

            train_scale = no_opponent_curriculum.field_scale(no_opponent_stage_index)
            no_opponent_metrics = evaluate_no_opponent_policy(
                current_policy,
                players_per_team=args.players_per_team,
                seed=args.seed + warm_start_trainer.epoch * 10_000,
                device=device,
                num_games=args.no_opponent_eval_games,
                max_steps=args.no_opponent_eval_max_steps,
                field_scale=train_scale,
            )
            latest_no_opponent_metrics = dict(no_opponent_metrics)
            print(
                "No-opponent eval "
                f"(epoch={warm_start_trainer.epoch}, games={int(no_opponent_metrics['games'])}): "
                f"stage={no_opponent_stage_index + 1}/{no_opponent_curriculum.stage_count()}, "
                f"train_scale={train_scale:.3f}, "
                f"goal_rate={no_opponent_metrics['goal_rate']:.3f}, "
                f"multi_goal_rate={no_opponent_metrics['multi_goal_rate']:.3f}, "
                f"mean_goals_scored={no_opponent_metrics['mean_goals_scored']:.2f}, "
                f"own_goal_rate={no_opponent_metrics['own_goal_rate']:.3f}, "
                f"mean_first_goal_step={no_opponent_metrics['mean_first_goal_step']:.2f}"
            )
            if logger is not None:
                logger.wandb.log(
                    {
                        "evaluation/progress_step": float(warm_start_trainer.global_step),
                        "evaluation/no_opponent/goal_rate": no_opponent_metrics[
                            "goal_rate"
                        ],
                        "evaluation/no_opponent/multi_goal_rate": no_opponent_metrics[
                            "multi_goal_rate"
                        ],
                        "evaluation/no_opponent/mean_goals_scored": no_opponent_metrics[
                            "mean_goals_scored"
                        ],
                        "evaluation/no_opponent/own_goal_rate": no_opponent_metrics[
                            "own_goal_rate"
                        ],
                        "evaluation/no_opponent/mean_first_goal_step": no_opponent_metrics[
                            "mean_first_goal_step"
                        ],
                        "evaluation/no_opponent/games": no_opponent_metrics["games"],
                        "evaluation/no_opponent/current_epoch": warm_start_trainer.epoch,
                    },
                    step=warm_start_trainer.global_step,
                )
            log_periodic_no_opponent_video(
                current_policy,
                args,
                logger=logger,
                epoch=warm_start_trainer.epoch,
                global_step=warm_start_trainer.global_step,
                field_scale=train_scale,
            )
            if (
                no_opponent_metrics["goal_rate"] >= no_opponent_phase.goal_rate_threshold
                and no_opponent_stage_index < no_opponent_curriculum.stage_count() - 1
            ):
                previous_scale = train_scale
                no_opponent_stage_index += 1
                next_scale = no_opponent_curriculum.field_scale(no_opponent_stage_index)
                maybe_set_training_field_scale(warm_start_vecenv, next_scale)
                print(
                    "Advanced no-opponent map stage: "
                    f"epoch={warm_start_trainer.epoch}, "
                    f"goal_rate={no_opponent_metrics['goal_rate']:.3f}, "
                    f"from_scale={previous_scale:.3f}, "
                    f"to_scale={next_scale:.3f}"
                )
                continue
            if (
                no_opponent_stage_index == no_opponent_curriculum.stage_count() - 1
                and
                warm_start_trainer.epoch >= no_opponent_phase.min_iterations
                and no_opponent_phase.completion_reached(no_opponent_metrics)
            ):
                final_no_opponent_metrics = dict(no_opponent_metrics)
                print(
                    "No-opponent phase complete: "
                    f"epoch={warm_start_trainer.epoch}, "
                    f"goal_rate={no_opponent_metrics['goal_rate']:.3f}, "
                    f"train_scale={train_scale:.3f}"
                )
                break

        if final_no_opponent_metrics is None:
            if (
                latest_no_opponent_metrics is None
                or not no_opponent_phase.completion_reached(latest_no_opponent_metrics)
            ):
                goal_rate = (
                    0.0
                    if latest_no_opponent_metrics is None
                    else float(latest_no_opponent_metrics["goal_rate"])
                )
                multi_goal_rate = (
                    0.0
                    if latest_no_opponent_metrics is None
                    else float(latest_no_opponent_metrics["multi_goal_rate"])
                )
                raise RuntimeError(
                    "No-opponent warm-start exhausted its budget without reaching the "
                    "required scoring thresholds. Refusing to continue into self-play with "
                    f"goal_rate={goal_rate:.3f}, "
                    f"multi_goal_rate={multi_goal_rate:.3f}, "
                    f"required_goal_rate={no_opponent_phase.goal_rate_threshold:.3f}, "
                    "required_multi_goal_rate="
                    f"{no_opponent_phase.multi_goal_rate_threshold:.3f}"
                )
            final_no_opponent_metrics = dict(latest_no_opponent_metrics)
        if logger is not None:
            logger.wandb.log(
                {
                    "evaluation/final_no_opponent/goal_rate": final_no_opponent_metrics[
                        "goal_rate"
                    ],
                    "evaluation/final_no_opponent/multi_goal_rate": final_no_opponent_metrics[
                        "multi_goal_rate"
                    ],
                    "evaluation/final_no_opponent/mean_goals_scored": final_no_opponent_metrics[
                        "mean_goals_scored"
                    ],
                    "evaluation/final_no_opponent/own_goal_rate": final_no_opponent_metrics[
                        "own_goal_rate"
                    ],
                    "evaluation/final_no_opponent/mean_first_goal_step": final_no_opponent_metrics[
                        "mean_first_goal_step"
                    ],
                    "evaluation/final_no_opponent/games": final_no_opponent_metrics[
                        "games"
                    ],
                    "evaluation/final_no_opponent/current_epoch": warm_start_trainer.epoch,
                    "evaluation/progress_step": float(warm_start_trainer.global_step),
                },
                step=warm_start_trainer.global_step,
            )
        self_play_initial_state = snapshot_policy_state(current_policy)
        remaining_total_timesteps = (
            None
            if args.total_timesteps is None
            else max(0, int(args.total_timesteps) - int(warm_start_trainer.global_step))
        )
        effective_train_config = warm_start_train_config
        effective_vec_config = warm_start_vec_config
        eval_interval_epochs = warm_start_eval_interval
        final_trainer = warm_start_trainer
        warm_start_trainer.print_dashboard()
        model_path = Path(warm_start_trainer.close())

    self_play_iterations = int(args.ppo_iterations) - (
        0 if final_trainer is None else int(final_trainer.epoch)
    )
    if self_play_iterations > 0:
        vec_config, autotune_result = resolve_training_vec_config(args)
        vecenv = make_soccer_vecenv(
            players_per_team=args.players_per_team,
            action_mode="discrete",
            game_length=400,
            render_mode=None,
            seed=args.seed,
            opponents_enabled=True,
            vec=vec_config,
        )
        if autotune_result is not None:
            print(
                "Autotune selected: "
                f"{format_benchmark_result(autotune_result)}\t"
                f"backend={autotune_result.backend}"
            )
        print(
            "Vecenv config: "
            f"backend={vec_config.backend}, "
            f"shard_num_envs={vec_config.shard_num_envs}, "
            f"num_shards={vec_config.num_shards}, "
            f"num_workers={vec_config.num_workers}, "
            f"batch_size={vec_config.batch_size}, "
            f"total_sim_envs={total_sim_envs(vec_config)}, "
            f"num_agents={vecenv.num_agents}"
        )

        train_config = build_phase_train_config(
            args,
            vecenv,
            device,
            ppo_iterations=self_play_iterations,
            total_timesteps=remaining_total_timesteps,
            opponents_enabled=True,
        )
        print(
            "Train config: "
            f"batch_size={train_config['batch_size']}, "
            f"bptt_horizon={train_config['bptt_horizon']}, "
            f"minibatch_size={train_config['minibatch_size']}, "
            f"total_timesteps={train_config['total_timesteps']}, "
            f"learning_rate={train_config['learning_rate']:.6g}, "
            f"update_epochs={train_config['update_epochs']}, "
            f"ent_coef={train_config['ent_coef']:.6g}"
        )

        current_policy = Policy(vecenv).to(device)
        if self_play_initial_state is not None:
            current_policy.load_state_dict(self_play_initial_state, strict=True)
            print(
                "Initialized self-play phase from no-opponent warm-start policy snapshot."
            )

        trainer = RegularizedPuffeRL(
            train_config,
            vecenv,
            current_policy,
            logger=logger,
            regularization_enabled=not args.no_regularization,
            past_kl_coef=args.past_kl_coef,
            uniform_kl_base_coef=args.uniform_kl_base_coef,
            uniform_kl_power=args.uniform_kl_power,
        )

        if best_record is not None:
            try:
                loaded_bundle = load_best_policy_bundle(best_record, device=device)
                if loaded_bundle is not None:
                    best_bundle_module, _bundle_manifest, best_bundle_dir = loaded_bundle
                    best_opponent = best_bundle_module
                    print(f"Loaded best policy bundle: {best_bundle_dir}")
                else:
                    best_state_dict, best_checkpoint_path = load_best_checkpoint_state(
                        best_record,
                        logger=logger,
                        cache_dir=best_cache_dir,
                    )
                    if (
                        best_record.get("cached_checkpoint_path")
                        != str(best_checkpoint_path)
                    ):
                        best_record = dict(best_record)
                        best_record["cached_checkpoint_path"] = str(best_checkpoint_path)
                        write_json_record(best_config_path, best_record)
                    best_opponent = best_state_dict
                    print(
                        "Loaded best checkpoint: "
                        f"{best_record.get('artifact_ref') or best_checkpoint_path}"
                    )
            except Exception as err:
                print(
                    "Best checkpoint load failed "
                    f"({err}); skipping best-checkpoint eval until a new best is registered."
                )
                best_record = None
                best_bundle_module = None
                best_state_dict = None
                best_opponent = None

        if args.fixed_best_checkpoint and best_opponent is None:
            raise RuntimeError(
                "fixed-best-checkpoint mode requires a readable best checkpoint record"
            )

        needs_evaluator = args.past_iterate_eval or args.final_best_eval_games > 0
        eval_vec_config = (
            resolve_eval_vec_config(vec_config, args.past_iterate_eval_envs)
            if needs_evaluator
            else None
        )
        evaluator = None
        if eval_vec_config is not None:
            evaluator = HeadToHeadEvaluator(
                players_per_team=args.players_per_team,
                game_length=args.past_iterate_eval_game_length,
                vec_config=eval_vec_config,
                device=device,
            )
        eval_interval_epochs = max(
            1,
            compute_eval_interval_epochs(
                trainer.total_epochs, args.past_iterate_eval_fractions
            ),
        )
        if evaluator is not None:
            assert eval_vec_config is not None
            print(
                "Eval vecenv config: "
                f"backend={eval_vec_config.backend}, "
                f"shard_num_envs={eval_vec_config.shard_num_envs}, "
                f"num_shards={eval_vec_config.num_shards}, "
                f"num_workers={eval_vec_config.num_workers}, "
                f"batch_size={eval_vec_config.batch_size}, "
                f"total_sim_envs={total_sim_envs(eval_vec_config)}"
            )

        past_iterate_state_dict = clone_state_dict(current_policy)
        past_iterate_epoch = 0

        while trainer.epoch < trainer.total_epochs:
            trainer.evaluate()
            trainer.train()
            should_run_periodic_event = should_run_periodic_training_event(
                trainer.epoch,
                trainer.total_epochs,
                eval_interval_epochs,
            )
            should_eval = args.past_iterate_eval and should_run_periodic_event
            if should_eval and evaluator is not None:
                eval_seed = args.seed + trainer.epoch * 10_000
                eval_metrics = evaluate_against_past_iterate(
                    current_policy,
                    past_iterate_state_dict,
                    evaluator=evaluator,
                    games=args.past_iterate_eval_games,
                    seed=eval_seed,
                )
                log_payload = {
                    "evaluation/progress_step": float(trainer.global_step),
                    "evaluation/past_iterate/win_rate": eval_metrics["win_rate"],
                    "evaluation/past_iterate/score_diff": eval_metrics["score_diff"],
                    "evaluation/past_iterate/games": eval_metrics["games"],
                    "evaluation/past_iterate/eval_epochs_interval": eval_interval_epochs,
                    "evaluation/past_iterate/baseline_epoch": float(past_iterate_epoch),
                    "evaluation/past_iterate/current_epoch": trainer.epoch,
                }
                print_match_summary("Past iterate eval", trainer.epoch, eval_metrics)

                if best_opponent is not None:
                    best_metrics = evaluator.evaluate(
                        current_policy,
                        best_opponent,
                        num_games=args.past_iterate_eval_games,
                        seed=eval_seed + 1_000,
                    )
                    log_payload.update(
                        {
                            "evaluation/best_checkpoint/win_rate": best_metrics["win_rate"],
                            "evaluation/best_checkpoint/score_diff": best_metrics[
                                "score_diff"
                            ],
                            "evaluation/best_checkpoint/games": best_metrics["games"],
                            "evaluation/best_checkpoint/eval_epochs_interval": eval_interval_epochs,
                            "evaluation/best_checkpoint/current_epoch": trainer.epoch,
                        }
                    )
                    latest_best_metrics = dict(best_metrics)
                    print_match_summary("Best checkpoint eval", trainer.epoch, best_metrics)
                    log_payload["evaluation/best_checkpoint/promotion_attempted"] = 0.0

                if logger is not None:
                    logger.wandb.log(log_payload, step=trainer.global_step)

            if should_run_periodic_event:
                log_periodic_self_play_video(
                    current_policy,
                    args,
                    logger=logger,
                    epoch=trainer.epoch,
                    global_step=trainer.global_step,
                    eval_interval_epochs=eval_interval_epochs,
                    baseline_epoch=past_iterate_epoch,
                )
                past_iterate_state_dict = snapshot_policy_state(current_policy)
                past_iterate_epoch = trainer.epoch

        trainer.print_dashboard()
        model_path = Path(trainer.close())
        final_trainer = trainer
        effective_train_config = train_config
        effective_vec_config = vec_config

        final_promotion_metrics: dict[str, float] | None = None
        if best_opponent is not None and args.final_best_eval_games > 0:
            if evaluator is None:
                raise RuntimeError(
                    "final best-checkpoint eval requested without an evaluator"
                )
            final_best_metrics = evaluator.evaluate(
                current_policy,
                best_opponent,
                num_games=args.final_best_eval_games,
                seed=args.seed + 50_000_000,
            )
            latest_best_metrics = dict(final_best_metrics)
            print_match_summary(
                "Final best checkpoint eval", trainer.epoch, final_best_metrics
            )
            if logger is not None:
                logger.wandb.log(
                    {
                        "evaluation/final_best_checkpoint/win_rate": final_best_metrics[
                            "win_rate"
                        ],
                        "evaluation/final_best_checkpoint/score_diff": final_best_metrics[
                            "score_diff"
                        ],
                        "evaluation/final_best_checkpoint/games": final_best_metrics[
                            "games"
                        ],
                        "evaluation/final_best_checkpoint/current_epoch": trainer.epoch,
                        "evaluation/progress_step": float(trainer.global_step),
                    },
                    step=trainer.global_step,
                )

            if not args.fixed_best_checkpoint and should_attempt_promotion(
                final_best_metrics
            ):
                final_promotion_metrics = run_promotion_evaluation(
                    current_policy,
                    best_opponent,
                    evaluator,
                    confidence=args.best_checkpoint_promotion_confidence,
                    min_batches=args.best_checkpoint_promotion_min_batches,
                    max_batches=args.best_checkpoint_promotion_max_batches,
                    seed=args.seed + 60_000_000,
                )
                print(
                    "Final best checkpoint promotion check "
                    f"(epoch={trainer.epoch}, games={int(final_promotion_metrics['games'])}, "
                    f"batches={int(final_promotion_metrics['batches'])}): "
                    f"win_rate={final_promotion_metrics['win_rate']:.3f}, "
                    f"score_diff={final_promotion_metrics['score_diff']:.3f}, "
                    f"lcb95={final_promotion_metrics['win_rate_lcb']:.3f}, "
                    f"promoted={bool(final_promotion_metrics['promoted'])}"
                )
                if logger is not None:
                    logger.wandb.log(
                        {
                            "evaluation/final_best_checkpoint/promotion_attempted": 1.0,
                            "evaluation/final_best_checkpoint/promotion_batches": (
                                final_promotion_metrics["batches"]
                            ),
                            "evaluation/final_best_checkpoint/promotion_games": (
                                final_promotion_metrics["games"]
                            ),
                            "evaluation/final_best_checkpoint/promotion_games_per_batch": (
                                final_promotion_metrics["games_per_batch"]
                            ),
                            "evaluation/final_best_checkpoint/promotion_win_rate": (
                                final_promotion_metrics["win_rate"]
                            ),
                            "evaluation/final_best_checkpoint/promotion_score_diff": (
                                final_promotion_metrics["score_diff"]
                            ),
                            "evaluation/final_best_checkpoint/promotion_win_rate_lcb": (
                                final_promotion_metrics["win_rate_lcb"]
                            ),
                            "evaluation/final_best_checkpoint/promotion_confidence": (
                                final_promotion_metrics["confidence"]
                            ),
                            "evaluation/final_best_checkpoint/promotion_promoted": (
                                final_promotion_metrics["promoted"]
                            ),
                            "evaluation/progress_step": float(trainer.global_step),
                        },
                        step=trainer.global_step,
                    )
            elif logger is not None:
                logger.wandb.log(
                    {
                        "evaluation/final_best_checkpoint/promotion_attempted": 0.0,
                        "evaluation/progress_step": float(trainer.global_step),
                    },
                    step=trainer.global_step,
                )

        if evaluator is not None:
            evaluator.close()

        if (
            final_promotion_metrics is not None
            and final_promotion_metrics["promoted"] > 0.5
        ):
            bundle_metadata = build_bundle_export_metadata(
                args=args,
                run_id="local-run" if logger is None else logger.run_id,
                epoch=trainer.epoch,
                global_step=trainer.global_step,
                event="promotion",
                checkpoint_path=model_path,
                previous_best=best_record,
                train_config=train_config,
                objective_metrics=final_best_metrics,
                promotion_metrics=final_promotion_metrics,
            )
            best_record = register_best_checkpoint(
                logger=logger,
                checkpoint_path=model_path,
                best_config_path=best_config_path,
                best_history_path=best_history_path,
                previous_best=best_record,
                vec_config=vec_config if eval_vec_config is None else eval_vec_config,
                run_id="local-run" if logger is None else logger.run_id,
                epoch=trainer.epoch,
                global_step=trainer.global_step,
                event="promotion",
                promotion_metrics=final_promotion_metrics,
                policy=current_policy,
                observation_shape=tuple(vecenv.single_observation_space.shape),
                bundle_metadata=bundle_metadata,
            )
            best_state_dict = snapshot_policy_state(current_policy)
            best_opponent = current_policy
            print(
                "Promoted new best checkpoint after final evaluation: "
                f"{best_record.get('artifact_ref') or model_path}"
            )

        if best_record is None and not args.fixed_best_checkpoint:
            bundle_metadata = build_bundle_export_metadata(
                args=args,
                run_id="local-run" if logger is None else logger.run_id,
                epoch=trainer.epoch,
                global_step=trainer.global_step,
                event="bootstrap",
                checkpoint_path=model_path,
                previous_best=None,
                train_config=train_config,
                objective_metrics=final_best_metrics,
                promotion_metrics=None,
            )
            best_record = register_best_checkpoint(
                logger=logger,
                checkpoint_path=model_path,
                best_config_path=best_config_path,
                best_history_path=best_history_path,
                previous_best=None,
                vec_config=vec_config if eval_vec_config is None else eval_vec_config,
                run_id="local-run" if logger is None else logger.run_id,
                epoch=trainer.epoch,
                global_step=trainer.global_step,
                event="bootstrap",
                promotion_metrics=None,
                policy=current_policy,
                observation_shape=tuple(vecenv.single_observation_space.shape),
                bundle_metadata=bundle_metadata,
            )
            best_state_dict = snapshot_policy_state(current_policy)
            best_opponent = current_policy
            print(
                "Bootstrapped best checkpoint from final model: "
                f"{best_record.get('artifact_ref') or model_path}"
            )

    if model_path is None or effective_train_config is None or effective_vec_config is None:
        raise RuntimeError("training did not produce a final model")
    if final_trainer is None:
        raise RuntimeError("training did not produce trainer metadata")

    best_video_path = None
    if args.export_videos and best_opponent is not None and current_policy is not None:
        best_video_path = save_best_checkpoint_video(
            current_policy, best_opponent, args
        )

    if best_video_path is not None:
        log_video_artifact(
            logger,
            args.best_checkpoint_video_key,
            best_video_path,
            args.video_fps,
            final_trainer.global_step,
        )

    run_summary = _build_run_summary(
        args=args,
        trainer=final_trainer,
        train_config=effective_train_config,
        eval_interval_epochs=eval_interval_epochs,
        vec_config=effective_vec_config,
        eval_vec_config=eval_vec_config,
        best_record=best_record,
        latest_best_metrics=latest_best_metrics,
        final_best_metrics=final_best_metrics,
        latest_no_opponent_metrics=latest_no_opponent_metrics,
        final_no_opponent_metrics=final_no_opponent_metrics,
        model_path=model_path,
    )
    maybe_write_run_summary(run_summary_path, run_summary)
    if logger is not None:
        logger.close(str(model_path))


def _write_video_frames(
    frames: list[np.ndarray],
    requested_path: Path,
    fps: int,
    label: str,
    *,
    overwrite_existing: bool = False,
) -> Path | None:
    """Write captured frames to disk with shared fallback behavior for every video export.

    Training generates a few different replay artifacts, but they all need the same resilience
    policy: prefer the requested location, keep periodic run-scoped videos on stable filenames,
    and never let local path collisions abort the training job. `overwrite_existing` is used for
    rolling per-run artifacts whose latest version matters more than preserving every older local
    copy.
    """

    if not overwrite_existing and requested_path.exists():
        requested_path = _unique_path(requested_path)
        print(f"Video output already exists; using {requested_path}")
    if not frames:
        print(f"No frames captured for {label}; skipping video export.")
        return None

    out_path = _resolve_writable_output_path(requested_path)
    frame_sequence = cast(list[Any], frames)
    try:
        imageio.mimsave(out_path, frame_sequence, fps=fps, macro_block_size=None)
        print(f"Saved {label}: {out_path}")
        return out_path
    except Exception as err:
        if _is_permission_like_error(err):
            retry_path = _resolve_writable_output_path(_unique_path(out_path))
            if retry_path != out_path:
                print(f"MP4 export path blocked; retrying {label} at {retry_path}")
                try:
                    imageio.mimsave(
                        retry_path, frame_sequence, fps=fps, macro_block_size=None
                    )
                    print(f"Saved {label}: {retry_path}")
                    return retry_path
                except Exception as retry_err:
                    print(f"MP4 retry failed ({retry_err}); falling back to GIF.")

        print(f"MP4 export failed ({err}); falling back to GIF for {label}.")
        fallback = _resolve_writable_output_path(out_path.with_suffix(".gif"))
        try:
            imageio.mimsave(fallback, frame_sequence, fps=fps)
            print(f"Saved {label} fallback: {fallback}")
            return fallback
        except Exception as gif_err:
            print(f"GIF export failed ({gif_err}); skipping {label} artifact.")
            return None


def save_match_video(
    current_policy: Policy,
    args,
    *,
    output_path: Path,
    label: str,
    opponents_enabled: bool = True,
    no_opponent_field_scale: float = 1.0,
    opponent_state_dict: Mapping[str, torch.Tensor] | None = None,
    opponent_policy_runner: Any | None = None,
    overwrite_existing: bool = False,
) -> Path | None:
    """Capture one rendered policy replay against either itself, no opponents, or a snapshot.

    Training now needs one shared video exporter for three closely related cases:
    no-opponent warm-start rollouts, ordinary self-play, and current-versus-best-checkpoint
    comparisons. Keeping those in one helper ensures they all use the same rendering loop,
    frame writing, and fallback behavior instead of drifting apart over time.

    ``opponents_enabled`` controls whether the environment should spawn the red team at all.
    When an ``opponent_state_dict`` or ``opponent_policy_runner`` is supplied, the current
    policy controls blue and the supplied opponent controls red. Otherwise the current policy
    simply controls every active agent in the environment, which preserves the old self-play
    behavior. ``overwrite_existing`` is used by periodic run-scoped videos so they update one
    stable local file that can then be uploaded to W&B without fighting older runs for
    filenames.
    """

    game_length = (
        resolve_no_opponent_game_length(int(args.no_opponent_eval_max_steps))
        if not opponents_enabled
        else 400
    )
    base_env = make_puffer_env(
        players_per_team=args.players_per_team,
        action_mode="discrete",
        game_length=game_length,
        render_mode="rgb_array",
        seed=args.seed,
        opponents_enabled=opponents_enabled,
    )
    env = (
        BlueTeamNoOpponentWrapper(base_env, args.players_per_team)
        if not opponents_enabled
        else base_env
    )
    if not opponents_enabled and no_opponent_field_scale != 1.0:
        maybe_set_training_field_scale(env, no_opponent_field_scale)

    frames: list[np.ndarray] = []
    obs, _ = env.reset(seed=args.seed)
    policy_device = next(current_policy.parameters()).device
    opponent_policy = None
    if opponent_state_dict is not None:
        opponent_policy = Policy(env).to(policy_device)
        opponent_policy.load_state_dict(opponent_state_dict, strict=True)
        opponent_policy.eval()
    elif opponent_policy_runner is not None:
        opponent_policy = opponent_policy_runner
        if hasattr(opponent_policy, "eval"):
            opponent_policy.eval()

    was_training = current_policy.training
    current_policy.eval()
    with torch.no_grad():
        for _ in range(args.video_max_steps):
            frame = env.render()
            if frame is not None:
                frames.append(frame.astype(np.uint8, copy=False))

            obs_tensor = torch.from_numpy(obs).to(policy_device)
            if opponent_policy is None:
                logits, _ = forward_policy_eval(current_policy, obs_tensor)
                actions = (
                    torch.argmax(logits, dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False)
                )
            else:
                split = args.players_per_team
                current_logits, _ = forward_policy_eval(current_policy, obs_tensor[:split])
                opponent_logits, _ = forward_policy_eval(opponent_policy, obs_tensor[split:])
                current_actions = (
                    torch.argmax(current_logits, dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False)
                )
                opponent_actions = (
                    torch.argmax(opponent_logits, dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False)
                )
                actions = np.concatenate((current_actions, opponent_actions), axis=0)

            obs, _, terminations, truncations, _ = env.step(actions)
            if bool(terminations.all() or truncations.all()):
                break

        frame = env.render()
        if frame is not None:
            frames.append(frame.astype(np.uint8, copy=False))

    env.close()
    if was_training:
        current_policy.train()

    return _write_video_frames(
        frames,
        output_path,
        args.video_fps,
        label,
        overwrite_existing=overwrite_existing,
    )


def save_self_play_video(
    policy: Policy,
    args,
    *,
    overwrite_existing: bool = False,
) -> Path | None:
    """Render the current policy playing against itself for the training video artifact.

    This helper serves both periodic training videos and one-off exports from utility scripts.
    The optional overwrite flag lets the training loop keep one rolling self-play file inside the
    active run directory without changing the older retain-unique behavior used by ad hoc tools.
    """

    return save_match_video(
        policy,
        args,
        output_path=Path(str(args.video_output)),
        label="self-play video",
        opponents_enabled=True,
        overwrite_existing=overwrite_existing,
    )


def save_best_checkpoint_video(
    policy: Policy,
    best_opponent: Mapping[str, torch.Tensor] | Any,
    args,
) -> Path | None:
    """Render the current policy against the stored best checkpoint opponent.

    This comparison video is usually written once near the end of training, so it keeps the
    default retain-unique behavior rather than reusing a rolling periodic filename.
    """

    return save_match_video(
        policy,
        args,
        output_path=Path(str(args.best_checkpoint_video_output)),
        label="best-checkpoint video",
        opponents_enabled=True,
        opponent_state_dict=best_opponent
        if isinstance(best_opponent, Mapping)
        else None,
        opponent_policy_runner=None
        if isinstance(best_opponent, Mapping)
        else best_opponent,
    )


if __name__ == "__main__":
    main()
