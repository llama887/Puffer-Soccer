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
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
import torch

import pufferlib
import pufferlib.pufferl as pufferl
import pufferlib.pytorch

from puffer_soccer.autotune import (
    BenchmarkResult,
    autotune_vecenv,
    format_benchmark_result,
    vec_config_from_benchmark,
)
from puffer_soccer.envs.marl2d import make_puffer_env
from puffer_soccer.vector_env import (
    VecEnvConfig,
    make_soccer_vecenv,
    physical_cpu_count,
    total_sim_envs,
)


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
        opponent_state: Mapping[str, torch.Tensor],
        num_games: int,
        seed: int,
    ) -> tuple[list[float], list[float]]:
        """Play a fixed number of games and return per-game win scores and score diffs.

        Returning raw per-game outcomes keeps the helper flexible. The regular training
        plots can average the results directly, while the promotion loop can keep batching
        them until its confidence target is met.
        """

        self.opponent_policy.load_state_dict(opponent_state, strict=True)
        obs, _ = self.eval_env.reset(seed=seed)
        self.eval_env.flush_log()

        was_training_current = current_policy.training
        current_policy.eval()
        self.opponent_policy.eval()

        completed_games = 0
        score_diffs: list[float] = []
        win_scores: list[float] = []
        with torch.no_grad():
            while completed_games < num_games:
                obs_tensor = torch.as_tensor(
                    obs, device=self.device, dtype=torch.float32
                )

                current_logits, _ = current_policy.forward_eval(
                    obs_tensor.index_select(0, self.current_indices)
                )
                opponent_logits, _ = self.opponent_policy.forward_eval(
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
        opponent_state: Mapping[str, torch.Tensor],
        num_games: int,
        seed: int,
    ) -> dict[str, float]:
        """Return mean head-to-head metrics for a fixed number of evaluation games."""

        win_scores, score_diffs = self.run_games(
            current_policy, opponent_state, num_games=num_games, seed=seed
        )
        return summarize_match_results(win_scores, score_diffs)

    def close(self) -> None:
        """Release the reusable evaluation environment."""

        self.eval_env.close()


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
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
    batch_size: int, horizon: int, requested_size: int
) -> int:
    """Project a requested minibatch size onto the nearest valid PuffeRL value.

    PuffeRL requires minibatches to be divisible by the rollout horizon, and this
    project works best when minibatches also divide the batch cleanly so each update
    sees the full batch instead of silently truncating the tail. Hyperparameter search
    therefore cannot use arbitrary integers directly.

    This helper converts a requested size into the nearest legal value while keeping the
    search space easy to describe in user-facing terms. It is intentionally shared by the
    training CLI and the RL sweep driver so both paths obey the same validation rules.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if batch_size % horizon != 0:
        raise ValueError("batch_size must be divisible by horizon")

    clamped = min(max(requested_size, horizon), batch_size)
    candidates = [
        candidate
        for candidate in range(horizon, batch_size + 1, horizon)
        if batch_size % candidate == 0
    ]
    if not candidates:
        raise ValueError(
            "expected at least one minibatch candidate divisible by both batch and horizon"
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
) -> tuple[int, int]:
    """Resolve batch and minibatch sizes for manual runs and hyperparameter sweeps.

    The default training script previously derived these values entirely from the active
    vector layout, which made rollout-size tuning impossible. For RL tuning we now need
    two behaviors at once: preserve the old defaults when no overrides are supplied, and
    safely coerce user- or sweep-provided sizes onto values that PuffeRL will accept.

    The returned batch size is always large enough to hold at least one full horizon for
    every live agent. The returned minibatch size is always a legal divisor-compatible
    multiple of the horizon. That keeps the training loop stable while still allowing the
    sweep driver to search over meaningful rollout-scale decisions.
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


def build_train_config(args, vecenv, device: str) -> dict[str, Any]:
    """Assemble the concrete PuffeRL train config for one run.

    The training script now serves two use cases: direct training from the CLI and many
    repeated sweep trials launched by the tuning driver. Both need one canonical place that
    applies user overrides, preserves the existing defaults when no override is supplied,
    and records the exact effective values that will be used for learning.
    """

    config = load_base_train_config()
    batch_size, minibatch_size = resolve_requested_train_sizes(
        vecenv.num_agents,
        horizon=args.bptt_horizon,
        requested_batch_size=args.train_batch_size,
        requested_minibatch_size=args.minibatch_size,
    )
    total_timesteps = resolve_total_timesteps(
        args.total_timesteps,
        args.ppo_iterations,
        batch_size,
    )
    config["batch_size"] = batch_size
    config["bptt_horizon"] = args.bptt_horizon
    config["minibatch_size"] = minibatch_size
    config["total_timesteps"] = total_timesteps
    config["learning_rate"] = args.learning_rate
    config["update_epochs"] = args.update_epochs
    config["ent_coef"] = args.ent_coef
    config["gamma"] = args.gamma
    config["gae_lambda"] = args.gae_lambda
    config["clip_coef"] = args.clip_coef
    config["vf_coef"] = args.vf_coef
    config["vf_clip_coef"] = args.vf_clip_coef
    config["max_grad_norm"] = args.max_grad_norm
    config["prio_alpha"] = args.prio_alpha
    config["prio_beta0"] = args.prio_beta0
    config["device"] = device
    config["seed"] = args.seed
    config["env"] = "puffer_soccer_marl2d"
    if args.checkpoint_interval is not None:
        config["checkpoint_interval"] = args.checkpoint_interval
    return config


def summarize_effective_hyperparameters(
    train_config: Mapping[str, Any], args
) -> dict[str, int | float | bool | str | None]:
    """Capture the learning settings that define one finished training run.

    Hyperparameter tuning only helps if we can later explain which exact values won. This
    summary intentionally records the effective values after all coercion and defaulting,
    not just the raw CLI arguments. That makes the generated summary files reliable inputs
    for follow-up confirmation runs and for manual inspection after a long sweep.
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
    }


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
    """Benchmark vector layouts before training and return the fastest saturated choice.

    The goal of the auto backend is to maximize environment throughput on the current
    machine while pushing CPU usage close to full utilization. The autotuner explores the
    supported vector backends, measures steps per second for each candidate, and prefers
    candidates that reach the near-100% CPU target before falling back to raw throughput.

    This helper exists so that training can support `--vec-backend auto` directly instead
    of requiring the user to run a separate benchmarking script first. It returns both the
    runtime `VecEnvConfig` and the benchmark record that justified the choice so the caller
    can print or log the decision in a human-readable way.
    """
    outcome = autotune_vecenv(
        players_per_team=args.players_per_team,
        seconds=args.autotune_seconds,
        action_mode="discrete",
        backend="auto",
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
    best_state_dict: Mapping[str, torch.Tensor],
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
            best_state_dict,
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


def current_timestamp() -> str:
    """Return a simple UTC timestamp string for metadata files."""

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
) -> dict[str, object]:
    """Persist and optionally upload a newly established best checkpoint."""

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
    write_json_record(best_config_path, record)
    append_jsonl_record(best_history_path, record)
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
    """Tell W&B to chart all evaluation metrics against a shared progress step.

    The previous configuration still referenced an old metric namespace. Defining a single
    evaluation step metric here keeps both the past-iterate plots and the best-checkpoint plots
    aligned on the same x-axis.
    """

    if logger is None or not hasattr(logger, "wandb"):
        return
    logger.wandb.define_metric("evaluation/progress_step")
    logger.wandb.define_metric("evaluation/*", step_metric="evaluation/progress_step")


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


def _unique_path(path: Path, max_attempts: int = 100) -> Path:
    if not path.exists():
        return path

    for idx in range(1, max_attempts + 1):
        candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available filename for {path}")


def _resolve_writable_output_path(path: Path) -> Path:
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


def print_match_summary(label: str, epoch: int, metrics: Mapping[str, float]) -> None:
    """Print one compact evaluation summary line for the active training epoch."""

    print(
        f"{label} (epoch={epoch}, games={int(metrics['games'])}): "
        f"win_rate={metrics['win_rate']:.3f}, score_diff={metrics['score_diff']:.3f}"
    )


def log_video_artifact(
    logger, video_key: str, video_path: Path, fps: int, step: int
) -> None:
    """Upload one generated video to W&B if logging is enabled."""

    if logger is None or not hasattr(logger, "wandb"):
        return
    video_format = "gif" if video_path.suffix.lower() == ".gif" else "mp4"
    logger.wandb.log(
        {
            video_key: logger.wandb.Video(
                str(video_path),
                fps=fps,
                format=video_format,
            )
        },
        step=step,
    )


def _build_run_summary(
    *,
    args,
    trainer,
    train_config: Mapping[str, Any],
    vec_config: VecEnvConfig,
    eval_vec_config: VecEnvConfig | None,
    best_record: Mapping[str, object] | None,
    latest_best_metrics: Mapping[str, float] | None,
    final_best_metrics: Mapping[str, float] | None,
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
        "objective_metrics": objective_metrics,
        "latest_best_checkpoint_metrics": None
        if latest_best_metrics is None
        else dict(latest_best_metrics),
        "final_best_checkpoint_metrics": None
        if final_best_metrics is None
        else dict(final_best_metrics),
        "vec_config": serialize_vec_config(vec_config),
        "eval_vec_config": None
        if eval_vec_config is None
        else serialize_vec_config(eval_vec_config),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=8)
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
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--prio-alpha", type=float, default=0.8)
    parser.add_argument("--prio-beta0", type=float, default=0.2)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--no-regularization", action="store_true")
    parser.add_argument("--past-kl-coef", type=float, default=0.1)
    parser.add_argument("--uniform-kl-base-coef", type=float, default=0.05)
    parser.add_argument("--uniform-kl-power", type=float, default=0.3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", type=str, default="robot-soccer")
    parser.add_argument("--wandb-group", type=str, default="puffer-default")
    parser.add_argument("--wandb-tag", type=str, default=None)
    parser.add_argument("--wandb-video-key", type=str, default="self_play_video")
    parser.add_argument("--video-output", type=str, default="experiments/self_play.mp4")
    parser.add_argument(
        "--best-checkpoint-video-key",
        type=str,
        default="best_checkpoint_video",
    )
    parser.add_argument(
        "--best-checkpoint-video-output",
        type=str,
        default="experiments/best_checkpoint.mp4",
    )
    parser.add_argument(
        "--export-videos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-max-steps", type=int, default=600)
    parser.add_argument(
        "--past-iterate-eval", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--past-iterate-eval-fractions", type=int, default=100)
    parser.add_argument("--past-iterate-eval-envs", type=int, default=None)
    parser.add_argument("--past-iterate-eval-games", type=int, default=64)
    parser.add_argument("--past-iterate-eval-game-length", type=int, default=400)
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
    args = parser.parse_args()
    load_env_file(".env")

    device = resolve_device(args.device)
    best_config_path = Path(args.best_checkpoint_config_path)
    best_history_path = Path(args.best_checkpoint_history_path)
    best_cache_dir = best_config_path.parent / "wandb_artifacts"
    run_summary_path = (
        None if args.run_summary_path is None else Path(args.run_summary_path)
    )

    vec_config, autotune_result = resolve_training_vec_config(args)
    vecenv = make_soccer_vecenv(
        players_per_team=args.players_per_team,
        action_mode="discrete",
        game_length=400,
        render_mode=None,
        seed=args.seed,
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

    train_config = build_train_config(args, vecenv, device)
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

    policy = Policy(vecenv).to(device)
    logger = None
    if args.wandb:
        logger_args = {
            "wandb_project": args.wandb_project,
            "wandb_group": args.wandb_group,
            "tag": args.wandb_tag,
        }
        logger = pufferl.WandbLogger(logger_args)
        _configure_iterate_metrics(logger)

    trainer = RegularizedPuffeRL(
        train_config,
        vecenv,
        policy,
        logger=logger,
        regularization_enabled=not args.no_regularization,
        past_kl_coef=args.past_kl_coef,
        uniform_kl_base_coef=args.uniform_kl_base_coef,
        uniform_kl_power=args.uniform_kl_power,
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

    best_record = read_json_record(best_config_path)
    best_state_dict: dict[str, torch.Tensor] | None = None
    latest_best_metrics: dict[str, float] | None = None
    final_best_metrics: dict[str, float] | None = None
    if best_record is not None:
        try:
            best_state_dict, best_checkpoint_path = load_best_checkpoint_state(
                best_record,
                logger=logger,
                cache_dir=best_cache_dir,
            )
            if best_record.get("cached_checkpoint_path") != str(best_checkpoint_path):
                best_record = dict(best_record)
                best_record["cached_checkpoint_path"] = str(best_checkpoint_path)
                write_json_record(best_config_path, best_record)
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
            best_state_dict = None

    if args.fixed_best_checkpoint and best_state_dict is None:
        raise RuntimeError(
            "fixed-best-checkpoint mode requires a readable best checkpoint record"
        )

    while trainer.epoch < trainer.total_epochs:
        previous_state_dict = clone_state_dict(policy)
        trainer.evaluate()
        trainer.train()
        should_eval = (
            args.past_iterate_eval
            and trainer.epoch > 0
            and (
                trainer.epoch % eval_interval_epochs == 0
                or trainer.epoch == trainer.total_epochs
            )
        )
        if should_eval and evaluator is not None:
            eval_seed = args.seed + trainer.epoch * 10_000
            eval_metrics = evaluate_against_past_iterate(
                policy,
                previous_state_dict,
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
                "evaluation/past_iterate/baseline_epoch": trainer.epoch - 1,
                "evaluation/past_iterate/current_epoch": trainer.epoch,
            }
            print_match_summary("Past iterate eval", trainer.epoch, eval_metrics)

            if best_state_dict is not None:
                best_metrics = evaluator.evaluate(
                    policy,
                    best_state_dict,
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

                if (
                    logger is not None
                    and not args.fixed_best_checkpoint
                    and should_attempt_promotion(best_metrics)
                ):
                    promotion_metrics = run_promotion_evaluation(
                        policy,
                        best_state_dict,
                        evaluator,
                        confidence=args.best_checkpoint_promotion_confidence,
                        min_batches=args.best_checkpoint_promotion_min_batches,
                        max_batches=args.best_checkpoint_promotion_max_batches,
                        seed=eval_seed + 2_000,
                    )
                    log_payload.update(
                        {
                            "evaluation/best_checkpoint/promotion_attempted": 1.0,
                            "evaluation/best_checkpoint/promotion_batches": promotion_metrics[
                                "batches"
                            ],
                            "evaluation/best_checkpoint/promotion_games": promotion_metrics[
                                "games"
                            ],
                            "evaluation/best_checkpoint/promotion_games_per_batch": promotion_metrics[
                                "games_per_batch"
                            ],
                            "evaluation/best_checkpoint/promotion_win_rate": promotion_metrics[
                                "win_rate"
                            ],
                            "evaluation/best_checkpoint/promotion_score_diff": promotion_metrics[
                                "score_diff"
                            ],
                            "evaluation/best_checkpoint/promotion_win_rate_lcb": promotion_metrics[
                                "win_rate_lcb"
                            ],
                            "evaluation/best_checkpoint/promotion_confidence": promotion_metrics[
                                "confidence"
                            ],
                            "evaluation/best_checkpoint/promotion_promoted": promotion_metrics[
                                "promoted"
                            ],
                        }
                    )
                    print(
                        "Best checkpoint promotion check "
                        f"(epoch={trainer.epoch}, games={int(promotion_metrics['games'])}, "
                        f"batches={int(promotion_metrics['batches'])}): "
                        f"win_rate={promotion_metrics['win_rate']:.3f}, "
                        f"score_diff={promotion_metrics['score_diff']:.3f}, "
                        f"lcb95={promotion_metrics['win_rate_lcb']:.3f}, "
                        f"promoted={bool(promotion_metrics['promoted'])}"
                    )
                    if promotion_metrics["promoted"] > 0.5:
                        assert eval_vec_config is not None
                        checkpoint_str = trainer.save_checkpoint()
                        if checkpoint_str is None:
                            raise RuntimeError(
                                "trainer.save_checkpoint() returned no path"
                            )
                        checkpoint_path = Path(checkpoint_str)
                        best_record = register_best_checkpoint(
                            logger=logger,
                            checkpoint_path=checkpoint_path,
                            best_config_path=best_config_path,
                            best_history_path=best_history_path,
                            previous_best=best_record,
                            vec_config=eval_vec_config,
                            run_id=logger.run_id,
                            epoch=trainer.epoch,
                            global_step=trainer.global_step,
                            event="promotion",
                            promotion_metrics=promotion_metrics,
                        )
                        best_state_dict = snapshot_policy_state(policy)
                        print(
                            "Promoted new best checkpoint: "
                            f"{best_record.get('artifact_ref') or checkpoint_path}"
                        )
                elif logger is not None:
                    log_payload["evaluation/best_checkpoint/promotion_attempted"] = 0.0

            if logger is not None:
                logger.wandb.log(log_payload, step=trainer.global_step)

    trainer.print_dashboard()
    model_path = Path(trainer.close())

    if best_state_dict is not None and args.final_best_eval_games > 0:
        if evaluator is None:
            raise RuntimeError(
                "final best-checkpoint eval requested without an evaluator"
            )
        final_best_metrics = evaluator.evaluate(
            policy,
            best_state_dict,
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

    if evaluator is not None:
        evaluator.close()

    if best_record is None and logger is not None and not args.fixed_best_checkpoint:
        best_record = register_best_checkpoint(
            logger=logger,
            checkpoint_path=model_path,
            best_config_path=best_config_path,
            best_history_path=best_history_path,
            previous_best=None,
            vec_config=vec_config if eval_vec_config is None else eval_vec_config,
            run_id=logger.run_id,
            epoch=trainer.epoch,
            global_step=trainer.global_step,
            event="bootstrap",
            promotion_metrics=None,
        )
        best_state_dict = snapshot_policy_state(policy)
        print(
            "Bootstrapped best checkpoint from final model: "
            f"{best_record.get('artifact_ref') or model_path}"
        )

    self_play_video_path = None
    best_video_path = None
    if args.export_videos:
        self_play_video_path = save_self_play_video(policy, args)
    if args.export_videos and best_state_dict is not None:
        best_video_path = save_best_checkpoint_video(policy, best_state_dict, args)

    if self_play_video_path is not None:
        log_video_artifact(
            logger,
            args.wandb_video_key,
            self_play_video_path,
            args.video_fps,
            trainer.global_step,
        )
    if best_video_path is not None:
        log_video_artifact(
            logger,
            args.best_checkpoint_video_key,
            best_video_path,
            args.video_fps,
            trainer.global_step,
        )

    run_summary = _build_run_summary(
        args=args,
        trainer=trainer,
        train_config=train_config,
        vec_config=vec_config,
        eval_vec_config=eval_vec_config,
        best_record=best_record,
        latest_best_metrics=latest_best_metrics,
        final_best_metrics=final_best_metrics,
        model_path=model_path,
    )
    maybe_write_run_summary(run_summary_path, run_summary)
    if logger is not None:
        logger.close(str(model_path))


def _write_video_frames(
    frames: list[np.ndarray], requested_path: Path, fps: int, label: str
) -> Path | None:
    """Write captured frames to disk with the same fallback logic for every video type."""

    if requested_path.exists():
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
    opponent_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> Path | None:
    """Capture one rendered match between the current policy and an optional opponent.

    The existing training video was already true self-play, because one policy controlled all
    agents in the environment. This helper keeps that path working while also allowing the
    current policy to play a stored best-checkpoint opponent in a second video artifact.
    """

    env = make_puffer_env(
        players_per_team=args.players_per_team,
        action_mode="discrete",
        game_length=400,
        render_mode="rgb_array",
        seed=args.seed,
    )

    frames: list[np.ndarray] = []
    obs, _ = env.reset(seed=args.seed)
    policy_device = next(current_policy.parameters()).device
    opponent_policy = None
    if opponent_state_dict is not None:
        opponent_policy = Policy(env).to(policy_device)
        opponent_policy.load_state_dict(opponent_state_dict, strict=True)
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
                logits, _ = current_policy.forward_eval(obs_tensor)
                actions = (
                    torch.argmax(logits, dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False)
                )
            else:
                split = args.players_per_team
                current_logits, _ = current_policy.forward_eval(obs_tensor[:split])
                opponent_logits, _ = opponent_policy.forward_eval(obs_tensor[split:])
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

    return _write_video_frames(frames, output_path, args.video_fps, label)


def save_self_play_video(policy: Policy, args) -> Path | None:
    """Render the current policy playing against itself for the training video artifact."""

    return save_match_video(
        policy,
        args,
        output_path=Path(str(args.video_output)),
        label="self-play video",
    )


def save_best_checkpoint_video(
    policy: Policy,
    best_state_dict: Mapping[str, torch.Tensor],
    args,
) -> Path | None:
    """Render the current policy against the stored best checkpoint opponent."""

    return save_match_video(
        policy,
        args,
        output_path=Path(str(args.best_checkpoint_video_output)),
        label="best-checkpoint video",
        opponent_state_dict=best_state_dict,
    )


if __name__ == "__main__":
    main()
