from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from collections.abc import Mapping
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import imageio.v2 as imageio
import numpy as np
import torch

import pufferlib
import pufferlib.pufferl as pufferl
import pufferlib.pytorch

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


class IterateComparator:
    def __init__(
        self,
        *,
        players_per_team: int,
        game_length: int,
        eval_envs: int,
        eval_episodes: int,
        device: str,
    ):
        self.eval_episodes = eval_episodes
        self.device = device
        self.eval_env = make_soccer_vecenv(
            players_per_team=players_per_team,
            vec=VecEnvConfig(backend="native", shard_num_envs=eval_envs, num_shards=1),
            action_mode="discrete",
            game_length=game_length,
            do_team_switch=True,
            render_mode=None,
            seed=0,
            log_interval=1,
        )
        self.previous_policy = Policy(self.eval_env).to(device)
        self.num_players = players_per_team * 2
        base = np.arange(eval_envs, dtype=np.int64)[:, None] * self.num_players
        blue_offsets = np.arange(players_per_team, dtype=np.int64)[None, :]
        red_offsets = (np.arange(players_per_team, dtype=np.int64) + players_per_team)[
            None, :
        ]
        self.blue_idx = (base + blue_offsets).reshape(-1)
        self.red_idx = (base + red_offsets).reshape(-1)
        self.blue_idx_t = torch.as_tensor(
            self.blue_idx, dtype=torch.long, device=device
        )
        self.red_idx_t = torch.as_tensor(self.red_idx, dtype=torch.long, device=device)
        self.action_buf = np.zeros((self.eval_env.num_agents,), dtype=np.int32)

    def evaluate(
        self,
        current_policy: torch.nn.Module,
        previous_state: Mapping[str, torch.Tensor],
        seed: int,
    ) -> dict[str, float]:
        self.previous_policy.load_state_dict(previous_state, strict=True)
        obs, _ = self.eval_env.reset(seed=seed)
        self.eval_env.flush_log()

        was_training_current = current_policy.training
        current_policy.eval()
        self.previous_policy.eval()

        total_episodes = 0.0
        total_wins = 0.0
        total_score_sum = 0.0
        with torch.no_grad():
            while total_episodes < self.eval_episodes:
                obs_tensor = torch.as_tensor(
                    obs, device=self.device, dtype=torch.float32
                )

                blue_obs = obs_tensor.index_select(0, self.blue_idx_t)
                red_obs = obs_tensor.index_select(0, self.red_idx_t)

                blue_logits, _ = current_policy.forward_eval(blue_obs)
                red_logits, _ = self.previous_policy.forward_eval(red_obs)

                blue_actions = (
                    torch.argmax(blue_logits, dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False)
                )
                red_actions = (
                    torch.argmax(red_logits, dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.int32, copy=False)
                )

                self.action_buf[self.blue_idx] = blue_actions
                self.action_buf[self.red_idx] = red_actions

                obs, _, _, _, _ = self.eval_env.step(self.action_buf)
                log = self.eval_env.flush_log()
                if log is None:
                    continue

                n = float(log.get("n", 0.0))
                if n <= 0:
                    continue
                total_episodes += n
                total_wins += float(log.get("wins_blue", 0.0))
                total_score_sum += float(log.get("score_diff", 0.0)) * n

        if was_training_current:
            current_policy.train()

        if total_episodes <= 0:
            return {"win_rate": 0.0, "score_diff": 0.0, "episodes": 0.0}

        return {
            "win_rate": total_wins / total_episodes,
            "score_diff": total_score_sum / total_episodes,
            "episodes": total_episodes,
        }

    def close(self) -> None:
        self.eval_env.close()


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def clone_state_dict(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in policy.state_dict().items()
    }


def compute_eval_interval_epochs(total_epochs: int, fractions: int) -> int:
    return max(1, total_epochs // max(1, fractions))


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def compute_train_sizes(total_agents: int, horizon: int = 64) -> tuple[int, int]:
    if total_agents < 1:
        raise ValueError("total_agents must be positive")

    batch_size = total_agents * horizon
    # PuffeRL requires minibatches to divide evenly into the BPTT horizon.
    minibatch_size = round_up_to_multiple(max(total_agents * 16, horizon), horizon)
    minibatch_size = min(minibatch_size, batch_size)
    return batch_size, minibatch_size


def resolve_vec_config(args) -> VecEnvConfig:
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


def make_side_assignment(num_envs: int) -> np.ndarray:
    current_on_blue = np.zeros((num_envs,), dtype=bool)
    current_on_blue[: (num_envs + 1) // 2] = True
    return current_on_blue


def score_metrics_from_perspective(
    goals_blue: int, goals_red: int, current_on_blue: bool
) -> tuple[float, float]:
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


def evaluate_against_past_iterate(
    current_policy: torch.nn.Module,
    previous_state_dict: dict[str, torch.Tensor],
    args,
    epoch: int,
) -> dict[str, float]:
    eval_env = make_soccer_vecenv(
        players_per_team=args.players_per_team,
        game_length=args.past_iterate_eval_game_length,
        action_mode="discrete",
        seed=args.seed + epoch,
        render_mode=None,
        log_interval=1,
        vec=VecEnvConfig(
            backend="native",
            shard_num_envs=args.past_iterate_eval_envs,
            num_shards=1,
        ),
    )

    device = next(current_policy.parameters()).device
    previous_policy = Policy(eval_env)
    previous_policy.load_state_dict(previous_state_dict)
    previous_policy.to(device)

    num_envs = args.past_iterate_eval_envs
    num_players = eval_env.num_players
    players_per_team = args.players_per_team
    total_agents = eval_env.num_agents
    current_on_blue = make_side_assignment(num_envs)
    current_agent_mask = np.zeros((total_agents,), dtype=bool)

    for env_idx in range(num_envs):
        start = env_idx * num_players
        split = start + players_per_team
        end = start + num_players
        if current_on_blue[env_idx]:
            current_agent_mask[start:split] = True
        else:
            current_agent_mask[split:end] = True

    current_indices = torch.as_tensor(
        np.flatnonzero(current_agent_mask), dtype=torch.long, device=device
    )
    previous_indices = torch.as_tensor(
        np.flatnonzero(~current_agent_mask), dtype=torch.long, device=device
    )

    completed_games = 0
    score_diffs: list[float] = []
    win_rates: list[float] = []
    was_training = current_policy.training
    current_policy.eval()
    previous_policy.eval()
    obs, _ = eval_env.reset(seed=args.seed + epoch)

    with torch.no_grad():
        while completed_games < args.past_iterate_eval_games:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            actions = torch.zeros((total_agents,), dtype=torch.int64, device=device)

            current_logits, _ = current_policy.forward_eval(obs_tensor[current_indices])
            previous_logits, _ = previous_policy.forward_eval(
                obs_tensor[previous_indices]
            )
            actions[current_indices] = torch.argmax(current_logits, dim=-1)
            actions[previous_indices] = torch.argmax(previous_logits, dim=-1)

            _, _, terminals, truncations, _ = eval_env.step(
                actions.cpu().numpy().astype(np.int32, copy=False)
            )
            done_envs = np.flatnonzero(
                terminals.reshape(num_envs, num_players).all(axis=1)
                | truncations.reshape(num_envs, num_players).all(axis=1)
            )

            for env_idx in done_envs:
                if completed_games >= args.past_iterate_eval_games:
                    break
                final_goals = eval_env.get_last_episode_scores(int(env_idx))
                if final_goals is None:
                    continue
                score_diff, win_rate = score_metrics_from_perspective(
                    final_goals[0], final_goals[1], bool(current_on_blue[env_idx])
                )
                score_diffs.append(score_diff)
                win_rates.append(win_rate)
                completed_games += 1

            obs = eval_env.observations

    eval_env.close()
    if was_training:
        current_policy.train()

    return {
        "win_rate": float(np.mean(win_rates)) if win_rates else 0.0,
        "score_diff": float(np.mean(score_diffs)) if score_diffs else 0.0,
        "games": float(completed_games),
    }


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return name


def snapshot_policy_state(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.state_dict().items()
    }


def _configure_iterate_metrics(logger) -> None:
    if logger is None or not hasattr(logger, "wandb"):
        return
    logger.wandb.define_metric("iterate_vs_prev/progress_step")
    logger.wandb.define_metric(
        "iterate_vs_prev/*", step_metric="iterate_vs_prev/progress_step"
    )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--vec-backend",
        type=str,
        default="native",
        choices=["native", "serial", "multiprocessing"],
    )
    parser.add_argument("--vec-num-shards", type=int, default=None)
    parser.add_argument("--vec-batch-size", type=int, default=None)
    parser.add_argument("--ppo-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--bptt-horizon", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=20_480)
    parser.add_argument("--update-epochs", type=int, default=2)
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
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-max-steps", type=int, default=600)
    parser.add_argument(
        "--past-iterate-eval", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--past-iterate-eval-fractions", type=int, default=10)
    parser.add_argument("--past-iterate-eval-envs", type=int, default=16)
    parser.add_argument("--past-iterate-eval-games", type=int, default=64)
    parser.add_argument("--past-iterate-eval-game-length", type=int, default=400)
    args = parser.parse_args()
    load_env_file(".env")

    device = resolve_device(args.device)

    vec_config = resolve_vec_config(args)
    vecenv = make_soccer_vecenv(
        players_per_team=args.players_per_team,
        action_mode="discrete",
        game_length=400,
        render_mode=None,
        seed=args.seed,
        vec=vec_config,
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

    # pufferl.load_config() parses sys.argv, so strip custom script args first.
    argv = sys.argv
    sys.argv = [argv[0]]
    cfg = pufferl.load_config("default")
    sys.argv = argv
    batch_size, minibatch_size = compute_train_sizes(vecenv.num_agents)
    cfg["train"]["batch_size"] = batch_size
    cfg["train"]["bptt_horizon"] = 64
    cfg["train"]["minibatch_size"] = minibatch_size
    cfg["train"]["total_timesteps"] = args.ppo_iterations * cfg["train"]["batch_size"]
    cfg["train"]["learning_rate"] = 3e-4
    cfg["train"]["update_epochs"] = args.update_epochs
    cfg["train"]["ent_coef"] = 0.0
    cfg["train"]["device"] = device
    cfg["train"]["seed"] = args.seed
    cfg["train"]["env"] = "puffer_soccer_marl2d"

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
        cfg["train"],
        vecenv,
        policy,
        logger=logger,
        regularization_enabled=not args.no_regularization,
        past_kl_coef=args.past_kl_coef,
        uniform_kl_base_coef=args.uniform_kl_base_coef,
        uniform_kl_power=args.uniform_kl_power,
    )
    eval_interval_epochs = compute_eval_interval_epochs(
        trainer.total_epochs, args.past_iterate_eval_fractions
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
        if should_eval:
            eval_metrics = evaluate_against_past_iterate(
                policy, previous_state_dict, args, trainer.epoch
            )
            log_payload = {
                "evaluation/past_iterate/win_rate": eval_metrics["win_rate"],
                "evaluation/past_iterate/score_diff": eval_metrics["score_diff"],
                "evaluation/past_iterate/games": eval_metrics["games"],
                "evaluation/past_iterate/eval_epochs_interval": eval_interval_epochs,
                "evaluation/past_iterate/baseline_epoch": trainer.epoch - 1,
                "evaluation/past_iterate/current_epoch": trainer.epoch,
            }
            if logger is not None:
                logger.wandb.log(log_payload, step=trainer.global_step)
            else:
                print(
                    "Past iterate eval "
                    f"(epoch={trainer.epoch}, games={int(eval_metrics['games'])}): "
                    f"win_rate={eval_metrics['win_rate']:.3f}, "
                    f"score_diff={eval_metrics['score_diff']:.3f}"
                )

    trainer.print_dashboard()
    model_path = trainer.close()
    video_path = save_self_play_video(policy, args)

    if logger is not None and video_path is not None:
        video_format = "gif" if video_path.suffix.lower() == ".gif" else "mp4"
        logger.wandb.log(
            {
                args.wandb_video_key: logger.wandb.Video(
                    str(video_path),
                    fps=args.video_fps,
                    format=video_format,
                )
            },
            step=trainer.global_step,
        )
    if logger is not None:
        logger.close(model_path)


def save_self_play_video(policy: torch.nn.Module, args):
    env = make_puffer_env(
        players_per_team=args.players_per_team,
        action_mode="discrete",
        game_length=400,
        render_mode="rgb_array",
        seed=args.seed,
    )

    frames = []
    obs, _ = env.reset(seed=args.seed)
    policy_device = next(policy.parameters()).device

    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        for _ in range(args.video_max_steps):
            frame = env.render()
            if frame is not None:
                frames.append(frame.astype(np.uint8, copy=False))

            obs_tensor = torch.from_numpy(obs).to(policy_device)
            logits, _ = policy.forward_eval(obs_tensor)
            actions = (
                torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int32, copy=False)
            )

            obs, _, terminations, truncations, _ = env.step(actions)
            if bool(terminations.all() or truncations.all()):
                break

        frame = env.render()
        if frame is not None:
            frames.append(frame.astype(np.uint8, copy=False))

    env.close()
    if was_training:
        policy.train()

    requested_path = Path(args.video_output)
    if requested_path.exists():
        requested_path = _unique_path(requested_path)
        print(f"Video output already exists; using {requested_path}")
    if not frames:
        print("No frames captured; skipping video export.")
        return None

    out_path = _resolve_writable_output_path(requested_path)
    try:
        imageio.mimsave(out_path, frames, fps=args.video_fps, macro_block_size=None)
        print(f"Saved self-play video: {out_path}")
        return out_path
    except Exception as err:
        if _is_permission_like_error(err):
            retry_path = _resolve_writable_output_path(_unique_path(out_path))
            if retry_path != out_path:
                print(f"MP4 export path blocked; retrying at {retry_path}")
                try:
                    imageio.mimsave(
                        retry_path, frames, fps=args.video_fps, macro_block_size=None
                    )
                    print(f"Saved self-play video: {retry_path}")
                    return retry_path
                except Exception as retry_err:
                    print(f"MP4 retry failed ({retry_err}); falling back to GIF.")

        print(f"MP4 export failed ({err}); falling back to GIF.")
        fallback = _resolve_writable_output_path(out_path.with_suffix(".gif"))
        try:
            imageio.mimsave(fallback, frames, fps=args.video_fps)
            print(f"Saved self-play video fallback: {fallback}")
            return fallback
        except Exception as gif_err:
            print(f"GIF export failed ({gif_err}); skipping video artifact.")
            return None


if __name__ == "__main__":
    main()
