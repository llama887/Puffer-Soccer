from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import torch

import pufferlib
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

    num_shards = args.vec_num_shards or physical_cpu_count()
    batch_size = args.vec_batch_size or num_shards
    shard_num_envs = args.shard_num_envs or 2
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
    eval_env = make_puffer_env(
        num_envs=args.past_iterate_eval_envs,
        players_per_team=args.players_per_team,
        game_length=args.past_iterate_eval_game_length,
        action_mode="discrete",
        seed=args.seed + epoch,
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


def main():
    try:
        import pufferlib.pufferl as pufferl
    except ImportError as err:
        raise SystemExit(
            "Failed to import pufferlib training backend. "
            "Run this script with `uv run` from the repo, or reinstall `pufferlib` in the active "
            "interpreter with `uv pip install --reinstall --no-build-isolation pufferlib==3.0.0`. "
            f"Original error: {err}"
        ) from err

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
    parser.add_argument("--shard-num-envs", type=int, default=None)
    parser.add_argument("--ppo-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--bptt-horizon", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=20_480)
    parser.add_argument("--update-epochs", type=int, default=2)
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

    trainer = pufferl.PuffeRL(cfg["train"], vecenv, policy, logger=logger)
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
        num_envs=1,
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

    out_path = Path(args.video_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        print("No frames captured; skipping video export.")
        return None

    try:
        imageio.mimsave(out_path, frames, fps=args.video_fps, macro_block_size=None)
        print(f"Saved self-play video: {out_path}")
        return out_path
    except Exception as err:
        print(f"MP4 export failed ({err}); falling back to GIF.")
        fallback = out_path.with_suffix(".gif")
        imageio.mimsave(fallback, frames, fps=args.video_fps)
        print(f"Saved self-play video fallback: {fallback}")
        return fallback


if __name__ == "__main__":
    main()
