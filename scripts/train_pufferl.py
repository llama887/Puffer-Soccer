from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import torch

import pufferlib
import pufferlib.pufferl as pufferl
import pufferlib.vector
from pufferlib.emulation import PettingZooPufferEnv

from puffer_soccer.envs.marl2d import make_parallel_env
from puffer_soccer.envs.marl2d.core import flatten_obs_dict


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", type=str, default="robot-soccer")
    parser.add_argument("--wandb-group", type=str, default="puffer-default")
    parser.add_argument("--wandb-tag", type=str, default=None)
    parser.add_argument("--wandb-video-key", type=str, default="self_play_video")
    parser.add_argument("--video-output", type=str, default="experiments/self_play.mp4")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-max-steps", type=int, default=600)
    args = parser.parse_args()
    load_env_file(".env")

    def env_creator(**kwargs):
        pz = make_parallel_env(
            players_per_team=args.players_per_team,
            action_mode="discrete",
            game_length=400,
            render_mode=None,
            seed=kwargs.get("seed", args.seed),
        )
        return PettingZooPufferEnv(env=pz)

    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=args.num_envs,
        backend=pufferlib.vector.Serial,
        seed=args.seed,
    )

    # pufferl.load_config() parses sys.argv, so strip custom script args first.
    argv = sys.argv
    sys.argv = [argv[0]]
    cfg = pufferl.load_config("default")
    sys.argv = argv
    cfg["train"]["total_timesteps"] = args.total_timesteps
    cfg["train"]["batch_size"] = vecenv.num_agents * 64
    cfg["train"]["bptt_horizon"] = 64
    cfg["train"]["minibatch_size"] = vecenv.num_agents * 16
    cfg["train"]["learning_rate"] = 3e-4
    cfg["train"]["device"] = "cpu"
    cfg["train"]["seed"] = args.seed
    cfg["train"]["env"] = "puffer_soccer_marl2d"

    policy = Policy(vecenv.driver_env)
    logger = None
    if args.wandb:
        logger_args = {
            "wandb_project": args.wandb_project,
            "wandb_group": args.wandb_group,
            "tag": args.wandb_tag,
        }
        logger = pufferl.WandbLogger(logger_args)

    trainer = pufferl.PuffeRL(cfg["train"], vecenv, policy, logger=logger)

    while trainer.epoch < trainer.total_epochs:
        trainer.evaluate()
        trainer.train()

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
    env = make_parallel_env(
        players_per_team=args.players_per_team,
        action_mode="discrete",
        game_length=400,
        render_mode="rgb_array",
        seed=args.seed,
    )

    frames = []
    obs, _ = env.reset(seed=args.seed)
    agents = env.possible_agents

    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        for _ in range(args.video_max_steps):
            frame = env.render()
            if frame is not None:
                frames.append(frame.astype(np.uint8, copy=False))

            obs_batch = np.stack([flatten_obs_dict(obs[a]) for a in agents]).astype(np.float32)
            logits, _ = policy.forward_eval(torch.from_numpy(obs_batch))
            actions = torch.argmax(logits, dim=-1).cpu().numpy()
            action_dict = {agent: int(actions[i]) for i, agent in enumerate(agents)}

            obs, _, terminations, truncations, _ = env.step(action_dict)
            if all(terminations.values()) or all(truncations.values()):
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
