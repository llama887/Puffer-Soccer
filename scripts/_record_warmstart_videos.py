"""Render two demo videos for the cached warm-start policy.

1. Warm-start mode (blue vs scripted max-kick red): pick a seed that scores.
2. Self-play (current policy controls both teams): just render one seed.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from puffer_soccer.envs.marl2d import make_puffer_env
from puffer_soccer.envs.marl2d.core import MARL2DPufferEnv  # noqa: F401


def _load_train_module():
    script_path = Path(__file__).resolve().with_name("train_pufferl.py")
    spec = importlib.util.spec_from_file_location("train_pufferl_video", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_policy(train, env, state_path: Path, device: str):
    state = train.load_checkpoint_state_dict(state_path)
    if "state_dict" in state and "format_version" in state:
        state = state["state_dict"]
    has_lstm = any(
        k.startswith("lstm.") or k.startswith("cell.") or k.startswith("policy.")
        for k in state.keys()
    )
    train._USE_LSTM = has_lstm
    policy = train.build_policy(env).to(device)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy


def _score_warmstart_seed(train, checkpoint: Path, seed: int, device: str,
                          players_per_team: int, max_steps: int) -> tuple[int, int]:
    """Run warm-start rollout silently and return (blue_goals, red_goals)."""
    base_env = make_puffer_env(
        players_per_team=players_per_team,
        action_mode="discrete",
        game_length=max_steps,
        render_mode="rgb_array",
        seed=seed,
        warm_start_reward_shaping=True,
    )
    env = train.BlueTeamNoOpponentWrapper(base_env, players_per_team)
    policy = _build_policy(train, env, checkpoint, device)

    obs, _ = env.reset(seed=seed)
    policy_device = next(policy.parameters()).device
    with torch.no_grad():
        for _ in range(max_steps):
            obs_tensor = torch.from_numpy(obs).to(policy_device)
            logits, _ = train.forward_policy_eval(policy, obs_tensor, None)
            actions = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int32)
            obs, _, term, trunc, _ = env.step(actions)
            if bool(term.all() or trunc.all()):
                break

    state = env.env.get_state(0) if hasattr(env, "env") else env.get_state(0)
    goals_blue, goals_red = state["goals"]
    env.close()
    return int(goals_blue), int(goals_red)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--warm-start-max-steps", type=int, default=600)
    parser.add_argument("--self-play-seed", type=int, default=0)
    parser.add_argument("--warmstart-seed-search-max", type=int, default=40)
    cli = parser.parse_args()

    train = _load_train_module()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cli.device == "auto" else cli.device
    cli.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching for a warm-start seed that scores (max {cli.warmstart_seed_search_max})...")
    chosen_seed = None
    for seed in range(cli.warmstart_seed_search_max):
        gb, gr = _score_warmstart_seed(
            train, cli.checkpoint, seed, device,
            cli.players_per_team, cli.warm_start_max_steps,
        )
        print(f"  seed={seed}: blue={gb} red={gr}")
        if gb > 0:
            chosen_seed = seed
            break
    if chosen_seed is None:
        raise SystemExit("No warm-start seed produced a blue goal in the search budget.")
    print(f"Picked warm-start seed {chosen_seed}.")

    warm_env = make_puffer_env(
        players_per_team=cli.players_per_team,
        action_mode="discrete",
        game_length=cli.warm_start_max_steps,
        render_mode="rgb_array",
        seed=chosen_seed,
        warm_start_reward_shaping=True,
    )
    warm_env = train.BlueTeamNoOpponentWrapper(warm_env, cli.players_per_team)
    warm_policy = _build_policy(train, warm_env, cli.checkpoint, device)

    warm_args = SimpleNamespace(
        players_per_team=cli.players_per_team,
        seed=chosen_seed,
        video_output=str(cli.out_dir / "warmstart_vs_scripted_scoring.mp4"),
        video_fps=cli.video_fps,
        video_max_steps=cli.warm_start_max_steps,
        no_opponent_eval_max_steps=cli.warm_start_max_steps,
        best_checkpoint_video_output=str(cli.out_dir / "warmstart_vs_scripted_scoring.mp4"),
    )
    warm_path = train.save_match_video(
        warm_policy,
        warm_args,
        output_path=cli.out_dir / "warmstart_vs_scripted_scoring.mp4",
        label="warmstart vs scripted (scoring)",
        warm_start=True,
        overwrite_existing=True,
    )
    print(f"Warm-start video: {warm_path}")

    sp_env = make_puffer_env(
        players_per_team=cli.players_per_team,
        action_mode="discrete",
        game_length=400,
        render_mode="rgb_array",
        seed=cli.self_play_seed,
    )
    sp_policy = _build_policy(train, sp_env, cli.checkpoint, device)
    sp_args = SimpleNamespace(
        players_per_team=cli.players_per_team,
        seed=cli.self_play_seed,
        video_output=str(cli.out_dir / "warmstart_self_play.mp4"),
        video_fps=cli.video_fps,
        video_max_steps=400,
        no_opponent_eval_max_steps=400,
        best_checkpoint_video_output=str(cli.out_dir / "warmstart_self_play.mp4"),
    )
    sp_path = train.save_self_play_video(sp_policy, sp_args, overwrite_existing=True)
    print(f"Self-play video: {sp_path}")


if __name__ == "__main__":
    main()
