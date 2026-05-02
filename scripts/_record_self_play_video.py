"""Render a self-play video from a raw training checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from puffer_soccer.envs.marl2d import make_puffer_env


def _load_train_module():
    script_path = Path(__file__).resolve().with_name("train_pufferl.py")
    spec = importlib.util.spec_from_file_location("train_pufferl_video", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-max-steps", type=int, default=400)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Render blue policy vs scripted max-kick red (warm-start conditions).",
    )
    parser.add_argument(
        "--warm-start-formation",
        action="store_true",
        help=(
            "When rendering in warm-start mode, place red in the regular self-play "
            "formation instead of the corners-placement bootstrap layout."
        ),
    )
    cli = parser.parse_args()

    train = _load_train_module()

    if cli.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cli.device

    env = make_puffer_env(
        players_per_team=cli.players_per_team,
        action_mode="discrete",
        game_length=400,
        render_mode="rgb_array",
        seed=cli.seed,
    )
    state = train.load_checkpoint_state_dict(cli.checkpoint)
    if "state_dict" in state and "format_version" in state:
        state = state["state_dict"]
    # Auto-detect LSTM checkpoints so we can wrap the base policy with
    # pufferlib's LSTMWrapper. LSTMWrapper keys are prefixed with "lstm.",
    # "cell." or "policy." depending on where the parameter lives.
    has_lstm = any(
        k.startswith("lstm.") or k.startswith("cell.") or k.startswith("policy.")
        for k in state.keys()
    )
    train._USE_LSTM = has_lstm
    policy = train.build_policy(env).to(device)
    policy.load_state_dict(state, strict=True)
    policy.eval()

    args_ns = SimpleNamespace(
        players_per_team=cli.players_per_team,
        seed=cli.seed,
        video_output=str(cli.output),
        video_fps=cli.video_fps,
        video_max_steps=cli.video_max_steps,
        no_opponent_eval_max_steps=600,
        best_checkpoint_video_output=str(cli.output),
    )

    cli.output.parent.mkdir(parents=True, exist_ok=True)
    if cli.warm_start:
        path = train.save_match_video(
            policy,
            args_ns,
            output_path=cli.output,
            label="warm-start video",
            warm_start=True,
            warm_start_red_in_formation=bool(cli.warm_start_formation),
            overwrite_existing=True,
        )
    else:
        path = train.save_self_play_video(policy, args_ns, overwrite_existing=True)
    if path is None:
        raise SystemExit("Video export failed.")
    print(f"Video written: {path}")


if __name__ == "__main__":
    main()
