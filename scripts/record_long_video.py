"""Render a long self-play video at the gallant (1f266f4) env."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "train_pufferl_vid", REPO / "scripts" / "train_pufferl.py"
)
_train = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _train
_spec.loader.exec_module(_train)

from puffer_soccer.envs.marl2d import make_puffer_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--game-length", type=int, default=400)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    env = make_puffer_env(
        players_per_team=args.players_per_team,
        action_mode="discrete",
        game_length=args.game_length,
        render_mode="rgb_array",
        seed=args.seed,
    )

    state = _train.load_checkpoint_state_dict(args.checkpoint)
    if "state_dict" in state and "format_version" in state:
        state = state["state_dict"]
    policy = _train.Policy(env).to(args.device)
    policy.load_state_dict(state, strict=True)
    policy.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    obs, _ = env.reset(seed=args.seed)
    ep = 0
    blue_goals = red_goals = 0
    with torch.no_grad():
        for t in range(args.total_steps):
            obs_t = torch.from_numpy(obs).to(args.device)
            logits, _vals = policy(obs_t)
            acts = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int32)
            obs, _rew, term, trunc, _info = env.step(acts)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            if bool(np.any(term)) or bool(np.any(trunc)):
                ep += 1
                st = env.get_state(0)
                bg, rg = st.get("goals", (0, 0))
                blue_goals = bg
                red_goals = rg

    print(f"frames captured: {len(frames)}")
    print(f"final goals: blue={blue_goals} red={red_goals} episodes={ep}")
    imageio.mimsave(str(args.output), frames, fps=args.fps, codec="libx264")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
