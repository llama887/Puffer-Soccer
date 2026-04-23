"""Action-mix sweep across repl's checkpoints: what fraction of actions are
NOOP / MOVE / ROT / KICK over training?

For each selected checkpoint, run a few native-vec self-play games and count
per-action frequencies across all agents and steps. Produces a line plot of
the 4 categorical action shares vs training epoch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "train_pufferl_axn", REPO / "scripts" / "train_pufferl.py"
)
_train = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _train
_spec.loader.exec_module(_train)

from puffer_soccer.envs.marl2d import make_native_vec_env

KICK_ACTION_MIN = 5
KICK_ACTION_MAX = 12
MODEL_NAME_RE = re.compile(r"model_(\d+)\.pt$")


def run_checkpoint(ckpt_path: Path, ppt: int, num_envs: int, steps: int, device: str, seed: int) -> dict[str, float]:
    env = make_native_vec_env(num_envs=num_envs, players_per_team=ppt,
                              action_mode="discrete", game_length=400,
                              render_mode=None, log_interval=1, seed=seed)
    state = _train.load_checkpoint_state_dict(ckpt_path)
    if "state_dict" in state and "format_version" in state:
        state = state["state_dict"]
    policy = _train.Policy(env).to(device)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    obs, _ = env.reset(seed=seed)
    total = np.zeros(13, dtype=np.int64)
    with torch.no_grad():
        for _ in range(steps):
            obs_t = torch.from_numpy(obs).to(device)
            logits, _ = policy(obs_t)
            acts = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int32)
            np.add.at(total, acts, 1)
            obs, _r, _te, _tr, _i = env.step(acts)
    env.close()
    n = int(total.sum())
    frac = total.astype(np.float64) / max(1, n)
    return {
        "epoch": int(MODEL_NAME_RE.search(ckpt_path.name).group(1)),
        "noop": float(frac[0]),
        "move": float(frac[1:3].sum()),
        "rot": float(frac[3:5].sum()),
        "kick": float(frac[KICK_ACTION_MIN : KICK_ACTION_MAX + 1].sum()),
        "total_actions": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(p for p in args.checkpoint_dir.glob("model_*.pt") if MODEL_NAME_RE.search(p.name))
    ckpts = ckpts[:: args.stride]
    print(f"{len(ckpts)} checkpoints")
    rows = []
    for i, c in enumerate(ckpts):
        print(f"[{i + 1}/{len(ckpts)}] {c.name}", flush=True)
        r = run_checkpoint(c, args.players_per_team, args.num_envs, args.steps,
                            args.device, args.seed + int(MODEL_NAME_RE.search(c.name).group(1)))
        rows.append(r)

    with open(args.output_dir / "action_mix.json", "w") as f:
        json.dump(rows, f, indent=2)

    epochs = np.array([r["epoch"] for r in rows])
    noop = np.array([r["noop"] for r in rows])
    move = np.array([r["move"] for r in rows])
    rot = np.array([r["rot"] for r in rows])
    kick = np.array([r["kick"] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(epochs, noop, move, rot, kick,
                  labels=["NOOP", "MOVE (F/B)", "ROT (L/R)", "KICK (8 buckets)"],
                  colors=["#888", "#ff7f0e", "#2ca02c", "#d62728"], alpha=0.85)
    ax.set_xlim(epochs.min(), epochs.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("fraction of argmax actions")
    ax.set_title("Action-category mix over training (argmax greedy rollouts)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = args.output_dir / "action_mix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
