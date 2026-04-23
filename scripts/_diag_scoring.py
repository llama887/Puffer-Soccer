"""Quick diagnostic: does argmax self-play score goals in a scalar env?"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
spec = importlib.util.spec_from_file_location("train_pufferl_d", REPO / "scripts" / "train_pufferl.py")
train = importlib.util.module_from_spec(spec); sys.modules[spec.name] = train
spec.loader.exec_module(train)

from puffer_soccer.envs.marl2d import make_puffer_env


def main():
    ckpt = Path("/scratch/fyy2003/repos/Puffer-Soccer-replication/sbatch-tmp/6856525/Puffer-Soccer-replication/experiments/61xajhha/model_049520.pt")
    env = make_puffer_env(players_per_team=5, action_mode="discrete", game_length=400, render_mode=None, seed=17)
    state = train.load_checkpoint_state_dict(ckpt)
    if "state_dict" in state: state = state["state_dict"]
    policy = train.Policy(env).cpu()
    policy.load_state_dict(state, strict=True); policy.eval()
    obs, _ = env.reset()
    goals_history = []
    with torch.no_grad():
        for ep in range(20):
            obs, _ = env.reset(seed=17 + ep)
            for t in range(400):
                obs_t = torch.from_numpy(obs)
                logits, _ = policy(obs_t)
                acts = torch.argmax(logits, dim=-1).numpy().astype(np.int32)
                obs, _r, te, tr, _i = env.step(acts)
                st = env.get_state(0)
                gb, gr = st["goals"]
                if np.any(te) or np.any(tr):
                    break
            st = env.get_state(0)
            gb, gr = st["goals"]
            goals_history.append((gb, gr, t + 1))
            print(f"ep={ep}: final goals blue={gb} red={gr}, steps={t + 1}", flush=True)
    n_blue = sum(1 for b, r, _ in goals_history if b > 0 and r == 0)
    n_red = sum(1 for b, r, _ in goals_history if r > 0 and b == 0)
    print(f"\nsingle-blue eps: {n_blue}, single-red eps: {n_red}")


if __name__ == "__main__":
    main()
