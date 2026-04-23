"""Wrap repl_pure's final model (model_049520.pt) in the cached-warm-start
format so the continuation run can load it via --cached-warm-start-path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> None:
    src = Path("/scratch/fyy2003/repos/Puffer-Soccer-replication/sbatch-tmp/6856525/Puffer-Soccer-replication/experiments/61xajhha/model_049520.pt")
    dst = Path("/scratch/fyy2003/repos/Puffer-Soccer-replication/experiments/cached_warm_start_repl_final.pt")
    state = torch.load(src, map_location="cpu")
    if not isinstance(state, dict):
        print("state is not a dict, aborting")
        sys.exit(1)
    cache = {
        "state_dict": state,
        "epoch": 49520,
        "global_step": 49520 * 400,  # approximate
        "players_per_team": 5,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, dst)
    print(f"wrote {dst} ({sum(v.numel() for v in state.values())} params)")


if __name__ == "__main__":
    main()
