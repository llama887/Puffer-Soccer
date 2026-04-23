"""Before/after V(s) heatmap using just 2 checkpoints: ep=200 (random-ish
prior) and ep=46000 (final). Removes the middle-checkpoint panels that
tell the same story. Formation overlays included."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from value_heatmap import (
    HALF_X,
    canonical_formation,
    draw_formation,
    eval_checkpoint,
)
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--players-per-team", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_x = np.arange(-HALF_X, HALF_X + 1.25, 2.5)
    grid_y = np.arange(-35, 35 + 1.25, 2.5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = [200, 46000]
    hms = []
    for ep in epochs:
        ckpt = args.checkpoint_dir / f"model_{ep:06d}.pt"
        print(f"eval {ckpt.name}")
        hm = eval_checkpoint(ckpt, args.players_per_team, grid_x, grid_y, device=device)
        hms.append(hm)

    xy, _, team = canonical_formation(args.players_per_team, blue_left=True)
    vmin = min(h.min() for h in hms)
    vmax = max(h.max() for h in hms)
    amp = max(abs(vmin), abs(vmax))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    titles = ["ep=200 (untrained-ish)", "ep=46000 (final)"]
    for ax, hm, title in zip(axes, hms, titles):
        im = ax.imshow(hm, extent=extent, origin="lower", cmap="RdBu_r",
                       vmin=-amp, vmax=amp, aspect="equal")
        for gx_edge, side in ((-HALF_X, "blue"), (HALF_X, "red")):
            ax.plot([gx_edge, gx_edge], [-20, 20], color=side, linewidth=3)
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
        draw_formation(ax, xy, team)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("ball x (blue attacks +x)")
    axes[0].set_ylabel("ball y")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="mean V(blue)")
    fig.suptitle(
        "Value function learned by the critic: before vs after 50k epochs of self-play\n"
        "Ball swept over the field, all 10 agents fixed in canonical formation (blue=goalie+4 defenders on left)",
        fontsize=10,
    )
    out = args.output_dir / "value_before_after.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
