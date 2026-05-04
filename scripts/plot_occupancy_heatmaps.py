"""Court occupancy heatmaps averaged over many games.

Reads one or more trace .npz files produced by scripts/teamplay_trace.py
(each containing (num_envs, T, num_players, 2) positions and
(num_envs, T, 4) ball state) and writes heatmaps of:

  * blue-team player occupancy (all 5 blue slots combined)
  * red-team player occupancy  (all 5 red slots combined)
  * both teams combined
  * ball position

Blue attacks +x and red attacks -x in every env we recorded (blue_left=True),
so heatmaps orient naturally.

Usage
-----
    python scripts/plot_occupancy_heatmaps.py \
        --traces experiments/teamplay_trace/.../traces/trace_epoch_049200.npz \
        --output-dir experiments/autoloop/plots/occupancy

Pass multiple --traces to stack across checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIELD_HALF_X = 50.0
FIELD_HALF_Y = 35.0
# 2-unit cells -> 50 x 35 bins
X_BINS = 50
Y_BINS = 35


def _heatmap(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, _, _ = np.histogram2d(
        xs, ys,
        bins=[X_BINS, Y_BINS],
        range=[[-FIELD_HALF_X, FIELD_HALF_X], [-FIELD_HALF_Y, FIELD_HALF_Y]],
    )
    return h  # shape (X_BINS, Y_BINS)


def _plot(
    heat: np.ndarray,
    title: str,
    out_path: Path,
    *,
    cmap: str = "inferno",
    log_scale: bool = False,
    n_samples: int | None = None,
) -> None:
    # heat shape (X_BINS, Y_BINS). imshow expects (Y, X) with origin at top-left,
    # so transpose and flip y so +y is "up" in image.
    img = heat.T[::-1, :]
    if log_scale:
        img = np.log1p(img)

    fig, ax = plt.subplots(figsize=(8, 5.7))
    im = ax.imshow(
        img,
        extent=[-FIELD_HALF_X, FIELD_HALF_X, -FIELD_HALF_Y, FIELD_HALF_Y],
        aspect="equal",
        cmap=cmap,
        interpolation="bilinear",
    )
    # Field boundary + midline + goal lines
    ax.axvline(0, color="white", alpha=0.55, linewidth=1)
    ax.add_patch(plt.Rectangle(
        (-FIELD_HALF_X, -FIELD_HALF_Y), 2 * FIELD_HALF_X, 2 * FIELD_HALF_Y,
        fill=False, edgecolor="white", linewidth=1, alpha=0.7,
    ))
    # Goals are at x = +/- FIELD_HALF_X, y in +/-20
    for sign in (-1, +1):
        ax.plot(
            [sign * FIELD_HALF_X, sign * FIELD_HALF_X],
            [-20, 20], color="white", alpha=0.8, linewidth=2,
        )

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("log(1 + count)" if log_scale else "count")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="+", required=True, type=Path,
                        help="one or more trace_epoch_*.npz files")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scales", default="linear,log",
                        help="comma list of {linear,log}; writes one file per scale with suffix")
    parser.add_argument("--label", default=None,
                        help="optional label used in filenames/titles (default: epoch of first trace)")
    args = parser.parse_args()

    all_blue_pos = []  # list of (K, 2)
    all_red_pos = []
    all_ball_pos = []
    epochs = []
    for p in args.traces:
        d = np.load(p)
        epochs.append(int(d["epoch"]))
        ppt = int(d["players_per_team"])
        positions = d["positions"]  # (N, T, 2*ppt, 2)
        ball = d["ball"]             # (N, T, 4)
        blue_left = d["blue_left"]   # (N,)
        if not np.all(blue_left):
            # flip any env where blue_left is False so blue is always left-attacking-right.
            mask = ~blue_left
            # Flip x for those envs' positions + ball.
            positions[mask, ..., 0] *= -1
            ball[mask, ..., 0] *= -1
            ball[mask, ..., 2] *= -1
            # Also swap team indices so blue indices always correspond to the blue-left team.
            blue_slots = positions[mask, :, :ppt].copy()
            red_slots = positions[mask, :, ppt:].copy()
            positions[mask, :, :ppt] = red_slots
            positions[mask, :, ppt:] = blue_slots
        blue = positions[:, :, :ppt].reshape(-1, 2)
        red = positions[:, :, ppt:].reshape(-1, 2)
        all_blue_pos.append(blue)
        all_red_pos.append(red)
        all_ball_pos.append(ball[..., :2].reshape(-1, 2))

    blue_all = np.concatenate(all_blue_pos, axis=0)
    red_all = np.concatenate(all_red_pos, axis=0)
    ball_all = np.concatenate(all_ball_pos, axis=0)

    label = args.label or (f"ep{epochs[0]:06d}" if len(epochs) == 1 else f"ep{min(epochs)}-{max(epochs)}")

    blue_heat = _heatmap(blue_all[:, 0], blue_all[:, 1])
    red_heat = _heatmap(red_all[:, 0], red_all[:, 1])
    all_heat = blue_heat + red_heat
    ball_heat = _heatmap(ball_all[:, 0], ball_all[:, 1])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    for scale in scales:
        if scale not in ("linear", "log"):
            raise SystemExit(f"unknown scale {scale!r}; use linear or log")
        log_scale = scale == "log"
        _plot(
            blue_heat,
            "Blue-team occupancy",
            args.output_dir / f"occupancy_blue_{label}_{scale}.png",
            cmap="Blues",
            log_scale=log_scale,
        )
        _plot(
            red_heat,
            "Red-team occupancy",
            args.output_dir / f"occupancy_red_{label}_{scale}.png",
            cmap="Reds",
            log_scale=log_scale,
        )
        _plot(
            all_heat,
            "All-player occupancy",
            args.output_dir / f"occupancy_all_{label}_{scale}.png",
            cmap="inferno",
            log_scale=log_scale,
        )
        _plot(
            ball_heat,
            "Ball occupancy",
            args.output_dir / f"occupancy_ball_{label}_{scale}.png",
            cmap="inferno",
            log_scale=log_scale,
        )


if __name__ == "__main__":
    main()
