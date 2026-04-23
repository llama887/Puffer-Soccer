"""Plot teamplay stats across training (output of teamplay_trace.py).

Reads every stats_epoch_*.json in --input-dir, sorts by epoch, and emits one
PNG per metric plus a single grid overview.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EPOCH_RE = re.compile(r"stats_epoch_(\d+)\.json$")

METRICS = [
    ("n_passes", "# passes per game", "passes (game average)"),
    ("mean_pass_length", "mean pass length", "distance units"),
    ("n_double_passes", "# double-passes per game", "double passes"),
    ("n_triple_passes", "# triple-passes per game", "triple passes"),
    ("n_dribbles", "# dribbles per game", "dribbles"),
    ("goalie_frac", "% of time with a goalie", "fraction"),
    ("off_while_def_frac", "% time: forward left while defending", "fraction"),
    ("off_while_def_mean_count", "mean forwards while defending", "count"),
    ("def_while_off_frac", "% time: defender back while attacking", "fraction"),
    ("def_while_off_mean_count", "mean defenders while attacking", "count"),
    ("velocity_toward_ball", "mean velocity toward ball (ball chasing)", "units/step"),
    ("formation_compactness", "formation std-dev (lower = tighter)", "std units"),
    ("mean_forward_press_x", "mean forward press x (higher = pressing up)", "x units"),
    ("ball_x_entropy", "ball x-position entropy (higher = full-field)", "nats"),
    ("num_touches", "inferred touches per game", "count"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--warmstart-end-epoch",
        type=int,
        default=0,
        help="Draw a vertical line at this epoch to mark end of warmstart "
        "phase (0 = no line; use for runs where warmstart was loaded from "
        "cache rather than trained in-run).",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("stats_epoch_*.json"))
    if not files:
        raise SystemExit(f"No stats JSONs in {args.input_dir}")

    rows = []
    for p in files:
        m = EPOCH_RE.search(p.name)
        if not m:
            continue
        with open(p) as f:
            data = json.load(f)
        data["_epoch"] = int(m.group(1))
        rows.append(data)
    rows.sort(key=lambda r: r["_epoch"])
    epochs = np.array([r["_epoch"] for r in rows])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    warm_end = int(args.warmstart_end_epoch)
    # single-metric plots
    for key, title, ylabel in METRICS:
        if key not in rows[0]:
            continue
        y = np.array([r[key] for r in rows])
        y_std = np.array([r.get(f"{key}_std", 0.0) for r in rows])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, y, marker="o", markersize=3, linewidth=1.2)
        ax.fill_between(epochs, y - y_std, y + y_std, alpha=0.15)
        if warm_end > 0:
            ax.axvline(warm_end, color="red", linestyle="--", linewidth=1, alpha=0.7, label="warmstart end")
            ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("training epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = args.output_dir / f"{key}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")

    # grid overview
    n = sum(1 for k, _, _ in METRICS if k in rows[0])
    cols = 3
    r = (n + cols - 1) // cols
    fig, axes = plt.subplots(r, cols, figsize=(cols * 5, r * 3))
    axes = np.array(axes).reshape(-1)
    i = 0
    for key, title, _ in METRICS:
        if key not in rows[0]:
            continue
        y = np.array([r[key] for r in rows])
        ax = axes[i]
        ax.plot(epochs, y, marker="o", markersize=2, linewidth=1)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.3)
        i += 1
    for j in range(i, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    out = args.output_dir / "teamplay_overview.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
