"""Curated teamplay-emergence plot: only the metrics that show clear signal.

Drops noisy ones (double/triple-pass, mean-pass-length, forward-press-x,
mode-action noise) and focuses on the 9 that actually tell the story.
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
    ("n_passes", "# passes / game", "count"),
    ("n_dribbles", "# dribbles / game", "count"),
    ("num_touches", "ball touches / game", "count"),
    ("goalie_frac", "% time a goalie is on the baseline", "fraction"),
    ("off_while_def_frac", "% time team leaves striker forward while defending", "fraction"),
    ("def_while_off_frac", "% time team keeps defender back while attacking", "fraction"),
    ("velocity_toward_ball", "mean velocity toward ball (ball-chasing)", "units/step"),
    ("formation_compactness", "team formation std-dev (smaller = tighter)", "std"),
    ("ball_x_entropy", "ball x-position entropy (higher = full-field play)", "nats"),
]


def smooth(y: np.ndarray, w: int = 5) -> np.ndarray:
    if len(y) <= w:
        return y
    kernel = np.ones(w) / w
    pad = w // 2
    ypad = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(ypad, kernel, mode="valid")[: len(y)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("stats_epoch_*.json"))
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

    # 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    axes_flat = axes.ravel()
    for ax, (key, title, ylabel) in zip(axes_flat, METRICS):
        if key not in rows[0]:
            ax.axis("off")
            continue
        y = np.array([r[key] for r in rows])
        ax.plot(epochs, y, color="tab:blue", alpha=0.35, linewidth=0.9, label="raw")
        y_smooth = smooth(y, w=5)
        ax.plot(epochs, y_smooth, color="tab:blue", linewidth=2.0, label="smoothed")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("training epoch")
    fig.suptitle(
        "Emergent soccer behaviors during self-play training (repl_pure, 50k epochs)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.output_dir / "emergence_overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
