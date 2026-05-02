"""Emergence-of-behaviors plots, one PNG per metric.

Reads stats_epoch_*.json produced by scripts/teamplay_trace.py, computes a
small set of conditional metrics on top of them, and writes one PNG per
metric into --output-dir. No smoothing; raw per-checkpoint value only.

Conditional metrics
-------------------
"mean # defenders back (given >=1 back while attacking)" comes from the
unconditional E[N] in the stats JSON. Since N=0 contributes 0 to E[N],
E[N | N>=1] = E[N] / P(N>=1) = mean_count / frac. Symmetric for the
attackers-forward-while-defending metric.

Usage
-----
    python scripts/plot_emergence_individual.py \
        --input-dir experiments/teamplay_trace/<run>/stats \
        --output-dir experiments/autoloop/plots/emergence
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EPOCH_RE = re.compile(r"stats_epoch_(\d+)\.json$")


def _load(input_dir: Path) -> tuple[np.ndarray, list[dict]]:
    files = sorted(input_dir.glob("stats_epoch_*.json"))
    if not files:
        raise SystemExit(f"No stats JSONs in {input_dir}")
    rows = []
    for p in files:
        m = EPOCH_RE.search(p.name)
        if not m:
            continue
        with open(p) as f:
            d = json.load(f)
        d["_epoch"] = int(m.group(1))
        rows.append(d)
    rows.sort(key=lambda r: r["_epoch"])
    epochs = np.array([r["_epoch"] for r in rows])
    return epochs, rows


def _series(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r.get(key, np.nan) for r in rows], dtype=float)


def _cond_mean(rows: list[dict], mean_key: str, frac_key: str) -> np.ndarray:
    """E[N | N>=1] = E[N] / P(N>=1). Uses nan where frac==0."""
    m = _series(rows, mean_key)
    f = _series(rows, frac_key)
    out = np.full_like(m, np.nan, dtype=float)
    ok = f > 1e-9
    out[ok] = m[ok] / f[ok]
    return out


def _plot_one(
    epochs: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(epochs, y, color="tab:blue", linewidth=1.4, marker="o", markersize=2.6)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("training epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="skip plots whose key is absent from the JSONs (useful while goalie_rotations is still being produced)",
    )
    args = parser.parse_args()

    epochs, rows = _load(args.input_dir)

    plans: list[tuple[str, np.ndarray, str, str]] = [
        (
            "n_passes",
            _series(rows, "n_passes"),
            "# passes / game",
            "passes",
        ),
        (
            "n_dribbles",
            _series(rows, "n_dribbles"),
            "# dribbles / game",
            "dribbles",
        ),
        (
            "num_touches",
            _series(rows, "num_touches"),
            "ball touches / game",
            "touches",
        ),
        (
            "goalie_frac",
            _series(rows, "goalie_frac"),
            "% time a goalie is on the baseline",
            "fraction",
        ),
        (
            "off_while_def_frac",
            _series(rows, "off_while_def_frac"),
            "% time team leaves a striker forward while defending",
            "fraction",
        ),
        (
            "def_while_off_frac",
            _series(rows, "def_while_off_frac"),
            "% time team keeps a defender back while attacking",
            "fraction",
        ),
        (
            "cond_n_forwards_while_defending",
            _cond_mean(rows, "off_while_def_mean_count", "off_while_def_frac"),
            "mean # forwards forward | >=1 forward while defending",
            "players",
        ),
        (
            "cond_n_defenders_while_attacking",
            _cond_mean(rows, "def_while_off_mean_count", "def_while_off_frac"),
            "mean # defenders back | >=1 defender back while attacking",
            "players",
        ),
        (
            "goalie_rotations",
            _series(rows, "goalie_rotations"),
            "goalie rotations / game (handoff within 20 steps)",
            "rotations",
        ),
        (
            "ball_x_entropy",
            _series(rows, "ball_x_entropy"),
            "ball x-position entropy (higher = fuller-field play)",
            "nats",
        ),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, y, title, ylabel in plans:
        if np.all(~np.isfinite(y)):
            if args.skip_missing:
                print(f"skip {name}: not in stats")
                continue
            print(f"warn: {name} absent from stats; writing empty axes")
        out = args.output_dir / f"{name}.png"
        _plot_one(epochs, y, title=title, ylabel=ylabel, out_path=out)


if __name__ == "__main__":
    main()
