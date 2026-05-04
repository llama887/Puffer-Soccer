"""Kick-strength → ball trajectory plot for the thesis Settings section.

For each of the 8 discrete kick strengths, simulate the ball alone (no
players) under the env's kick-impulse + 5.0 speed clip + 0.85 decay.
Constants are pulled from src/puffer_soccer/envs/marl2d/csrc/binding.c.

The figure has two panels:
- left: ball x-position over time for each kick strength (top-down sim
  with the ball starting at x=−40, kick along +x)
- right: total distance traveled vs. kick strength (bar chart)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LEG_SPEED = 4.0          # ball_check_hit() in binding.c
BALL_DECAY = 0.85
MAX_BALL_SPEED = 5.0
KICK_SCALES = np.array([
    0.1, 0.22857143, 0.35714287, 0.4857143,
    0.6142857, 0.74285716, 0.87142855, 1.0,
], dtype=np.float32)
N_STEPS = 60
START_X = -40.0


def simulate_kick(kick_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, vs) over N_STEPS for a one-shot kick along +x at t=0."""
    x = np.zeros(N_STEPS, dtype=np.float32)
    v = np.zeros(N_STEPS, dtype=np.float32)
    x[0] = START_X
    # Kick impulse applied at t=0, ball was at rest.
    v_after_kick = LEG_SPEED * KICK_SCALES[kick_idx]
    v_after_kick = min(v_after_kick, MAX_BALL_SPEED)
    v[0] = v_after_kick
    for t in range(1, N_STEPS):
        # b_{t+1} = b_t + clip(v_t)
        v_clip = min(v[t - 1], MAX_BALL_SPEED)
        x[t] = x[t - 1] + v_clip
        # u_{t+1} = 0.85 * clip(v_t)
        v[t] = BALL_DECAY * v_clip
    return x, v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_t, ax_d) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2.4, 1.0]},
    )

    cmap = plt.cm.viridis
    distances = np.zeros(len(KICK_SCALES), dtype=np.float32)
    for k in range(len(KICK_SCALES)):
        xs, _ = simulate_kick(k)
        distances[k] = float(xs[-1] - xs[0])
        ax_t.plot(xs, color=cmap(k / max(len(KICK_SCALES) - 1, 1)),
                  linewidth=1.8, label=f"kick {k + 1}")

    ax_t.axhline(50.0, color="black", linestyle=":", linewidth=0.8,
                 alpha=0.6, label="goal line (x=+50)")
    ax_t.set_xlabel("simulator step", fontsize=11)
    ax_t.set_ylabel("ball x", fontsize=11)
    ax_t.set_title("Ball position after a single kick", fontsize=13)
    ax_t.legend(fontsize=8, ncol=2, loc="lower right")
    ax_t.grid(True, alpha=0.3)

    ax_d.bar(np.arange(1, len(KICK_SCALES) + 1), distances,
             color=cmap(np.linspace(0, 1, len(KICK_SCALES))),
             edgecolor="black", linewidth=0.5)
    ax_d.set_xlabel("kick strength index", fontsize=11)
    ax_d.set_ylabel("total distance traveled", fontsize=11)
    ax_d.set_title("Distance per kick strength", fontsize=13)
    ax_d.set_xticks(np.arange(1, len(KICK_SCALES) + 1))
    ax_d.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = args.output_dir / "kick_strengths.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    np.savez(args.output_dir / "kick_strengths_data.npz",
             kick_scales=KICK_SCALES, distances=distances,
             leg_speed=LEG_SPEED, ball_decay=BALL_DECAY,
             max_ball_speed=MAX_BALL_SPEED, n_steps=N_STEPS)


if __name__ == "__main__":
    main()
