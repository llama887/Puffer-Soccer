"""Annotated field schematic for the thesis Settings section (3.1).

Renders the 100x70 playing field with the goal regions (|y| <= 20) at the
left and right end lines, plus one example reset of the 5v5 environment
showing the random starting positions of all 10 players + the ball.
Constants come straight from src/puffer_soccer/envs/marl2d/csrc/binding.c
so the figure is guaranteed to match the env at this commit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from puffer_soccer.envs.marl2d import make_native_vec_env

FIELD_HALF_X = 50.0
FIELD_HALF_Y = 35.0
GOAL_HALF_Y = 20.0
AGENT_RADIUS = 1.0
BALL_RADIUS = 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--players-per-team", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = make_native_vec_env(
        num_envs=1,
        players_per_team=args.players_per_team,
        action_mode="discrete",
        seed=args.seed,
    )
    env.reset()
    state = env.get_state(0)
    pos = np.asarray(state["positions"], dtype=np.float32)         # (2n, 2)
    rot = np.asarray(state["rotations"], dtype=np.float32)         # (2n,)
    ball = np.asarray(state["ball"], dtype=np.float32)             # (4,) x,y,vx,vy
    n = args.players_per_team
    teams = np.array([0] * n + [1] * n, dtype=np.int32)            # 0=blue, 1=red
    env.close()

    fig, ax = plt.subplots(figsize=(10, 7))

    # Field outline
    ax.add_patch(mpatches.Rectangle(
        (-FIELD_HALF_X, -FIELD_HALF_Y), 2 * FIELD_HALF_X, 2 * FIELD_HALF_Y,
        fill=False, edgecolor="black", linewidth=1.5,
    ))
    # Midline
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    # Goals (vertical strips at |x|=50, y in [-20, 20])
    for sign, color in ((-1, "tab:blue"), (+1, "tab:red")):
        ax.add_patch(mpatches.Rectangle(
            (sign * FIELD_HALF_X - 0.5, -GOAL_HALF_Y),
            1.0, 2 * GOAL_HALF_Y,
            facecolor=color, edgecolor=color, linewidth=2, alpha=0.6,
        ))
        ax.text(sign * (FIELD_HALF_X - 6), GOAL_HALF_Y + 2.5,
                f"goal\n|y| ≤ {GOAL_HALF_Y:g}",
                ha="center", va="bottom", fontsize=9, color=color)

    # Players
    for i, (xy, theta, t) in enumerate(zip(pos, rot, teams)):
        color = "tab:blue" if t == 0 else "tab:red"
        ax.add_patch(mpatches.Circle(xy, AGENT_RADIUS, color=color,
                                      ec="black", linewidth=0.5, zorder=4))
        # heading arrow
        ax.arrow(xy[0], xy[1],
                 2.5 * np.cos(theta), 2.5 * np.sin(theta),
                 head_width=0.9, head_length=1.0, fc="black", ec="black",
                 length_includes_head=True, zorder=5)

    # Ball
    ax.add_patch(mpatches.Circle((ball[0], ball[1]), BALL_RADIUS,
                                  color="white", ec="black",
                                  linewidth=1.0, zorder=6))

    # Axis annotations
    ax.set_xlim(-FIELD_HALF_X - 8, FIELD_HALF_X + 8)
    ax.set_ylim(-FIELD_HALF_Y - 8, FIELD_HALF_Y + 8)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.set_title("PufferSoccer field — 5v5 example reset", fontsize=13)

    # Dimension annotations
    ax.annotate("", xy=(-FIELD_HALF_X, -FIELD_HALF_Y - 4),
                xytext=(FIELD_HALF_X, -FIELD_HALF_Y - 4),
                arrowprops=dict(arrowstyle="<->", color="gray"))
    ax.text(0, -FIELD_HALF_Y - 6, f"{2 * FIELD_HALF_X:g} units",
            ha="center", va="top", fontsize=9, color="gray")
    ax.annotate("", xy=(FIELD_HALF_X + 4, -FIELD_HALF_Y),
                xytext=(FIELD_HALF_X + 4, FIELD_HALF_Y),
                arrowprops=dict(arrowstyle="<->", color="gray"))
    ax.text(FIELD_HALF_X + 5.5, 0, f"{2 * FIELD_HALF_Y:g} units",
            ha="left", va="center", fontsize=9, color="gray", rotation=90)

    # Legend
    legend_handles = [
        mpatches.Patch(color="tab:blue", label="blue team (attacks +x)"),
        mpatches.Patch(color="tab:red", label="red team (attacks −x)"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="ball"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False, fontsize=9)

    fig.tight_layout()
    out = args.output_dir / "field_schematic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # Also save the data so the figure is reproducible without env access
    np.savez(args.output_dir / "field_schematic_state.npz",
             positions=np.asarray(pos, dtype=np.float32),
             rotations=np.asarray(rot, dtype=np.float32),
             teams=np.asarray(teams, dtype=np.int32),
             ball=np.asarray(ball, dtype=np.float32))


if __name__ == "__main__":
    main()
