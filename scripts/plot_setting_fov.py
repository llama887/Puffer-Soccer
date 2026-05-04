"""Field-of-view overlay for the thesis Settings section.

Loads the final-checkpoint trace, picks one frame, draws the field with
all 10 players + the ball, then highlights one observing player's 180°
forward FOV cone. Other agents are drawn solid if inside the cone
(visible) or hollow/grey if outside (zero-masked in the local
observation), making partial observability concrete.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

FIELD_HALF_X = 50.0
FIELD_HALF_Y = 35.0
GOAL_HALF_Y = 20.0
AGENT_RADIUS = 1.0
BALL_RADIUS = 1.0
VISION_RANGE = np.pi   # 180° FOV — matches binding.c default


def in_fov(observer_xy: np.ndarray, observer_rot: float,
            target_xy: np.ndarray, half_fov: float) -> bool:
    rel = target_xy - observer_xy
    if np.hypot(rel[0], rel[1]) < 1e-6:
        return True
    bearing = np.arctan2(rel[1], rel[0])
    diff = (bearing - observer_rot + np.pi) % (2 * np.pi) - np.pi
    return abs(diff) <= half_fov


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=400,
                        help="step index within the trace")
    parser.add_argument("--env-idx", type=int, default=0,
                        help="env in the (E, T, ...) trace to draw")
    parser.add_argument("--observer", type=int, default=2,
                        help="player index to highlight as the observer")
    parser.add_argument("--players-per-team", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.trace)
    pos_all = data["positions"]   # (E, T, 2n, 2)
    rot_all = data["rotations"] if "rotations" in data.files else None
    ball_all = data["ball"]       # (E, T, 4)
    e, t = args.env_idx, args.frame
    pos = pos_all[e, t]           # (2n, 2)
    if rot_all is not None:
        rot = rot_all[e, t]       # (2n,)
    else:
        # fall back to inferring heading from velocity (positions[t]-positions[t-1])
        rot = np.zeros(pos.shape[0], dtype=np.float32)
        if t > 0:
            d = pos_all[e, t] - pos_all[e, t - 1]
            for i in range(pos.shape[0]):
                if np.hypot(d[i, 0], d[i, 1]) > 1e-3:
                    rot[i] = np.arctan2(d[i, 1], d[i, 0])
    ball = ball_all[e, t]

    n = args.players_per_team
    teams = np.array([0] * n + [1] * n, dtype=np.int32)
    obs_idx = args.observer
    obs_xy = pos[obs_idx]
    obs_rot = float(rot[obs_idx])
    half_fov = VISION_RANGE / 2.0

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.add_patch(mpatches.Rectangle(
        (-FIELD_HALF_X, -FIELD_HALF_Y), 2 * FIELD_HALF_X, 2 * FIELD_HALF_Y,
        fill=False, edgecolor="black", linewidth=1.5,
    ))
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    for sign, color in ((-1, "tab:blue"), (+1, "tab:red")):
        ax.add_patch(mpatches.Rectangle(
            (sign * FIELD_HALF_X - 0.5, -GOAL_HALF_Y),
            1.0, 2 * GOAL_HALF_Y,
            facecolor=color, alpha=0.5,
        ))

    # FOV cone: a wedge of radius R, centered at obs_xy, spanning [obs_rot - π/2, obs_rot + π/2]
    cone_radius = 60.0
    deg_lo = np.degrees(obs_rot - half_fov)
    deg_hi = np.degrees(obs_rot + half_fov)
    ax.add_patch(mpatches.Wedge(
        center=obs_xy, r=cone_radius, theta1=deg_lo, theta2=deg_hi,
        facecolor="gold", alpha=0.18, edgecolor="goldenrod",
        linewidth=1.0, linestyle="--", zorder=2,
    ))

    # Players
    for i, (xy, theta, t_) in enumerate(zip(pos, rot, teams)):
        team_color = "tab:blue" if t_ == 0 else "tab:red"
        if i == obs_idx:
            ax.add_patch(mpatches.Circle(xy, AGENT_RADIUS * 1.2,
                                          facecolor=team_color,
                                          edgecolor="black", linewidth=2.0,
                                          zorder=6))
            ax.text(xy[0], xy[1] - 3.2, "observer",
                    ha="center", fontsize=9, color="black", zorder=7)
        else:
            visible = in_fov(obs_xy, obs_rot, xy, half_fov)
            if visible:
                ax.add_patch(mpatches.Circle(xy, AGENT_RADIUS,
                                              facecolor=team_color,
                                              edgecolor="black",
                                              linewidth=0.5, zorder=5))
            else:
                ax.add_patch(mpatches.Circle(xy, AGENT_RADIUS,
                                              facecolor="white",
                                              edgecolor=team_color,
                                              linewidth=1.2,
                                              linestyle="--",
                                              alpha=0.7, zorder=5))
        # heading arrow
        ax.arrow(xy[0], xy[1], 2.5 * np.cos(theta), 2.5 * np.sin(theta),
                 head_width=0.9, head_length=1.0, fc="black", ec="black",
                 length_includes_head=True, zorder=6, alpha=0.8)

    # Ball
    ball_xy = ball[:2]
    ball_visible = in_fov(obs_xy, obs_rot, ball_xy, half_fov)
    if ball_visible:
        ax.add_patch(mpatches.Circle(ball_xy, BALL_RADIUS,
                                      facecolor="white", edgecolor="black",
                                      linewidth=1.0, zorder=7))
    else:
        ax.add_patch(mpatches.Circle(ball_xy, BALL_RADIUS,
                                      facecolor="white", edgecolor="grey",
                                      linewidth=1.0, linestyle="--",
                                      alpha=0.6, zorder=7))

    ax.set_xlim(-FIELD_HALF_X - 8, FIELD_HALF_X + 8)
    ax.set_ylim(-FIELD_HALF_Y - 8, FIELD_HALF_Y + 8)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.set_title("Decentralized partial observation — 180° forward field of view",
                 fontsize=13)

    legend_handles = [
        mpatches.Patch(facecolor="gold", alpha=0.5,
                        label="observer's 180° FOV"),
        mpatches.Patch(facecolor="tab:blue", edgecolor="black",
                        label="visible (in FOV)"),
        mpatches.Patch(facecolor="white", edgecolor="tab:blue",
                        linestyle="--", linewidth=1.2,
                        label="masked-to-zero (outside FOV)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False, fontsize=9)

    fig.tight_layout()
    out = args.output_dir / "fov_overlay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
