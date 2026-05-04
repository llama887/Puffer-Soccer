"""Goalie ablation: V(no red goalie) - V(with red goalie), same sweep.

Shows whether the trained critic agrees that 'goalies are a good idea'
for the team defending their own goal.

Setup is identical to formation_value_heatmap.py except the red goalie
is REMOVED and re-placed as a 5th member of the red defensive line at
y=0 (so the red line is now 5 agents at y in {-20, -10, 0, +10, +20}).

The plot shows diff = V_no_goalie - V_with_goalie with a diverging
RdBu colormap centered at 0:
  - BLUE: diff > 0, V higher when red has no goalie. Blue team
    benefits from red losing its goalie -> the goalie was useful
    for red (expected).
  - RED:  diff < 0, V lower when red has no goalie. Red team
    benefits from trading the goalie for a 5th line defender
    (the state may still be bad for red overall, but marginally
    better without the goalie).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "train_pufferl_fv", REPO / "scripts" / "train_pufferl.py"
)
_train = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _train
_spec.loader.exec_module(_train)

_spec2 = importlib.util.spec_from_file_location(
    "formation_value_heatmap", REPO / "scripts" / "formation_value_heatmap.py"
)
_fvh = importlib.util.module_from_spec(_spec2)
sys.modules[_spec2.name] = _fvh
_spec2.loader.exec_module(_fvh)

from value_heatmap import build_observation_batch  # noqa: E402


PPT = 5
LINE_YS = (-20.0, -7.0, 7.0, 20.0)
RED_LINE_YS_NOGOALIE = (-20.0, -10.0, 0.0, 10.0, 20.0)

CARRIER_POS = np.array([0.0, 0.0], dtype=np.float32)
CARRIER_ROT = 0.0
BALL_POS = np.array([2.0, 0.0], dtype=np.float32)
BLUE_GOALIE = np.array([-49.5, 0.0], dtype=np.float32)


def build_formation_no_goalie(
    blue_line_x: float, red_line_x: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Same layout as build_formation but red has 5 line players, no goalie."""
    blue_line_ys = LINE_YS[:3]
    xy = np.zeros((2 * PPT, 2), dtype=np.float32)
    rot = np.zeros((2 * PPT,), dtype=np.float32)
    team = np.zeros((2 * PPT,), dtype=np.int32)
    team[PPT:] = 1

    # Blue: carrier + goalie + 3 line (same as original)
    xy[0] = CARRIER_POS
    rot[0] = CARRIER_ROT
    xy[1] = BLUE_GOALIE
    rot[1] = 0.0
    for i, y in enumerate(blue_line_ys):
        xy[2 + i] = [blue_line_x, y]
        rot[2 + i] = 0.0

    # Red: 5 line agents, no goalie.
    for i, y in enumerate(RED_LINE_YS_NOGOALIE):
        xy[PPT + i] = [red_line_x, y]
        rot[PPT + i] = np.pi
    return xy, rot, team, 0


def eval_ckpt_no_goalie(
    ckpt_path: Path,
    grid_blue_x: np.ndarray,
    grid_red_x: np.ndarray,
    device: str,
) -> np.ndarray:
    state = _train.load_checkpoint_state_dict(ckpt_path)
    if "state_dict" in state and "format_version" in state:
        state = state["state_dict"]

    obs_size = 16 + 14 * PPT

    class _Shim:
        class _Obs:
            shape = (obs_size,)

        class _Act:
            n = 13

        single_observation_space = _Obs()
        single_action_space = _Act()

    policy = _train.Policy(_Shim()).to(device)
    policy.load_state_dict(state, strict=True)
    policy.eval()

    Nb = len(grid_blue_x)
    Nr = len(grid_red_x)
    obs_batch = np.zeros((Nr * Nb, obs_size), dtype=np.float32)
    for ri, rlx in enumerate(grid_red_x):
        for bi, blx in enumerate(grid_blue_x):
            xy, rot, team, _ = build_formation_no_goalie(float(blx), float(rlx))
            obs = build_observation_batch(
                agents_xy=xy,
                agents_rot=rot,
                agents_team=team,
                ball_xy_grid=BALL_POS[None],
                ball_v=np.zeros(2, dtype=np.float32),
                blue_left=True,
                focus_team=0,
            )[0]
            obs_batch[ri * Nb + bi] = obs[0]

    with torch.no_grad():
        t = torch.from_numpy(obs_batch).to(device)
        _, values = policy(t)
        v = values.squeeze(-1).cpu().numpy().reshape(Nr, Nb)
    return v


def _draw_field_mini_nogoalie(ax, blue_line_x: float, red_line_x: float) -> None:
    """Top-down field rendering of the no-goalie red configuration."""
    ax.set_xlim(-55, 55)
    ax.set_ylim(-27, 27)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(plt.Rectangle((-50, -25), 100, 50, fill=False, edgecolor="gray", linewidth=0.8))
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    # Carrier + ball
    ax.scatter([CARRIER_POS[0]], [CARRIER_POS[1]], s=55, c="royalblue",
               edgecolors="black", linewidths=0.6, zorder=4)
    ax.scatter([BALL_POS[0]], [BALL_POS[1]], s=14, c="black", zorder=5)
    # Blue goalie only (red goalie removed)
    ax.scatter([BLUE_GOALIE[0]], [BLUE_GOALIE[1]], s=40, marker="s", c="royalblue",
               edgecolors="black", linewidths=0.5, zorder=3)
    # Mark the empty red goal with an X
    ax.scatter([49.5], [0], s=60, marker="x", c="crimson", linewidths=1.8, zorder=3)
    # Blue line
    for y in LINE_YS[:3]:
        ax.scatter([blue_line_x], [y], s=30, c="royalblue",
                   edgecolors="black", linewidths=0.4, zorder=3)
    # Red line: 5 agents (formerly goalie now in line at y=0)
    for y in RED_LINE_YS_NOGOALIE:
        ax.scatter([red_line_x], [y], s=30, c="crimson",
                   edgecolors="black", linewidths=0.4, zorder=3)


def plot_delta(
    v_no: np.ndarray,
    v_with: np.ndarray,
    grid_blue_x: np.ndarray,
    grid_red_x: np.ndarray,
    output_path: Path,
) -> None:
    diff = v_no - v_with
    amp = float(np.max(np.abs(diff)))

    extent = [grid_blue_x.min() - 2.5, grid_blue_x.max() + 2.5,
              grid_red_x.min() - 2.5, grid_red_x.max() + 2.5]

    fig = plt.figure(figsize=(14.5, 9.5))
    n_corners = 5
    gs = fig.add_gridspec(
        n_corners, 3, width_ratios=[3.2, 0.14, 1.0],
        hspace=0.65, wspace=0.45,
    )
    ax = fig.add_subplot(gs[:, 0])
    cax = fig.add_subplot(gs[:, 1])
    mini_axes = [fig.add_subplot(gs[i, 2]) for i in range(n_corners)]

    im = ax.imshow(diff, extent=extent, origin="lower",
                   cmap="RdBu", vmin=-amp, vmax=amp, aspect="auto")
    ax.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.55)
    ax.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.55)
    lo = max(grid_blue_x.min(), grid_red_x.min())
    hi = min(grid_blue_x.max(), grid_red_x.max())
    xs = np.linspace(lo, hi, 2)
    ax.plot(xs, xs, color="black", linestyle=":", linewidth=1, alpha=0.55)
    ax.set_xlabel("blue teammate-line x", fontsize=11)
    ax.set_ylabel("red defensive-line x", fontsize=11)
    ax.set_title(
        "Blue value difference between red team with and without goalie",
        fontsize=13,
    )

    cbar = fig.colorbar(im, cax=cax)
    cax.set_title("ΔV", fontsize=10)

    corners = [
        (+40, -10, "A: red behind carrier,\nblue forward"),
        (-40, -10, "B: red behind carrier,\nblue trailing"),
        (+40, +35, "C1: red deep, blue past line"),
        (+20, +35, "C2: red deep, blue at line"),
        (-40, +35, "D: red deep, blue behind"),
    ]
    ann_colors = ["#1b7837", "#2a9d8f", "#7b3294", "#ef6c00", "#b7410e"]
    for (bx, rx, lbl), col, mini_ax in zip(corners, ann_colors, mini_axes):
        ax.scatter([bx], [rx], s=120, marker="o", facecolors="none",
                   edgecolors=col, linewidths=2.0, zorder=5)
        ax.text(bx, rx, lbl.split(":")[0], color=col, fontsize=10,
                fontweight="bold", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.0", facecolor="white",
                          edgecolor=col, linewidth=1.2))
        _draw_field_mini_nogoalie(mini_ax, bx, rx)
        ri = int(np.argmin(np.abs(grid_red_x - rx)))
        bi = int(np.argmin(np.abs(grid_blue_x - bx)))
        d = diff[ri, bi]
        mini_ax.set_title(f"{lbl}\nΔV = {d:+.3f}", fontsize=8,
                          color=col, loc="center")
        for spine in mini_ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(1.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--v-with-goalie",
        type=Path,
        required=True,
        help="path to formation_v.npy produced by formation_value_heatmap.py",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grid_blue_x = np.arange(-45, 46, 5, dtype=np.float32)
    grid_red_x = np.arange(-15, 46, 5, dtype=np.float32)

    v_with = np.load(args.v_with_goalie)
    assert v_with.shape == (len(grid_red_x), len(grid_blue_x)), \
        f"v_with_goalie shape mismatch: got {v_with.shape}"

    print(f"scoring no-goalie formation with {args.checkpoint.name}")
    v_no = eval_ckpt_no_goalie(args.checkpoint, grid_blue_x, grid_red_x, args.device)
    np.save(args.output_dir / "formation_v_no_goalie.npy", v_no)

    plot_delta(
        v_no, v_with, grid_blue_x, grid_red_x,
        args.output_dir / "formation_value_goalie_delta.png",
    )


if __name__ == "__main__":
    main()
