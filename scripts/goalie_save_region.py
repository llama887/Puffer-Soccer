"""Where does the critic value the red goalie?  (save-region test)

The existing `formation_value_goalie_delta.py` conflates two things:
  - presence of a red goalie at (+49.5, 0),
  - vs a 5th red line-defender added to the defensive line.

So the 'goalie removal' in that plot is actually 'goalie replaced with
line defender', and the critic has no reason to reward 'goalie-at-net'
over 'extra-line-defender' when the ball is at midfield (47 units from
the goal).

This script isolates goalie value by:

  1. Keeping the 4 red line-defenders fixed at x=+40.
  2. Sweeping the BLUE CARRIER position (with ball) over red's attacking
     third: x in [+15, +48], y in [-25, +25].
  3. Comparing two red configurations:
       A: 5th red = GOALIE at (+49.5, 0)            (defender in the net)
       B: 5th red = SAME player moved to a 'retired'
          corner at (+45, +24)                       (away from the net)
     Team sizes stay at 5v5, so the only difference is where that one
     red player is positioned.

ΔV = V(no-goalie) − V(with-goalie). Positive = blue benefits from the
goalie being out of position, i.e. the critic values goalie positioning.

If goalie-like positioning is an emergent behavior the critic has
learned to reward, this plot should show a strong positive hotspot
right in front of red's net.
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

from value_heatmap import build_observation_batch  # noqa: E402


PPT = 5
RED_LINE_X = 40.0
# Realistic 'game in progress' allocation:
#   variant A (with goalie):  2 line defenders + goalie at net + 2 offense
#   variant B (no goalie):    3 line defenders (goalie collapsed into the
#                              line at y=0)          + 2 offense
# Blue: 2 forwards already past red's line + 1 central midfielder + goalie.
RED_LINE_YS_A = (-10.0, 10.0)
RED_LINE_YS_B = (-10.0, 0.0, 10.0)
RED_OFFENSE_POS = np.array([[-25.0, -7.0], [-25.0, 7.0]], dtype=np.float32)

BLUE_TEAMMATE_POS = np.array(
    [
        [44.0, -15.0],  # forward L, past red line
        [44.0, 15.0],   # forward R, past red line
        [5.0, 0.0],     # central midfielder, behind red line
    ],
    dtype=np.float32,
)

BLUE_GOALIE = np.array([-49.5, 0.0], dtype=np.float32)
RED_GOALIE_AT_NET = np.array([49.5, 0.0], dtype=np.float32)

BALL_OFFSET = np.array([2.0, 0.0], dtype=np.float32)  # ball slightly ahead


def build_formation(
    carrier_xy: np.ndarray,
    with_goalie: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xy = np.zeros((2 * PPT, 2), dtype=np.float32)
    rot = np.zeros((2 * PPT,), dtype=np.float32)
    team = np.zeros((2 * PPT,), dtype=np.int32)
    team[PPT:] = 1

    # Blue: carrier (swept) + goalie + 2 forwards past red line + 1 mid.
    xy[0] = carrier_xy
    rot[0] = 0.0
    xy[1] = BLUE_GOALIE
    rot[1] = 0.0
    for i, pos in enumerate(BLUE_TEAMMATE_POS):
        xy[2 + i] = pos
        rot[2 + i] = 0.0

    # Red line defenders (2 in variant A, 3 in variant B).
    line_ys = RED_LINE_YS_A if with_goalie else RED_LINE_YS_B
    idx = PPT
    for y in line_ys:
        xy[idx] = [RED_LINE_X, y]
        rot[idx] = np.pi
        idx += 1
    # Red 2 offense in blue half (same in both variants).
    for pos in RED_OFFENSE_POS:
        xy[idx] = pos
        rot[idx] = np.pi
        idx += 1
    # 5th slot is the goalie in A (already filled by line defenders in B).
    if with_goalie:
        xy[idx] = RED_GOALIE_AT_NET
        rot[idx] = np.pi
        idx += 1
    assert idx == 2 * PPT, f"formation under/overfilled: idx={idx}"
    return xy, rot, team


def eval_ckpt(
    ckpt_path: Path,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    with_goalie: bool,
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

    Nx = len(grid_x)
    Ny = len(grid_y)
    obs_batch = np.zeros((Ny * Nx, obs_size), dtype=np.float32)
    for yi, cy in enumerate(grid_y):
        for xi, cx in enumerate(grid_x):
            carrier_xy = np.array([cx, cy], dtype=np.float32)
            xy, rot, team = build_formation(carrier_xy, with_goalie=with_goalie)
            ball_xy = carrier_xy + BALL_OFFSET
            obs = build_observation_batch(
                agents_xy=xy,
                agents_rot=rot,
                agents_team=team,
                ball_xy_grid=ball_xy[None],
                ball_v=np.zeros(2, dtype=np.float32),
                blue_left=True,
                focus_team=0,
            )[0]
            obs_batch[yi * Nx + xi] = obs[0]

    with torch.no_grad():
        t = torch.from_numpy(obs_batch).to(device)
        _, values = policy(t)
        v = values.squeeze(-1).cpu().numpy().reshape(Ny, Nx)
    return v


def _draw_field_cartoon(ax, with_goalie: bool) -> None:
    ax.set_xlim(-55, 55)
    ax.set_ylim(-27, 27)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(plt.Rectangle((-50, -25), 100, 50, fill=False, edgecolor="gray", linewidth=0.8))
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

    # Carrier at a representative sweep point (mid-attacking third).
    carrier = np.array([35.0, 0.0])
    ball = carrier + BALL_OFFSET
    ax.scatter([carrier[0]], [carrier[1]], s=55, c="royalblue",
               edgecolors="black", linewidths=0.6, zorder=4)
    ax.scatter([ball[0]], [ball[1]], s=14, c="black", zorder=5)
    ax.scatter([BLUE_GOALIE[0]], [BLUE_GOALIE[1]], s=40, marker="s", c="royalblue",
               edgecolors="black", linewidths=0.5, zorder=3)
    for pos in BLUE_TEAMMATE_POS:
        ax.scatter([pos[0]], [pos[1]], s=26, c="royalblue",
                   edgecolors="black", linewidths=0.4, zorder=3)

    line_ys = RED_LINE_YS_A if with_goalie else RED_LINE_YS_B
    for y in line_ys:
        ax.scatter([RED_LINE_X], [y], s=26, c="crimson",
                   edgecolors="black", linewidths=0.4, zorder=3)
    for pos in RED_OFFENSE_POS:
        ax.scatter([pos[0]], [pos[1]], s=26, c="crimson",
                   edgecolors="black", linewidths=0.4, zorder=3)
    if with_goalie:
        ax.scatter([RED_GOALIE_AT_NET[0]], [RED_GOALIE_AT_NET[1]],
                   s=40, marker="s", c="crimson",
                   edgecolors="black", linewidths=0.5, zorder=3)


def plot(
    v_with: np.ndarray,
    v_no: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    output_path: Path,
) -> None:
    extent = [grid_x.min() - 1.0, grid_x.max() + 1.0,
              grid_y.min() - 1.0, grid_y.max() + 1.0]

    diff = v_no - v_with
    damp = max(float(np.max(np.abs(diff))), 1e-6)

    fig = plt.figure(figsize=(13.5, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[3.6, 0.14, 1.7],
        height_ratios=[1.0, 1.0],
    )
    ax_diff = fig.add_subplot(gs[:, 0])
    cax_diff = fig.add_subplot(gs[:, 1])
    leg_a = fig.add_subplot(gs[0, 2])
    leg_b = fig.add_subplot(gs[1, 2])

    im_diff = ax_diff.imshow(
        diff, extent=extent, origin="lower", aspect="auto",
        cmap="RdBu", vmin=-damp, vmax=damp,
    )
    ax_diff.set_xlabel("blue carrier x", fontsize=11)
    ax_diff.set_ylabel("blue carrier y", fontsize=11)
    ax_diff.set_title(
        "ΔV = V(no goalie) − V(with goalie)  —  where does the red goalie add value?",
        fontsize=11,
    )

    # Reference lines + position markers on the diff heatmap.
    # Red defensive line (x=+40) — present in both variants; the goal line
    # (x=+49.5) as the right-edge reference; midfield y=0.
    ax_diff.axvline(RED_LINE_X, color="crimson", linestyle="--",
                    linewidth=1.0, alpha=0.7)
    ax_diff.text(RED_LINE_X + 0.3, grid_y.max() - 0.5,
                 "red defensive line", color="crimson",
                 fontsize=9, va="top", ha="left")
    ax_diff.axvline(RED_GOALIE_AT_NET[0], color="k", linestyle=":",
                    linewidth=0.8, alpha=0.55)
    ax_diff.axhline(0, color="k", linestyle=":", linewidth=0.8, alpha=0.4)

    # Blue teammates visible inside the sweep: the two forwards at x=+44.
    blue_in_sweep = [p for p in BLUE_TEAMMATE_POS
                     if grid_x.min() <= p[0] <= grid_x.max()
                     and grid_y.min() <= p[1] <= grid_y.max()]
    if blue_in_sweep:
        xs = [p[0] for p in blue_in_sweep]
        ys = [p[1] for p in blue_in_sweep]
        ax_diff.scatter(xs, ys, s=80, c="royalblue", marker="o",
                        edgecolors="black", linewidths=0.8, zorder=5,
                        label="blue teammate")

    # Red line defenders visible in the sweep. Positions at (+40, +/-10)
    # are shared between variants A and B; (+40, 0) is the extra defender
    # that only exists in B; (+49.5, 0) is the goalie that only exists in A.
    shared_ys = [y for y in RED_LINE_YS_A]
    ax_diff.scatter([RED_LINE_X] * len(shared_ys), shared_ys,
                    s=80, c="crimson", marker="o",
                    edgecolors="black", linewidths=0.8, zorder=5,
                    label="red defender (A & B)")
    ax_diff.scatter([RED_LINE_X], [0.0], s=90, c="white",
                    marker="o", edgecolors="crimson", linewidths=1.6,
                    zorder=5, label="3rd red defender (B only)")
    # Goalie sits at x=+49.5, just off the right edge of the sweep but
    # inside the axis limits since we draw a reference line there.
    ax_diff.scatter([RED_GOALIE_AT_NET[0]], [RED_GOALIE_AT_NET[1]],
                    s=90, c="crimson", marker="s",
                    edgecolors="black", linewidths=0.8, zorder=5,
                    label="red goalie at net (A only)")
    ax_diff.set_xlim(extent[0], RED_GOALIE_AT_NET[0] + 1.0)

    ax_diff.legend(loc="lower left", fontsize=8, framealpha=0.88)

    fig.colorbar(im_diff, cax=cax_diff)
    cax_diff.set_title("ΔV", fontsize=10)

    _draw_field_cartoon(leg_a, with_goalie=True)
    leg_a.set_title("A: goalie at net (2 line defenders)", fontsize=9)
    _draw_field_cartoon(leg_b, with_goalie=False)
    leg_b.set_title("B: no goalie (3 line defenders)", fontsize=9)

    fig.suptitle(
        f"Goalie value in a realistic mid-attack state   "
        f"(ΔV ∈ [{diff.min():+.3f}, {diff.max():+.3f}], mean {diff.mean():+.3f}, "
        f"{(diff > 0).mean() * 100:.0f}% of cells positive)",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Sweep the blue carrier over red's attacking third (+ ball at carrier).
    grid_x = np.arange(15, 49, 2, dtype=np.float32)
    grid_y = np.arange(-25, 26, 2, dtype=np.float32)

    print(f"eval with_goalie=True  (grid {len(grid_y)}x{len(grid_x)})")
    v_with = eval_ckpt(args.checkpoint, grid_x, grid_y, with_goalie=True, device=args.device)
    print(f"eval with_goalie=False")
    v_no = eval_ckpt(args.checkpoint, grid_x, grid_y, with_goalie=False, device=args.device)

    np.save(args.output_dir / "goalie_save_region_v_with.npy", v_with)
    np.save(args.output_dir / "goalie_save_region_v_no.npy", v_no)

    plot(
        v_with, v_no, grid_x, grid_y,
        args.output_dir / "goalie_save_region.png",
    )


if __name__ == "__main__":
    main()
