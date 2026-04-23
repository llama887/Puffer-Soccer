"""2D heatmap of V(s) as a function of two defensive-line positions.

Scenario (static):
  - Blue carrier: (0, 0), facing +x, ball 2 units ahead at (2, 0).
  - Blue goalie: (-49.5, 0).
  - Red goalie:   (+49.5, 0).
  - Blue teammate LINE: 4 agents at (blue_line_x, y) for y in {-20, -7, +7, +20}.
  - Red defensive LINE: 4 agents at (red_line_x, y) for y in {-20, -7, +7, +20}.

Sweep:
  x-axis: blue_line_x in [-45, +45] (full field)
  y-axis: red_line_x  in [-15, +45] (red from "just past carrier" to own third)

Color: V of the carrier as predicted by the trained critic.

Reference structure drawn on the heatmap:
  - vertical line at blue_line_x = 0 (carrier's x)
  - horizontal line at red_line_x = 0 (carrier's x)
  - diagonal blue_line_x == red_line_x (blue and red lines coincide)
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
LINE_YS = (-20.0, -7.0, 7.0, 20.0)  # 4-agent vertical line

# Fixed:
CARRIER_POS = np.array([0.0, 0.0], dtype=np.float32)
CARRIER_ROT = 0.0
BALL_POS = np.array([2.0, 0.0], dtype=np.float32)
BLUE_GOALIE = np.array([-49.5, 0.0], dtype=np.float32)
RED_GOALIE = np.array([49.5, 0.0], dtype=np.float32)


def build_formation(blue_line_x: float, red_line_x: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (agents_xy (10, 2), agents_rot (10,), team (10,), carrier_idx).

    Indices:
      0 blue goalie         -> carrier 0 is reused as CARRIER actually
      0 carrier             at (0,0) rot=0
      1 blue goalie         at (-49.5, 0)
      2..4 blue line        at (blue_line_x, {-20, -7, +7, +20})  -- 4 needed (only 3 slots; ppt=5)

    We have ppt=5 blue: 1 carrier + 1 goalie + 3 line agents (we drop the y=-20 slot to stay at ppt=5).
    Similarly red has ppt=5: 1 goalie + 4 line defenders.
    """
    # Use 3-agent blue line + goalie + carrier = 5 blue
    blue_line_ys = LINE_YS[:3]  # 3 agents
    xy = np.zeros((2 * PPT, 2), dtype=np.float32)
    rot = np.zeros((2 * PPT,), dtype=np.float32)
    team = np.zeros((2 * PPT,), dtype=np.int32)
    team[PPT:] = 1

    # Blue:
    xy[0] = CARRIER_POS
    rot[0] = CARRIER_ROT
    xy[1] = BLUE_GOALIE
    rot[1] = 0.0  # blue goalie faces +x
    for i, y in enumerate(blue_line_ys):
        xy[2 + i] = [blue_line_x, y]
        rot[2 + i] = 0.0

    # Red:
    xy[PPT + 0] = RED_GOALIE
    rot[PPT + 0] = np.pi  # red faces -x
    for i, y in enumerate(LINE_YS):  # 4 red line agents
        xy[PPT + 1 + i] = [red_line_x, y]
        rot[PPT + 1 + i] = np.pi
    return xy, rot, team, 0  # carrier is blue index 0


def eval_ckpt(ckpt_path: Path, grid_blue_x: np.ndarray, grid_red_x: np.ndarray, device: str) -> np.ndarray:
    state = _train.load_checkpoint_state_dict(ckpt_path)
    if "state_dict" in state and "format_version" in state:
        state = state["state_dict"]

    # build one example to get obs dim
    xy0, rot0, team0, carrier_idx = build_formation(0.0, 33.0)
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

    # Build a big batch of all (blue_line_x, red_line_x) cells.
    Nb = len(grid_blue_x)
    Nr = len(grid_red_x)
    obs_batch = np.zeros((Nr * Nb, obs_size), dtype=np.float32)
    for ri, rlx in enumerate(grid_red_x):
        for bi, blx in enumerate(grid_blue_x):
            xy, rot, team, carrier_idx = build_formation(float(blx), float(rlx))
            obs = build_observation_batch(
                agents_xy=xy,
                agents_rot=rot,
                agents_team=team,
                ball_xy_grid=BALL_POS[None],  # (1, 2)
                ball_v=np.zeros(2, dtype=np.float32),
                blue_left=True,
                focus_team=0,
            )[0]  # (F, D)
            # carrier is blue index 0 (the first blue in the focus list)
            obs_batch[ri * Nb + bi] = obs[0]

    with torch.no_grad():
        t = torch.from_numpy(obs_batch).to(device)
        _, values = policy(t)
        v = values.squeeze(-1).cpu().numpy().reshape(Nr, Nb)
    return v


def _draw_field_mini(ax, blue_line_x: float, red_line_x: float) -> None:
    """Render a compact top-down field diagram for one (blue_line_x, red_line_x)."""
    ax.set_xlim(-55, 55)
    ax.set_ylim(-27, 27)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    # Pitch rectangle + midline + goal lines
    ax.add_patch(plt.Rectangle((-50, -25), 100, 50, fill=False, edgecolor="gray", linewidth=0.8))
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    # Carrier + ball
    ax.scatter([CARRIER_POS[0]], [CARRIER_POS[1]], s=55, c="royalblue",
               edgecolors="black", linewidths=0.6, zorder=4)
    ax.scatter([BALL_POS[0]], [BALL_POS[1]], s=14, c="black", zorder=5)
    # Goalies
    ax.scatter([BLUE_GOALIE[0]], [BLUE_GOALIE[1]], s=40, marker="s", c="royalblue",
               edgecolors="black", linewidths=0.5, zorder=3)
    ax.scatter([RED_GOALIE[0]], [RED_GOALIE[1]], s=40, marker="s", c="crimson",
               edgecolors="black", linewidths=0.5, zorder=3)
    # Blue line (3 agents) + red line (4 agents)
    blue_line_ys = LINE_YS[:3]
    for y in blue_line_ys:
        ax.scatter([blue_line_x], [y], s=30, c="royalblue",
                   edgecolors="black", linewidths=0.4, zorder=3)
    for y in LINE_YS:
        ax.scatter([red_line_x], [y], s=30, c="crimson",
                   edgecolors="black", linewidths=0.4, zorder=3)


def plot(
    v_map: np.ndarray,
    grid_blue_x: np.ndarray,
    grid_red_x: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    # Team-color convention: high V = blue winning → blue colormap.
    # Negative V (if any) = red winning → use RdBu diverging so blue/red
    # team colors line up with who is winning.
    vmin = float(v_map.min())
    vmax = float(v_map.max())
    if vmin < -0.02:
        amp = max(abs(vmin), abs(vmax))
        cmap_kwargs = dict(cmap="RdBu", vmin=-amp, vmax=amp)
    else:
        cmap_kwargs = dict(cmap="Blues", vmin=max(vmin, 0.0), vmax=vmax)

    extent = [grid_blue_x.min() - 2.5, grid_blue_x.max() + 2.5,
              grid_red_x.min() - 2.5, grid_red_x.max() + 2.5]

    fig = plt.figure(figsize=(14.0, 9.5))
    # Main heatmap on left; 5 corner renderings stacked on right.
    n_corners = 5
    gs = fig.add_gridspec(
        n_corners, 3, width_ratios=[3.2, 0.12, 1.0],
        hspace=0.65, wspace=0.12,
    )
    ax = fig.add_subplot(gs[:, 0])
    cax = fig.add_subplot(gs[:, 1])
    mini_axes = [fig.add_subplot(gs[i, 2]) for i in range(n_corners)]

    im = ax.imshow(v_map, extent=extent, origin="lower", aspect="auto", **cmap_kwargs)
    ax.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.55, label="blue line at carrier x")
    ax.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.55, label="red line at carrier x")
    lo = max(grid_blue_x.min(), grid_red_x.min())
    hi = min(grid_blue_x.max(), grid_red_x.max())
    xs = np.linspace(lo, hi, 2)
    ax.plot(xs, xs, color="black", linestyle=":", linewidth=1, alpha=0.55,
            label="blue_line_x = red_line_x")
    ax.set_xlabel("blue teammate-line x (blue at +x = forward)", fontsize=11)
    ax.set_ylabel("red defensive-line x (red at +x = in own half)", fontsize=11)
    ax.set_title(
        title + f"\nactual V range on this grid: [{vmin:+.3f}, {vmax:+.3f}]",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("V(s) from blue carrier", fontsize=10)

    # Five representative corners. C splits by whether the blue
    # teammate line is past the red defensive line (C1) or still on
    # the carrier's side of it (C2) — a qualitatively different
    # formation even though both have red deep.
    corners = [
        (+40, -10, "A: red behind carrier,\nblue forward (breakaway)"),
        (-40, -10, "B: red behind carrier,\nblue team trailing"),
        (+40, +35, "C1: red deep in own third,\nblue line past red line"),
        (+20, +35, "C2: red deep in own third,\nblue stalled at red line"),
        (-40, +35, "D: red deep in own third,\nblue team well behind"),
    ]
    ann_colors = ["#1b7837", "#2a9d8f", "#7b3294", "#ef6c00", "#b7410e"]
    for (bx, rx, lbl), col, mini_ax in zip(corners, ann_colors, mini_axes):
        # Marker on main heatmap
        ax.scatter([bx], [rx], s=120, marker="o", facecolors="none",
                   edgecolors=col, linewidths=2.0, zorder=5)
        ax.text(bx, rx, lbl.split(":")[0], color=col, fontsize=10,
                fontweight="bold", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.0", facecolor="white",
                          edgecolor=col, linewidth=1.2))
        # Mini field rendering
        _draw_field_mini(mini_ax, bx, rx)
        # Look up V at nearest cell
        ri = int(np.argmin(np.abs(grid_red_x - rx)))
        bi = int(np.argmin(np.abs(grid_blue_x - bx)))
        vcell = v_map[ri, bi]
        mini_ax.set_title(f"{lbl}\nV = {vcell:+.3f}", fontsize=8,
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
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--evolution-epochs", default="200,12000,24000,36000,46000")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grid_blue_x = np.arange(-45, 46, 5, dtype=np.float32)
    grid_red_x = np.arange(-15, 46, 5, dtype=np.float32)

    # main plot: final checkpoint
    v = eval_ckpt(args.checkpoint, grid_blue_x, grid_red_x, args.device)
    np.save(args.output_dir / "formation_v.npy", v)
    plot(
        v, grid_blue_x, grid_red_x,
        args.output_dir / "formation_value_heatmap.png",
        "V(carrier) under varied defensive-line / blue-line positions\n"
        "(blue carrier at (0,0) with ball; blue+red goalies fixed)",
    )

    # Optional evolution: same scenario across earlier checkpoints
    if args.checkpoint_dir is not None:
        evo = [int(e) for e in args.evolution_epochs.split(",") if e.strip()]
        print(f"[evolution] evaluating {len(evo)} checkpoints")
        vs = []
        for ep in evo:
            cp = args.checkpoint_dir / f"model_{ep:06d}.pt"
            if not cp.exists():
                print(f"  missing {cp}")
                continue
            print(f"  eval {cp.name}")
            vs.append((ep, eval_ckpt(cp, grid_blue_x, grid_red_x, args.device)))

        if vs:
            # shared color scale across evolution. Use diverging RdBu
            # centered at 0 so early critics' negative regions render
            # as red (red winning) vs late critics' all-positive as blue.
            amp = float(max(np.max(np.abs(v)) for _, v in vs))
            fig, axes = plt.subplots(1, len(vs), figsize=(4.3 * len(vs), 4.2), sharey=True)
            if len(vs) == 1:
                axes = [axes]
            extent = [grid_blue_x.min() - 2.5, grid_blue_x.max() + 2.5,
                      grid_red_x.min() - 2.5, grid_red_x.max() + 2.5]
            for ax, (ep, vmap) in zip(axes, vs):
                im = ax.imshow(vmap, extent=extent, origin="lower", cmap="RdBu",
                               vmin=-amp, vmax=amp, aspect="auto")
                ax.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
                ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
                lo = max(grid_blue_x.min(), grid_red_x.min())
                hi = min(grid_blue_x.max(), grid_red_x.max())
                xs = np.linspace(lo, hi, 2)
                ax.plot(xs, xs, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
                ax.set_title(f"ep={ep}", fontsize=10)
                ax.set_xlabel("blue line x", fontsize=9)
            axes[0].set_ylabel("red line x", fontsize=10)
            fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="V(carrier)")
            fig.suptitle(
                "Emergence of formation-aware value: same scenario scored by 5 checkpoints",
                fontsize=11,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            out2 = args.output_dir / "formation_value_evolution.png"
            fig.savefig(out2, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out2}")


if __name__ == "__main__":
    main()
