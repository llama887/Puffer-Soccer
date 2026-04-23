"""V(s) heatmaps over ball position for 5 training-progress checkpoints.

For each picked checkpoint we fix the 10 agents in a canonical starting
formation (blue-left, facing +x), sweep the ball over a 2D grid of field
positions, build each blue agent's observation in Python (mirrors the C env's
observation layout at commit 1f266f4), forward-pass the policy to get a per-
agent V(s), and plot the mean blue V as a heatmap.

We do the observation build in Python because the native env has no setter for
ball or agent positions; only field_scale is exposed. Compute is tiny — ~300
grid cells × 5 agents × 5 checkpoints = ~7500 forward passes total.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "train_pufferl_heatmap", REPO / "scripts" / "train_pufferl.py"
)
_train = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _train
_spec.loader.exec_module(_train)

from puffer_soccer.envs.marl2d.constants import (
    INIT_POSITION_11,
    MAX_BALL_SPEED,
)

# env geometry (scale=1.0, blue on left)
FIELD_X = 110.0
FIELD_Y = 76.0
IN_FIELD_X = 100.0
IN_FIELD_Y = 70.0
HALF_X = IN_FIELD_X / 2.0
HALF_Y = IN_FIELD_Y / 2.0
VISION_RANGE = np.pi  # env default


def wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def visible_and_view_mask(focus_rot: np.ndarray, obj_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (is_visible, obj_view_angle) for focus->object."""
    d = wrap_pi(obj_rot - focus_rot)
    half_cone = VISION_RANGE / 2.0
    return np.abs(d) <= half_cone, d


def build_observation_batch(
    agents_xy: np.ndarray,  # (num_agents, 2)
    agents_rot: np.ndarray,  # (num_agents,)
    agents_team: np.ndarray,  # (num_agents,)  0=blue, 1=red
    ball_xy_grid: np.ndarray,  # (N, 2)  N ball positions to evaluate
    ball_v: np.ndarray,  # (2,)  shared across grid cells (typ: zero)
    blue_left: bool,
    focus_team: int,
) -> np.ndarray:
    """Return (N, num_focus_agents, obs_size) observation batch for the focus
    team's agents, given a grid of ball positions. Rotations/positions/ball
    velocity are shared across the grid; only ball_xy varies.
    """

    num_agents = agents_xy.shape[0]
    ppt = num_agents // 2
    obs_size = 16 + 14 * ppt

    focus_idx = np.where(agents_team == focus_team)[0]
    N = ball_xy_grid.shape[0]
    F = focus_idx.shape[0]
    out = np.zeros((N, F, obs_size), dtype=np.float32)

    max_dist = np.hypot(HALF_X, HALF_Y)
    time_left = 1.0

    # per-focus arrays
    fxy = agents_xy[focus_idx]  # (F, 2)
    frot = agents_rot[focus_idx]  # (F,)

    sign = 1.0 if blue_left and focus_team == 0 else -1.0
    if not blue_left and focus_team == 1:
        sign = 1.0
    # team_on_left logic in env: blue is on left iff blue_left; red is on left iff !blue_left.
    team_on_left = (focus_team == 0) == blue_left
    sign = 1.0 if team_on_left else -1.0

    for fi, global_idx in enumerate(focus_idx):
        # self block (18 floats: 7 self + 11 onehot)
        out[:, fi, 0] = time_left
        out[:, fi, 1] = sign * fxy[fi, 0] / HALF_X
        out[:, fi, 2] = sign * fxy[fi, 1] / HALF_Y
        out[:, fi, 3] = sign * np.cos(frot[fi])
        out[:, fi, 4] = sign * np.sin(frot[fi])
        out[:, fi, 5] = 0.0  # last_move
        out[:, fi, 6] = 0.0  # last_rot
        # one-hot player index (11 slots)
        if int(global_idx) < 11:
            out[:, fi, 7 + int(global_idx)] = 1.0

        # ball block (5 floats: present, dist, view_angle, rel_vx, rel_vy)
        d_ball_xy = ball_xy_grid - fxy[fi]  # (N, 2)
        obj_rot_ball = np.arctan2(d_ball_xy[:, 1], d_ball_xy[:, 0])
        vis, obj_view = visible_and_view_mask(
            np.full(N, frot[fi]), obj_rot_ball
        )
        d_ball = np.hypot(d_ball_xy[:, 0], d_ball_xy[:, 1])
        vel_rot = np.arctan2(ball_v[1], ball_v[0])
        abs_val = np.hypot(ball_v[0], ball_v[1])
        out[:, fi, 18] = np.where(vis, 1.0, 0.0)
        out[:, fi, 19] = np.where(vis, d_ball / max_dist, 0.0)
        out[:, fi, 20] = np.where(vis, obj_view / (VISION_RANGE / 2.0), 0.0)
        rel_x = np.cos(vel_rot - frot[fi]) * abs_val / MAX_BALL_SPEED
        rel_y = np.sin(vel_rot - frot[fi]) * abs_val / MAX_BALL_SPEED
        out[:, fi, 21] = np.where(vis, rel_x, 0.0)
        out[:, fi, 22] = np.where(vis, rel_y, 0.0)

        # teammates then opponents, in their per-team order (7 floats each)
        o = 23
        # teammates
        team_mates = [
            j for j in range(num_agents) if agents_team[j] == focus_team and j != global_idx
        ]
        opponents = [j for j in range(num_agents) if agents_team[j] != focus_team]
        for j in team_mates + opponents:
            dxy = agents_xy[j] - fxy[fi]
            obj_rot = np.arctan2(dxy[1], dxy[0])
            vis_j_scalar, obj_view_j = visible_and_view_mask(
                np.array([frot[fi]]), np.array([obj_rot])
            )
            vis_j = bool(vis_j_scalar[0])
            if not vis_j:
                o += 7
                continue
            d_j = float(np.hypot(dxy[0], dxy[1]))
            out[:, fi, o + 0] = 1.0
            out[:, fi, o + 1] = d_j / max_dist
            out[:, fi, o + 2] = float(obj_view_j[0]) / (VISION_RANGE / 2.0)
            out[:, fi, o + 3] = float(np.cos(agents_rot[j] - frot[fi]))
            out[:, fi, o + 4] = float(np.sin(agents_rot[j] - frot[fi]))
            out[:, fi, o + 5] = 0.0  # other.last_move
            out[:, fi, o + 6] = 0.0  # other.last_rot
            o += 7

    return out


def draw_formation(
    ax,
    agents_xy: np.ndarray,
    agents_team: np.ndarray,
    attacker_idx: int | None = None,
) -> None:
    """Draw the canonical formation on a heatmap axis.

    - Blue agents are blue dots, red agents red dots.
    - If `attacker_idx` is provided, that agent's original position gets an
      open (hollow) marker with an arrow hint indicating it is the moved
      agent. This makes clear which slot is being swept across the grid.
    """

    for i in range(agents_xy.shape[0]):
        color = "tab:blue" if agents_team[i] == 0 else "tab:red"
        if attacker_idx is not None and i == attacker_idx:
            ax.plot(
                agents_xy[i, 0], agents_xy[i, 1],
                marker="o", markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=1.5, markersize=9,
            )
            ax.annotate(
                "moved",
                xy=(agents_xy[i, 0], agents_xy[i, 1]),
                xytext=(agents_xy[i, 0] - 6, agents_xy[i, 1] - 4),
                fontsize=6, color=color,
            )
        else:
            ax.plot(
                agents_xy[i, 0], agents_xy[i, 1],
                marker="o", color=color, markersize=6, markeredgecolor="black",
                markeredgewidth=0.5,
            )


def canonical_formation(players_per_team: int, blue_left: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blue on left, red mirrored. Everyone looks forward."""
    num_agents = 2 * players_per_team
    xy = np.zeros((num_agents, 2), dtype=np.float32)
    rot = np.zeros((num_agents,), dtype=np.float32)
    team = np.zeros((num_agents,), dtype=np.int32)
    team[players_per_team:] = 1

    # env uses INIT_POSITION_11 indexed by player-within-team; norm = FIELD[0]
    # (both x and y by FIELD_X = 110 per the C code `* 110.0f`).
    for i in range(players_per_team):
        yy, xx = INIT_POSITION_11[i]  # note: file stores (y, x) ordering
        # env code: new_x = init[pidx][1] * 110; new_y = init[pidx][0] * 110
        new_x = xx * FIELD_X
        new_y = yy * FIELD_X
        # clamp like env: x in [x_out_start, 0] i.e. negative side for blue
        new_x = max(-HALF_X, min(new_x, 0.0))
        new_y = max(-HALF_Y, min(new_y, HALF_Y))
        xy[i, 0] = new_x
        xy[i, 1] = new_y
        rot[i] = 0.0  # facing +x (toward red goal)
    # red: mirror blue
    for i in range(players_per_team):
        xy[players_per_team + i, 0] = -xy[i, 0]
        xy[players_per_team + i, 1] = -xy[i, 1]
        rot[players_per_team + i] = np.pi  # facing -x (toward blue goal)

    if not blue_left:
        xy[:, 0] *= -1
        xy[:, 1] *= -1
        rot += np.pi
    return xy, rot, team


def eval_checkpoint(
    ckpt_path: Path,
    players_per_team: int,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    device: str,
) -> np.ndarray:
    """Return per-cell V_mean_blue over the (Nx, Ny) grid for this checkpoint."""
    xy, rot, team = canonical_formation(players_per_team, blue_left=True)

    # build grid of ball positions (Nx * Ny, 2)
    gx, gy = np.meshgrid(grid_x, grid_y, indexing="xy")  # gx, gy: (Ny, Nx)
    ball_grid = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)  # (N, 2)
    ball_v = np.zeros(2, dtype=np.float32)

    obs_batch = build_observation_batch(
        agents_xy=xy,
        agents_rot=rot,
        agents_team=team,
        ball_xy_grid=ball_grid,
        ball_v=ball_v,
        blue_left=True,
        focus_team=0,
    )
    N, F, D = obs_batch.shape
    obs_flat = obs_batch.reshape(N * F, D)
    obs_tensor = torch.from_numpy(obs_flat).to(device)

    # build a bare minimal env-shim for the Policy constructor
    class _Shim:
        class _Obs:
            shape = (D,)

        class _Act:
            n = 13

        single_observation_space = _Obs()
        single_action_space = _Act()

    state = _train.load_checkpoint_state_dict(ckpt_path)
    if "state_dict" in state and "format_version" in state:
        state = state["state_dict"]
    policy = _train.Policy(_Shim()).to(device)
    policy.load_state_dict(state, strict=True)
    policy.eval()

    with torch.no_grad():
        _logits, values = policy(obs_tensor)
    values = values.squeeze(-1).cpu().numpy().reshape(N, F)
    mean_blue_v = values.mean(axis=1).reshape(gx.shape)  # (Ny, Nx)
    return mean_blue_v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--epochs", default="200,12000,24000,36000,46000")
    parser.add_argument("--grid-x-step", type=float, default=2.5)
    parser.add_argument("--grid-y-step", type=float, default=2.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    grid_x = np.arange(-HALF_X, HALF_X + args.grid_x_step / 2, args.grid_x_step)
    grid_y = np.arange(-HALF_Y, HALF_Y + args.grid_y_step / 2, args.grid_y_step)
    epochs = [int(e) for e in args.epochs.split(",") if e.strip()]

    heatmaps: list[np.ndarray] = []
    labels: list[str] = []
    for ep in epochs:
        ckpt = args.checkpoint_dir / f"model_{ep:06d}.pt"
        if not ckpt.exists():
            print(f"Missing checkpoint for epoch {ep}: {ckpt}")
            continue
        print(f"Evaluating {ckpt.name}...", flush=True)
        v = eval_checkpoint(
            ckpt, args.players_per_team, grid_x, grid_y, device=args.device
        )
        heatmaps.append(v)
        labels.append(f"ep={ep}")
        np.save(args.output_dir / f"v_heatmap_epoch_{ep:06d}.npy", v)

    # shared color scale
    vmin = min(h.min() for h in heatmaps)
    vmax = max(h.max() for h in heatmaps)

    xy_formation, _, team_formation = canonical_formation(args.players_per_team, blue_left=True)
    fig, axes = plt.subplots(1, len(heatmaps), figsize=(4.2 * len(heatmaps), 4.2), sharey=True)
    if len(heatmaps) == 1:
        axes = [axes]
    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    for ax, hm, lbl in zip(axes, heatmaps, labels):
        im = ax.imshow(
            hm, extent=extent, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal"
        )
        # goals at x = +/- HALF_X, y in [-20, 20]
        for gx_edge, side in ((-HALF_X, "blue"), (HALF_X, "red")):
            ax.plot([gx_edge, gx_edge], [-20, 20], color=side, linewidth=3)
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
        draw_formation(ax, xy_formation, team_formation)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("ball x (blue attacks +x)")
    axes[0].set_ylabel("ball y")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="mean V(blue)")
    fig.suptitle("Value heatmap over ball position (mean V across blue team)\n"
                 "Blue/red dots = canonical formation (all 10 agents fixed)", fontsize=10)
    out = args.output_dir / "value_heatmap_grid.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # symmetry-check plots
    fig2, axes2 = plt.subplots(1, len(heatmaps), figsize=(4.2 * len(heatmaps), 4.2), sharey=True)
    if len(heatmaps) == 1:
        axes2 = [axes2]
    for ax, hm, lbl in zip(axes2, heatmaps, labels):
        # For a heatmap over (y, x), flipping x means np.fliplr
        diff = hm - np.fliplr(hm)
        amp = np.max(np.abs(diff))
        im = ax.imshow(
            diff,
            extent=extent,
            origin="lower",
            cmap="RdBu_r",
            vmin=-amp,
            vmax=amp,
            aspect="equal",
        )
        for gx_edge, side in ((-HALF_X, "blue"), (HALF_X, "red")):
            ax.plot([gx_edge, gx_edge], [-20, 20], color=side, linewidth=3)
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
        draw_formation(ax, xy_formation, team_formation)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("ball x")
    axes2[0].set_ylabel("ball y")
    fig2.colorbar(im, ax=axes2, fraction=0.02, pad=0.02, label="V(x,y) - V(-x,y)")
    fig2.suptitle("Asymmetry (blue-side bias): V(x,y) - V(-x,y)", fontsize=12)
    out2 = args.output_dir / "value_heatmap_asymmetry.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {out2}")

    # metadata
    meta = {
        "epochs": epochs,
        "grid_x": grid_x.tolist(),
        "grid_y": grid_y.tolist(),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "formation": "canonical starting (INIT_POSITION_11 for ppt), blue on left, all agents facing forward",
        "ball_velocity": [0.0, 0.0],
    }
    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
