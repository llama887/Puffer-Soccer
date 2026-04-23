"""Passing-cone heatmap: where does the carrier want to pass vs dribble.

Scenario built per grid cell:
    receiver  - fixed blue teammate positioned past red's defensive line at
                (+40, 0), in the y-gap between red defenders at (+33, -8)
                and (+33, +8).
    carrier   - swept across the field. At cell (cx, cy):
                  position = (cx, cy)
                  rotation = atan2(rec_y - cy, rec_x - cx)  (faces receiver)
                  ball     = (cx, cy) + 2 * (cos, sin)(rotation)
                             i.e. 2 units in front of the carrier along its
                             facing direction, inside kick range (4.0).
    other blues - goalie at (-49.5, 0), two defenders at (-33, -25) and
                  (-33, +8) stay put. The original slot of the receiver is
                  empty on blue's side (one less defender).
    red team   - all 5 in canonical formation.

Since every KICK action kicks along the carrier's rotation (= toward the
receiver), P(KICK) high <=> "the policy wants to pass." P(MOVE) high <=>
the policy prefers to carry the ball itself / dribble.

Expected emergence: a "passing cone" of cells where red defenders do not
block the carrier-to-receiver line -> P(KICK) high. Outside the cone the
policy should prefer MOVE (dribble around defenders first).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "train_pufferl_pass", REPO / "scripts" / "train_pufferl.py"
)
_train = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _train
_spec.loader.exec_module(_train)

from value_heatmap import (  # noqa: E402
    HALF_X,
    HALF_Y,
    canonical_formation,
    build_observation_batch,
    draw_formation,
)

KICK_ACTION_MIN = 5
KICK_ACTION_MAX = 12
NOOP = 0
MOVE_F = 1
MOVE_B = 2
ROT_L = 3
ROT_R = 4
NUM_ACTIONS = 13

RECEIVER_POS = np.array([40.0, 0.0], dtype=np.float32)
BALL_OFFSET = 2.0  # units in front of carrier along its rotation


def build_scenario(players_per_team: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return (base_xy, team, carrier_idx, receiver_idx)."""
    xy, _rot, team = canonical_formation(players_per_team, blue_left=True)
    # Blue idx layout: 0 goalie (x=-49.5), 1..4 defenders at x=-33
    # with y = -25, -8, +8, +25.
    receiver_idx = 4  # originally (-33, +25)
    carrier_idx = 2   # originally (-33, -8)
    xy[receiver_idx] = RECEIVER_POS  # move to (+40, 0)
    return xy, team, carrier_idx, receiver_idx


def eval_pass_scenario(
    ckpt_path: Path,
    players_per_team: int,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    """Return V, P(KICK), P(MOVE), P(ROT), entropy, mode-action per grid cell."""

    base_xy, team, carrier_idx, receiver_idx = build_scenario(players_per_team)

    gx, gy = np.meshgrid(grid_x, grid_y, indexing="xy")
    Ny, Nx = gx.shape
    N = gx.size

    obs_rows: list[np.ndarray] = []
    carrier_rot_grid = np.zeros(N, dtype=np.float32)
    ball_pos_grid = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        cx = float(gx.ravel()[i])
        cy = float(gy.ravel()[i])
        dx = RECEIVER_POS[0] - cx
        dy = RECEIVER_POS[1] - cy
        carrier_rot = float(np.arctan2(dy, dx))
        ball_x = cx + BALL_OFFSET * np.cos(carrier_rot)
        ball_y = cy + BALL_OFFSET * np.sin(carrier_rot)
        xy_i = base_xy.copy()
        xy_i[carrier_idx] = [cx, cy]
        rot_i = np.zeros_like(base_xy[:, 0])
        # default rotations: blues face +x, reds face -x
        rot_i[team == 0] = 0.0
        rot_i[team == 1] = np.pi
        rot_i[carrier_idx] = carrier_rot
        ball_grid = np.array([[ball_x, ball_y]], dtype=np.float32)
        ball_v = np.zeros(2, dtype=np.float32)
        obs = build_observation_batch(
            agents_xy=xy_i,
            agents_rot=rot_i,
            agents_team=team,
            ball_xy_grid=ball_grid,
            ball_v=ball_v,
            blue_left=True,
            focus_team=0,
        )[0]  # (F, D)
        blue_idx = np.where(team == 0)[0]
        carrier_fi = int(np.where(blue_idx == carrier_idx)[0][0])
        obs_rows.append(obs[carrier_fi])
        carrier_rot_grid[i] = carrier_rot
        ball_pos_grid[i] = [ball_x, ball_y]

    obs_stack = np.stack(obs_rows, axis=0).astype(np.float32)

    class _Shim:
        class _Obs:
            shape = (obs_stack.shape[1],)

        class _Act:
            n = NUM_ACTIONS

        single_observation_space = _Obs()
        single_action_space = _Act()

    state = _train.load_checkpoint_state_dict(ckpt_path)
    if "state_dict" in state and "format_version" in state:
        state = state["state_dict"]
    policy = _train.Policy(_Shim()).to(device)
    policy.load_state_dict(state, strict=True)
    policy.eval()

    with torch.no_grad():
        obs_t = torch.from_numpy(obs_stack).to(device)
        logits, values = policy(obs_t)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)
        p_kick = probs[:, KICK_ACTION_MIN : KICK_ACTION_MAX + 1].sum(dim=-1)
        p_move = probs[:, MOVE_F : MOVE_B + 1].sum(dim=-1)
        p_rot = probs[:, ROT_L : ROT_R + 1].sum(dim=-1)
        mode_action = probs.argmax(dim=-1)

    return {
        "v": values.squeeze(-1).cpu().numpy().reshape(Ny, Nx),
        "p_kick": p_kick.cpu().numpy().reshape(Ny, Nx),
        "p_move": p_move.cpu().numpy().reshape(Ny, Nx),
        "p_rot": p_rot.cpu().numpy().reshape(Ny, Nx),
        "entropy": entropy.cpu().numpy().reshape(Ny, Nx),
        "mode_action": mode_action.cpu().numpy().reshape(Ny, Nx),
        "base_xy": base_xy,
        "team": team,
        "carrier_idx": carrier_idx,
        "receiver_idx": receiver_idx,
    }


def plot_panel_grid(
    heatmaps: list[np.ndarray],
    labels: list[str],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    title: str,
    cmap: str,
    output_path: Path,
    base_xy: np.ndarray,
    team: np.ndarray,
    carrier_idx: int,
    receiver_idx: int,
    vmin: float | None = None,
    vmax: float | None = None,
    center: bool = False,
) -> None:
    if vmin is None:
        vmin = float(min(h.min() for h in heatmaps))
    if vmax is None:
        vmax = float(max(h.max() for h in heatmaps))
    if center:
        amp = max(abs(vmin), abs(vmax))
        vmin, vmax = -amp, amp
    fig, axes = plt.subplots(1, len(heatmaps), figsize=(4.2 * len(heatmaps), 4.2), sharey=True)
    if len(heatmaps) == 1:
        axes = [axes]
    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    for ax, hm, lbl in zip(axes, heatmaps, labels):
        im = ax.imshow(
            hm, extent=extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal"
        )
        for gx_edge, side in ((-HALF_X, "blue"), (HALF_X, "red")):
            ax.plot([gx_edge, gx_edge], [-20, 20], color=side, linewidth=3)
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
        # Draw every blue/red except the carrier (which is the sweeping agent).
        draw_formation(ax, base_xy, team, attacker_idx=carrier_idx)
        # mark the receiver with a gold ring
        rx, ry = base_xy[receiver_idx]
        ax.plot(rx, ry, marker="*", markersize=13, markerfacecolor="gold",
                markeredgecolor="black", markeredgewidth=0.8)
        ax.annotate("receiver", xy=(rx, ry), xytext=(rx + 2, ry + 3), fontsize=7, color="black")
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("carrier x (blue attacks +x)")
    axes[0].set_ylabel("carrier y")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle(
        title + "\ncarrier faces receiver; ball 2 units ahead of carrier",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_mode_action(
    heatmaps: list[np.ndarray],
    labels: list[str],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    output_path: Path,
    base_xy: np.ndarray,
    team: np.ndarray,
    carrier_idx: int,
    receiver_idx: int,
) -> None:
    cat_colors = [
        mcolors.to_rgb("#888888"),  # NOOP
        mcolors.to_rgb("#ff7f0e"),  # MOVE
        mcolors.to_rgb("#2ca02c"),  # ROT
        mcolors.to_rgb("#d62728"),  # KICK
    ]
    cmap_cat = mcolors.ListedColormap(cat_colors)

    fig, axes = plt.subplots(1, len(heatmaps), figsize=(4.2 * len(heatmaps), 4.2), sharey=True)
    if len(heatmaps) == 1:
        axes = [axes]
    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    for ax, hm, lbl in zip(axes, heatmaps, labels):
        cat = np.zeros_like(hm)
        cat[hm == NOOP] = 0
        cat[(hm >= MOVE_F) & (hm <= MOVE_B)] = 1
        cat[(hm >= ROT_L) & (hm <= ROT_R)] = 2
        cat[(hm >= KICK_ACTION_MIN) & (hm <= KICK_ACTION_MAX)] = 3
        ax.imshow(
            cat, extent=extent, origin="lower", cmap=cmap_cat,
            vmin=-0.5, vmax=3.5, aspect="equal",
        )
        for gx_edge, side in ((-HALF_X, "blue"), (HALF_X, "red")):
            ax.plot([gx_edge, gx_edge], [-20, 20], color=side, linewidth=3)
        ax.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
        draw_formation(ax, base_xy, team, attacker_idx=carrier_idx)
        rx, ry = base_xy[receiver_idx]
        ax.plot(rx, ry, marker="*", markersize=13, markerfacecolor="gold",
                markeredgecolor="black", markeredgewidth=0.8)
        ax.annotate("receiver", xy=(rx, ry), xytext=(rx + 2, ry + 3), fontsize=7, color="black")
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("carrier x")
    axes[0].set_ylabel("carrier y")
    legend_patches = [
        Patch(facecolor=cat_colors[0], edgecolor="black", label="NOOP"),
        Patch(facecolor=cat_colors[1], edgecolor="black", label="MOVE"),
        Patch(facecolor=cat_colors[2], edgecolor="black", label="ROT"),
        Patch(facecolor=cat_colors[3], edgecolor="black", label="KICK (= pass)"),
    ]
    fig.legend(handles=legend_patches, loc="center right",
               bbox_to_anchor=(0.995, 0.5), frameon=True, fontsize=9,
               title="argmax action", title_fontsize=9)
    fig.suptitle(
        "Carrier action choice (KICK = pass toward receiver)",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


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

    all_v: list[np.ndarray] = []
    all_pk: list[np.ndarray] = []
    all_pm: list[np.ndarray] = []
    all_pr: list[np.ndarray] = []
    all_h: list[np.ndarray] = []
    all_mode: list[np.ndarray] = []
    labels: list[str] = []
    base_xy = team = carrier_idx = receiver_idx = None
    for ep in epochs:
        ckpt = args.checkpoint_dir / f"model_{ep:06d}.pt"
        if not ckpt.exists():
            print(f"Missing checkpoint for epoch {ep}: {ckpt}")
            continue
        print(f"Evaluating {ckpt.name}...", flush=True)
        out = eval_pass_scenario(
            ckpt, args.players_per_team, grid_x, grid_y, device=args.device
        )
        all_v.append(out["v"])
        all_pk.append(out["p_kick"])
        all_pm.append(out["p_move"])
        all_pr.append(out["p_rot"])
        all_h.append(out["entropy"])
        all_mode.append(out["mode_action"])
        labels.append(f"ep={ep}")
        base_xy = out["base_xy"]; team = out["team"]
        carrier_idx = out["carrier_idx"]; receiver_idx = out["receiver_idx"]
        np.savez(
            args.output_dir / f"pass_heatmap_epoch_{ep:06d}.npz",
            v=out["v"], p_kick=out["p_kick"], p_move=out["p_move"],
            p_rot=out["p_rot"], entropy=out["entropy"], mode_action=out["mode_action"],
        )

    kw = dict(base_xy=base_xy, team=team, carrier_idx=carrier_idx, receiver_idx=receiver_idx)
    plot_panel_grid(all_v, labels, grid_x, grid_y,
        "V(s) with carrier ready to pass to receiver past red's defensive line",
        "RdBu_r", args.output_dir / "value_passing.png", center=True, **kw)
    plot_panel_grid(all_pk, labels, grid_x, grid_y,
        "P(any KICK) -- interpret as P(attempt pass toward receiver)",
        "viridis", args.output_dir / "p_pass.png", vmin=0.0, vmax=1.0, **kw)
    plot_panel_grid(all_pm, labels, grid_x, grid_y,
        "P(MOVE)", "viridis",
        args.output_dir / "p_move.png", vmin=0.0, vmax=1.0, **kw)
    plot_panel_grid(all_pr, labels, grid_x, grid_y,
        "P(ROT)", "viridis",
        args.output_dir / "p_rot.png", vmin=0.0, vmax=1.0, **kw)
    plot_panel_grid(all_h, labels, grid_x, grid_y,
        "Action-distribution entropy H(pi|s)", "magma",
        args.output_dir / "entropy.png", **kw)
    plot_mode_action(all_mode, labels, grid_x, grid_y,
        args.output_dir / "mode_action.png", **kw)

    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump({
            "epochs": epochs,
            "receiver_pos": RECEIVER_POS.tolist(),
            "ball_offset_ahead_of_carrier": BALL_OFFSET,
            "carrier_rot_semantics": "atan2(receiver_y - carrier_y, receiver_x - carrier_x); KICK direction equals carrier rotation, so KICK = pass toward receiver",
        }, f, indent=2)


if __name__ == "__main__":
    main()
