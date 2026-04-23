"""Policy-behavior heatmaps over ball position for training checkpoints.

For each ball position in a 2D grid we *place a blue attacker on the ball*,
keep everyone else in the canonical formation, build that attacker's ego-
centric observation in Python (obs layout matches the C env at commit
1f266f4, verified bit-exact), forward-pass the policy, and extract several
behavioral quantities:

  - V (critic output): value the attacker assigns to having the ball here
  - P(any KICK): sum of the 8 KICK-bucket action probabilities
  - Action entropy: H(pi(a|s)) over the 13 discrete actions
  - Mode action: argmax action, visualised as an arrow field showing kick
    direction (for KICK modes) or move vector (for MOVE_F/MOVE_B/ROT modes)

These give a much richer view than the passive V(ball position) plot: where
does blue *want* to kick, where does blue commit to a single action, and
where is the policy indecisive.
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
    "train_pufferl_polh", REPO / "scripts" / "train_pufferl.py"
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

FIELD_X = 110.0
FIELD_Y = 76.0

KICK_ACTION_MIN = 5
KICK_ACTION_MAX = 12
NUM_ACTIONS = 13
NOOP = 0
MOVE_F = 1
MOVE_B = 2
ROT_L = 3
ROT_R = 4

# Kick scales per bucket (from binding.c:discrete_kick_scale). For arrow
# rendering we only use the *direction*, not the magnitude, so 8 kick actions
# correspond to the player's current forward rotation.


def eval_policy_on_attacker_grid(
    ckpt_path: Path,
    players_per_team: int,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    """Return per-cell V, P(kick), entropy, mode-action over the grid."""

    # baseline formation + rotations
    xy, rot, team = canonical_formation(players_per_team, blue_left=True)
    # pick "attacker": blue most forward (max x among blue — ties broken by y)
    blue_idx = np.where(team == 0)[0]
    attacker = int(blue_idx[np.argmax(xy[blue_idx, 0])])

    gx, gy = np.meshgrid(grid_x, grid_y, indexing="xy")  # (Ny, Nx)
    N = gx.size
    Ny, Nx = gx.shape

    # For each cell we need a *different* observation because the attacker's
    # position changes. build_observation_batch currently fixes agents and only
    # varies ball_xy; we'd need to loop. That's fine — N is ~300, tiny.

    obs_rows: list[np.ndarray] = []
    for i in range(N):
        bx, by = float(gx.ravel()[i]), float(gy.ravel()[i])
        xy_i = xy.copy()
        # move attacker to the ball position
        xy_i[attacker] = [bx, by]
        # have the attacker face +x (toward red goal) so KICK actions shoot
        # forward into red's half. Matches the canonical blue-left setup.
        rot_i = rot.copy()
        rot_i[attacker] = 0.0
        # ball at (bx, by), zero velocity
        ball_grid = np.array([[bx, by]], dtype=np.float32)
        ball_v = np.zeros(2, dtype=np.float32)
        obs = build_observation_batch(
            agents_xy=xy_i,
            agents_rot=rot_i,
            agents_team=team,
            ball_xy_grid=ball_grid,
            ball_v=ball_v,
            blue_left=True,
            focus_team=0,
        )[0]  # (F, D), take cell 0
        # pick the attacker's row within focus_team blue (attacker's index in blue_idx)
        attacker_fi = int(np.where(blue_idx == attacker)[0][0])
        obs_rows.append(obs[attacker_fi])

    obs_stack = np.stack(obs_rows, axis=0).astype(np.float32)

    # policy shim
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
    }


def plot_panel_grid(
    heatmaps: list[np.ndarray],
    labels: list[str],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    title: str,
    cmap: str,
    output_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    center: bool = False,
    formation_xy: np.ndarray | None = None,
    formation_team: np.ndarray | None = None,
    attacker_idx: int | None = None,
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
        if formation_xy is not None:
            draw_formation(ax, formation_xy, formation_team, attacker_idx=attacker_idx)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("ball x (blue attacks +x)")
    axes[0].set_ylabel("ball y")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle(title, fontsize=12)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_mode_action_arrows(
    heatmaps: list[np.ndarray],
    labels: list[str],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    output_path: Path,
    formation_xy: np.ndarray | None = None,
    formation_team: np.ndarray | None = None,
    attacker_idx: int | None = None,
) -> None:
    """Plot mode-action category (NOOP/MOVE/ROT/KICK) with a proper legend."""
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    # fixed colormap: 0=NOOP, 1=MOVE, 2=ROT, 3=KICK
    cat_colors = [
        mcolors.to_rgb("#888888"),  # NOOP  gray
        mcolors.to_rgb("#ff7f0e"),  # MOVE  orange (tab:orange)
        mcolors.to_rgb("#2ca02c"),  # ROT   green (tab:green)
        mcolors.to_rgb("#d62728"),  # KICK  red   (tab:red)
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
        if formation_xy is not None:
            draw_formation(ax, formation_xy, formation_team, attacker_idx=attacker_idx)
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("ball x")
    axes[0].set_ylabel("ball y")

    legend_patches = [
        Patch(facecolor=cat_colors[0], edgecolor="black", label="NOOP (no action)"),
        Patch(facecolor=cat_colors[1], edgecolor="black", label="MOVE (forward / back)"),
        Patch(facecolor=cat_colors[2], edgecolor="black", label="ROT (rotate in place)"),
        Patch(facecolor=cat_colors[3], edgecolor="black", label="KICK (any of 8 kick actions)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        frameon=True,
        fontsize=9,
        title="argmax action",
        title_fontsize=9,
    )
    fig.suptitle("Mode action category (argmax over 13 discrete actions)", fontsize=12)
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
    for ep in epochs:
        ckpt = args.checkpoint_dir / f"model_{ep:06d}.pt"
        if not ckpt.exists():
            print(f"Missing checkpoint for epoch {ep}: {ckpt}")
            continue
        print(f"Evaluating {ckpt.name}...", flush=True)
        out = eval_policy_on_attacker_grid(
            ckpt, args.players_per_team, grid_x, grid_y, device=args.device
        )
        all_v.append(out["v"])
        all_pk.append(out["p_kick"])
        all_pm.append(out["p_move"])
        all_pr.append(out["p_rot"])
        all_h.append(out["entropy"])
        all_mode.append(out["mode_action"])
        labels.append(f"ep={ep}")
        np.savez(
            args.output_dir / f"policy_heatmap_epoch_{ep:06d}.npz",
            v=out["v"], p_kick=out["p_kick"], p_move=out["p_move"],
            p_rot=out["p_rot"], entropy=out["entropy"], mode_action=out["mode_action"],
        )

    # reconstruct formation and which agent gets moved, for overlays
    fxy, _frot, fteam = canonical_formation(args.players_per_team, blue_left=True)
    blue_idx_ = np.where(fteam == 0)[0]
    attacker_ = int(blue_idx_[np.argmax(fxy[blue_idx_, 0])])

    kw = dict(formation_xy=fxy, formation_team=fteam, attacker_idx=attacker_)
    plot_panel_grid(all_v, labels, grid_x, grid_y,
        "V(s) with blue attacker ON the ball\n(hollow blue dot = attacker's original slot, now at ball)", "RdBu_r",
        args.output_dir / "value_attacker_has_ball.png", center=True, **kw)
    plot_panel_grid(all_pk, labels, grid_x, grid_y,
        "P(any KICK) by the attacker", "viridis",
        args.output_dir / "p_kick.png", vmin=0.0, vmax=1.0, **kw)
    plot_panel_grid(all_pm, labels, grid_x, grid_y,
        "P(MOVE) by the attacker", "viridis",
        args.output_dir / "p_move.png", vmin=0.0, vmax=1.0, **kw)
    plot_panel_grid(all_pr, labels, grid_x, grid_y,
        "P(ROT) by the attacker", "viridis",
        args.output_dir / "p_rot.png", vmin=0.0, vmax=1.0, **kw)
    plot_panel_grid(all_h, labels, grid_x, grid_y,
        "Action-distribution entropy H(pi|s)", "magma",
        args.output_dir / "entropy.png", **kw)
    plot_mode_action_arrows(all_mode, labels, grid_x, grid_y,
        args.output_dir / "mode_action_category.png", **kw)

    # metadata
    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump({"epochs": epochs, "scenario": "blue attacker placed on ball"}, f, indent=2)


if __name__ == "__main__":
    main()
