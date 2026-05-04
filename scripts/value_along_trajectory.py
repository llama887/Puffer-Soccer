"""V(s) along scoring trajectories + save per-trace videos.

Two-pass approach for speed:
  pass 1 (headless): scan many episodes WITHOUT rendering. Record per-step
    V and final goals per episode. Identify K blue-scoring and K red-
    scoring single-goal episodes (by seed).
  pass 2 (render): replay each chosen seed with render_mode="rgb_array"
    and save the video.

Much faster because only ~K episodes incur the ~40s pygame render cost.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "train_pufferl_vt", REPO / "scripts" / "train_pufferl.py"
)
_train = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _train
_spec.loader.exec_module(_train)

from puffer_soccer.envs.marl2d import make_puffer_env


def roll_one_episode(env, policy, ppt: int, game_length: int, device: str, record_frames: bool = False, seed: int | None = None, record_obs: bool = False):
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    frames: list[np.ndarray] = []
    v_blue = []
    v_red = []
    obs_seq: list[np.ndarray] = []
    goal_step = None
    scoring_team = None
    # goals counters are cumulative, so we detect new goals via deltas
    st0 = env.get_state(0)
    prev_gb, prev_gr = st0["goals"]
    with torch.no_grad():
        for t in range(game_length):
            obs_t = torch.from_numpy(obs).to(device)
            if record_obs:
                obs_seq.append(obs.copy())
            logits, values = policy(obs_t)
            v = values.squeeze(-1).cpu().numpy()
            v_blue.append(float(v[:ppt].mean()))
            v_red.append(float(v[ppt:].mean()))
            acts = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int32)
            obs, _r, te, tr, _i = env.step(acts)
            if record_frames:
                f = env.render()
                if f is not None:
                    frames.append(f)
            st = env.get_state(0)
            gb, gr = st["goals"]
            new_blue = gb > prev_gb
            new_red = gr > prev_gr
            if new_blue or new_red:
                if goal_step is None:
                    goal_step = t
                    scoring_team = "blue" if new_blue else "red"
                else:
                    return {"multi_goal": True}
            prev_gb, prev_gr = gb, gr
            if bool(np.any(te)) or bool(np.any(tr)):
                break
    return {
        "v_blue": np.asarray(v_blue, dtype=np.float32),
        "v_red": np.asarray(v_red, dtype=np.float32),
        "goal_step": goal_step,
        "scoring_team": scoring_team,
        "frames": frames,
        "obs_seq": (np.stack(obs_seq, axis=0) if obs_seq else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="if set, score trajectories with additional earlier checkpoints sampled uniformly")
    parser.add_argument("--evolution-epochs", default="200,12000,24000,36000,46000")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--num-per-side", type=int, default=5)
    parser.add_argument("--max-episodes", type=int, default=400)
    parser.add_argument("--game-length", type=int, default=400)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "videos").mkdir(exist_ok=True)

    # PASS 1: headless scan
    print(f"[pass 1] headless scan (up to {args.max_episodes} episodes)", flush=True)
    env = make_puffer_env(
        players_per_team=args.players_per_team,
        action_mode="discrete",
        game_length=args.game_length,
        render_mode=None,
        seed=args.seed_start,
    )
    state = _train.load_checkpoint_state_dict(args.checkpoint)
    if "state_dict" in state and "format_version" in state:
        state = state["state_dict"]
    policy = _train.Policy(env).to(args.device)
    policy.load_state_dict(state, strict=True)
    policy.eval()

    chosen_blue: list[tuple[int, dict]] = []  # (seed, result)
    chosen_red: list[tuple[int, dict]] = []
    for i in range(args.max_episodes):
        seed_i = args.seed_start + i
        result = roll_one_episode(env, policy, args.players_per_team,
                                    args.game_length, args.device,
                                    record_frames=False, seed=seed_i,
                                    record_obs=True)
        if result.get("multi_goal"):
            continue
        gs = result.get("goal_step")
        if gs is None:
            continue
        side = result["scoring_team"]
        if side == "blue" and len(chosen_blue) < args.num_per_side:
            chosen_blue.append((seed_i, result))
            print(f"  [blue {len(chosen_blue)}/{args.num_per_side}] seed={seed_i} goal_step={gs}", flush=True)
        elif side == "red" and len(chosen_red) < args.num_per_side:
            chosen_red.append((seed_i, result))
            print(f"  [red  {len(chosen_red)}/{args.num_per_side}] seed={seed_i} goal_step={gs}", flush=True)
        if len(chosen_blue) >= args.num_per_side and len(chosen_red) >= args.num_per_side:
            break
    env.close()

    # PASS 2: render each chosen seed with rgb_array and save video
    print(f"\n[pass 2] rendering {len(chosen_blue) + len(chosen_red)} traces", flush=True)
    env_r = make_puffer_env(
        players_per_team=args.players_per_team,
        action_mode="discrete",
        game_length=args.game_length,
        render_mode="rgb_array",
        seed=args.seed_start,
    )

    def _save(side: str, idx: int, seed: int, result: dict) -> None:
        r2 = roll_one_episode(env_r, policy, args.players_per_team,
                                args.game_length, args.device, record_frames=True, seed=seed)
        frames = r2.get("frames", [])
        vid_path = args.output_dir / "videos" / f"{side}_trace_{idx:02d}.mp4"
        imageio.mimsave(str(vid_path), frames, fps=args.fps, codec="libx264")
        print(f"  [render {side} {idx}] saved {vid_path.name} ({len(frames)} frames)", flush=True)

    for i, (seed_i, result) in enumerate(chosen_blue, start=1):
        _save("blue", i, seed_i, result)
    for i, (seed_i, result) in enumerate(chosen_red, start=1):
        _save("red", i, seed_i, result)
    env_r.close()

    # Unified list of (scoring-team-V, goal_step, obs_seq) for ALL traces.
    # For a blue-scoring trace the scoring V is v_blue; for red-scoring, v_red.
    X_MIN, X_MAX = -150, 0

    def _scoring_v(tr):
        return tr["v_blue"] if tr["scoring_team"] == "blue" else tr["v_red"]

    all_traces = [(sd, tr) for sd, tr in chosen_blue + chosen_red]

    # Save all trajectories (obs, goal_step, scoring_team, V) for re-scoring.
    trace_data = {
        "num_traces": len(all_traces),
        "game_length": args.game_length,
    }
    for i, (sd, tr) in enumerate(all_traces):
        trace_data[f"trace_{i:02d}_obs"] = tr["obs_seq"]
        trace_data[f"trace_{i:02d}_goal_step"] = np.int32(tr["goal_step"])
        trace_data[f"trace_{i:02d}_scoring_team"] = tr["scoring_team"]
        trace_data[f"trace_{i:02d}_seed"] = np.int32(sd)
        trace_data[f"trace_{i:02d}_v_final_ckpt_scoring"] = _scoring_v(tr)
    np.savez(args.output_dir / "trajectories.npz", **trace_data)

    # ---- Plot 1: squashed traces + mean line, clipped at goal
    print("[plotting main]")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    # Interpolate each trace onto a common time grid to compute a mean.
    grid = np.arange(X_MIN, X_MAX + 1, dtype=np.float32)
    interp_stack = []
    seen_blue = False
    seen_red = False
    for i, (sd, tr) in enumerate(all_traces):
        v = _scoring_v(tr)
        t_goal = tr["goal_step"]
        x = np.arange(len(v)) - t_goal
        mask = (x >= X_MIN) & (x <= X_MAX)
        is_blue = tr["scoring_team"] == "blue"
        label = None
        if is_blue and not seen_blue:
            label = "blue scored"
            seen_blue = True
        elif (not is_blue) and not seen_red:
            label = "red scored"
            seen_red = True
        ax.plot(x[mask], v[mask], alpha=0.55, linewidth=1.2,
                color="C0" if is_blue else "C3", label=label)
        # interpolate onto grid for mean (use NaN outside coverage)
        interp = np.full_like(grid, np.nan, dtype=np.float32)
        m2 = (grid >= x.min()) & (grid <= x.max())
        if m2.any():
            interp[m2] = np.interp(grid[m2], x, v)
        interp_stack.append(interp)
    interp_arr = np.stack(interp_stack, axis=0)  # (N, len(grid))
    mean = np.nanmean(interp_arr, axis=0)
    std = np.nanstd(interp_arr, axis=0)
    ax.plot(grid, mean, color="black", linewidth=3.0, label=f"mean over {len(all_traces)} traces")
    ax.fill_between(grid, mean - std, mean + std, alpha=0.18, color="black",
                    label="± 1 std")
    ax.axvline(0, color="k", linestyle="--", alpha=0.7)
    ax.axhline(1.0, color="green", linestyle=":", alpha=0.7, label="V=+1 (goal reward)")
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.4)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xlabel("step relative to goal (t=0 is the goal)", fontsize=11)
    ax.set_ylabel("value", fontsize=11)
    ax.set_title("Value leading to goal", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    out = args.output_dir / "value_along_trajectory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ---- Plot 2: evolution across training checkpoints
    if args.checkpoint_dir is not None:
        evo_epochs = [int(e) for e in args.evolution_epochs.split(",") if e.strip()]
        print(f"[plotting evolution] {len(evo_epochs)} earlier checkpoints")
        fig2, axes2 = plt.subplots(1, len(evo_epochs), figsize=(4.2 * len(evo_epochs), 4.2), sharey=True)
        if len(evo_epochs) == 1:
            axes2 = [axes2]
        ppt = args.players_per_team

        for col, ep in enumerate(evo_epochs):
            ckpt_ep = args.checkpoint_dir / f"model_{ep:06d}.pt"
            state2 = _train.load_checkpoint_state_dict(ckpt_ep)
            if "state_dict" in state2 and "format_version" in state2:
                state2 = state2["state_dict"]
            policy2 = _train.Policy(env_r).to(args.device)
            policy2.load_state_dict(state2, strict=True)
            policy2.eval()

            interp_stack_2 = []
            with torch.no_grad():
                for (sd, tr) in all_traces:
                    obs_seq = tr["obs_seq"]
                    t_goal = tr["goal_step"]
                    ob = torch.from_numpy(obs_seq).to(args.device)
                    # obs_seq has shape (T, num_agents, obs_dim)
                    T, A, D = ob.shape
                    _, values = policy2(ob.reshape(T * A, D))
                    v_per_agent = values.squeeze(-1).cpu().numpy().reshape(T, A)
                    scoring_team_idx = 0 if tr["scoring_team"] == "blue" else 1
                    start = scoring_team_idx * ppt
                    v_team = v_per_agent[:, start : start + ppt].mean(axis=1)
                    x = np.arange(T) - t_goal
                    mask = (x >= X_MIN) & (x <= X_MAX)
                    axes2[col].plot(x[mask], v_team[mask], alpha=0.35, linewidth=0.9,
                                     color="C0" if tr["scoring_team"] == "blue" else "C3")
                    interp_e = np.full_like(grid, np.nan, dtype=np.float32)
                    m2 = (grid >= x.min()) & (grid <= x.max())
                    if m2.any():
                        interp_e[m2] = np.interp(grid[m2], x, v_team)
                    interp_stack_2.append(interp_e)
            arr = np.stack(interp_stack_2, axis=0)
            mean_e = np.nanmean(arr, axis=0)
            std_e = np.nanstd(arr, axis=0)
            axes2[col].plot(grid, mean_e, color="black", linewidth=2.5, label="mean")
            axes2[col].fill_between(grid, mean_e - std_e, mean_e + std_e, alpha=0.18, color="black")
            axes2[col].axvline(0, color="k", linestyle="--", alpha=0.6)
            axes2[col].axhline(1.0, color="green", linestyle=":", alpha=0.5)
            axes2[col].axhline(0.0, color="gray", linestyle=":", alpha=0.4)
            axes2[col].set_xlim(X_MIN, X_MAX)
            axes2[col].set_title(f"ep={ep}", fontsize=11)
            axes2[col].set_xlabel("step to goal", fontsize=9)
            axes2[col].grid(True, alpha=0.3)
            if col == 0:
                axes2[col].legend(fontsize=8, loc="upper left")
                axes2[col].set_ylabel("V of scoring team", fontsize=10)
        fig2.suptitle(
            f"Same {len(all_traces)} trajectories scored by 5 checkpoints' critics: the value function becomes more predictive with training",
            fontsize=12,
        )
        fig2.tight_layout(rect=[0, 0, 1, 0.94])
        out2 = args.output_dir / "value_along_trajectory_evolution.png"
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"wrote {out2}")

    summary = {
        "n_blue": len(chosen_blue),
        "n_red": len(chosen_red),
        "blue_seeds": [s for s, _ in chosen_blue],
        "red_seeds": [s for s, _ in chosen_red],
        "blue_goal_steps": [tr["goal_step"] for _, tr in chosen_blue],
        "red_goal_steps": [tr["goal_step"] for _, tr in chosen_red],
        "blue_final_v": [float(tr["v_blue"][-1]) for _, tr in chosen_blue],
        "red_final_v": [float(tr["v_red"][-1]) for _, tr in chosen_red],
        "checkpoint": str(args.checkpoint),
    }
    with open(args.output_dir / "trajectory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
