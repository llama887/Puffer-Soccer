"""Cherry-picked goalie-transition clips.

We want clips that satisfy the narrative: the previous goalie (A) now has
possession of the ball and is transitioning to offense, while another
teammate (B) slides into the goalie slot. Generic "A left baseline, B
entered baseline within 20 steps" handoffs are too noisy; this script
adds two extra conditions:

  1. Possession check: between A's last goalie-active step and B's first
     goalie-active step, A was within POSSESSION_RADIUS of the ball for at
     least POSSESSION_MIN_STEPS consecutive steps.
  2. Forward-progress check: within that same window, A's x-position in
     the attacking direction advanced by at least FORWARD_PROGRESS_UNITS
     (measured as max(A_x*attack_sign) - min(A_x*attack_sign) within the
     window, with max taken strictly after t_A_last to ensure the
     progression is post-handoff rather than arbitrary).

Events are scored by possession_duration * forward_progress and the top
--max-clips are rendered, longest-scoring first.

Two-pass implementation:
  Pass 1: state-only rollout to collect positions/ball/actions and detect
          event windows.
  Pass 2: re-run the same rollout with render_mode="rgb_array" and record
          only frames inside selected windows.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "train_pufferl_gt", REPO / "scripts" / "train_pufferl.py"
)
_train = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _train
_spec.loader.exec_module(_train)

from puffer_soccer.envs.marl2d import make_puffer_env

FIELD_HALF_X = 50.0
GOAL_HALF_Y = 20.0
GOALIE_BASELINE_EPS = 10.0
GOALIE_ROTATION_WINDOW = 20
POSSESSION_RADIUS = 5.0
POSSESSION_MIN_STEPS = 3
FORWARD_PROGRESS_UNITS = 15.0


def load_policy(ckpt_path: Path, env, device: str):
    state = _train.load_checkpoint_state_dict(ckpt_path)
    if "state_dict" in state and "format_version" in state:
        state = state["state_dict"]
    has_lstm = any(
        k.startswith("lstm.") or k.startswith("cell.") or k.startswith("policy.")
        for k in state.keys()
    )
    _train._USE_LSTM = has_lstm
    policy = _train.build_policy(env).to(device)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy


def compute_goalie_ids(positions: np.ndarray, blue_left: bool, ppt: int) -> np.ndarray:
    """Return (T, 2) array of goalie player-idx per step per team, or -1."""
    T = positions.shape[0]
    num_players = 2 * ppt
    team = np.concatenate([np.zeros(ppt, dtype=np.int32), np.ones(ppt, dtype=np.int32)])
    blue_baseline_x = -FIELD_HALF_X if blue_left else FIELD_HALF_X
    red_baseline_x = FIELD_HALF_X if blue_left else -FIELD_HALF_X
    out = np.full((T, 2), -1, dtype=np.int32)
    for tm, baseline_x in ((0, blue_baseline_x), (1, red_baseline_x)):
        idx = np.arange(num_players)[team == tm]
        pos = positions[:, idx]  # (T, ppt, 2)
        at_baseline = np.abs(pos[:, :, 0] - baseline_x) < GOALIE_BASELINE_EPS
        in_goal_y = np.abs(pos[:, :, 1]) < GOAL_HALF_Y
        mask = at_baseline & in_goal_y
        any_ = np.any(mask, axis=1)
        dist = np.abs(pos[:, :, 0] - baseline_x) + 1e-3 * np.abs(pos[:, :, 1])
        dist_masked = np.where(mask, dist, np.inf)
        best = np.argmin(dist_masked, axis=1)
        out[:, tm] = np.where(any_, idx[best], -1)
    return out


def find_events(positions, ball, blue_left, ppt) -> list[dict]:
    T = positions.shape[0]
    goalie_ids = compute_goalie_ids(positions, blue_left, ppt)
    num_players = 2 * ppt
    team_of = np.concatenate([np.zeros(ppt, dtype=np.int32), np.ones(ppt, dtype=np.int32)])
    blue_attack_sign = +1.0 if blue_left else -1.0
    attack_sign = np.array([blue_attack_sign, -blue_attack_sign])

    events: list[dict] = []

    for tm in (0, 1):
        seq = goalie_ids[:, tm]
        last_id, last_step = -1, -GOALIE_ROTATION_WINDOW - 1
        for t in range(T):
            cur = int(seq[t])
            if cur == -1:
                continue
            handoff = (
                last_id != -1
                and cur != last_id
                and (t - last_step) <= GOALIE_ROTATION_WINDOW
            )
            if handoff:
                A, B = int(last_id), int(cur)
                tA, tB = int(last_step), int(t)
                # Window: start a bit before A left, end a bit after B arrived.
                w0 = max(0, tA - 2)
                w1 = min(T - 1, tB + 30)
                # Possession streak by A in [w0, w1]
                A_pos = positions[w0 : w1 + 1, A]
                ball_pos = ball[w0 : w1 + 1, :2]
                d_a = np.linalg.norm(A_pos - ball_pos, axis=1)
                close = d_a < POSSESSION_RADIUS
                # longest consecutive run of close==True
                best_run = 0
                run = 0
                for c in close:
                    run = run + 1 if c else 0
                    if run > best_run:
                        best_run = run
                if best_run < POSSESSION_MIN_STEPS:
                    last_id, last_step = cur, t
                    continue
                # Forward progress by A in [tA, w1]: max forward minus min forward
                post = positions[tA : w1 + 1, A, 0] * attack_sign[tm]
                forward_progress = float(post.max() - post.min())
                if forward_progress < FORWARD_PROGRESS_UNITS:
                    last_id, last_step = cur, t
                    continue
                # ALSO: B should actually be the new goalie at some step in [tB, tB+8]
                # (already satisfied by seq[tB]=B; extra safety)
                events.append({
                    "team": tm,
                    "A": A,
                    "B": B,
                    "tA_last": tA,
                    "tB_first": tB,
                    "w0": w0,
                    "w1": w1,
                    "possession_steps": int(best_run),
                    "forward_progress": forward_progress,
                    "score": float(best_run) * forward_progress,
                })
            last_id, last_step = cur, t

    # Deduplicate events that overlap heavily (within the same team, same A)
    events.sort(key=lambda e: -e["score"])
    kept: list[dict] = []
    for e in events:
        overlap = any(
            k["team"] == e["team"]
            and abs(k["tA_last"] - e["tA_last"]) < 20
            for k in kept
        )
        if not overlap:
            kept.append(e)
    return kept


def render_clips(
    ckpt_path: Path,
    events: list[dict],
    output_dir: Path,
    seed: int,
    ppt: int,
    total_steps: int,
    fps: int,
    pre_steps: int,
    post_steps: int,
    device: str,
) -> None:
    if not events:
        print("no events — nothing to render")
        return
    # Map event -> (clip_start, clip_end) in env-step coords
    clip_ranges = []
    for i, e in enumerate(events):
        cs = max(0, e["tA_last"] - pre_steps)
        ce = min(total_steps, e["tB_first"] + post_steps)
        clip_ranges.append((i, cs, ce))

    # Build per-step mapping of active clips
    active_per_step: dict[int, list[int]] = {}
    for i, cs, ce in clip_ranges:
        for t in range(cs, ce + 1):
            active_per_step.setdefault(t, []).append(i)

    output_dir.mkdir(parents=True, exist_ok=True)
    # Open writers lazily
    writers: dict[int, imageio.core.Format.Writer] = {}

    env = make_puffer_env(
        players_per_team=ppt,
        action_mode="discrete",
        game_length=400,
        render_mode="rgb_array",
        seed=seed,
    )
    policy = load_policy(ckpt_path, env, device)
    obs, _ = env.reset(seed=seed)

    def open_writer(i: int):
        e = events[i]
        label = f"team{'blue' if e['team']==0 else 'red'}_A{e['A']}_to_B{e['B']}"
        score_tag = f"poss{e['possession_steps']}_fwd{int(e['forward_progress'])}"
        name = f"{i+1:02d}_step{e['tA_last']:04d}_{label}_{score_tag}.mp4"
        path = output_dir / name
        w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
        print(f"opening {path}  (poss={e['possession_steps']}, fwd={e['forward_progress']:.1f})")
        # sidecar metadata
        meta = path.with_suffix(".txt")
        meta.write_text(
            f"behavior=goalie_transition\n"
            f"team={'blue' if e['team']==0 else 'red'}\n"
            f"A_player={e['A']}\nB_player={e['B']}\n"
            f"tA_last={e['tA_last']}\ntB_first={e['tB_first']}\n"
            f"possession_consecutive_steps={e['possession_steps']}\n"
            f"forward_progress_units={e['forward_progress']:.2f}\n"
            f"fps={fps}\n"
        )
        return w

    def writer_for(i: int):
        w = writers.get(i)
        if w is None:
            w = open_writer(i)
            writers[i] = w
        return w

    # step t=0 initial frame
    f0 = np.asarray(env.render())
    if 0 in active_per_step:
        for i in active_per_step[0]:
            writer_for(i).append_data(f0)

    with torch.no_grad():
        for t in range(1, total_steps + 2):
            obs_t = torch.from_numpy(obs).to(device)
            logits, _v = policy(obs_t)
            acts = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int32)
            obs, _r, _term, _trunc, _info = env.step(acts)
            if t not in active_per_step:
                continue
            f = np.asarray(env.render())
            for i in active_per_step[t]:
                writer_for(i).append_data(f)

    for w in writers.values():
        w.close()
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--total-steps", type=int, default=3200)
    parser.add_argument("--max-clips", type=int, default=5)
    parser.add_argument("--pre-steps", type=int, default=15)
    parser.add_argument("--post-steps", type=int, default=25)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ppt = args.players_per_team
    num_players = 2 * ppt

    # Pass 1: state-only rollout
    env = make_puffer_env(
        players_per_team=ppt,
        action_mode="discrete",
        game_length=400,
        render_mode="rgb_array",
        seed=args.seed,
    )
    policy = load_policy(args.checkpoint, env, args.device)
    obs, _ = env.reset(seed=args.seed)
    st = env.get_state(0)
    blue_left = bool(st["blue_left"])
    T_alloc = args.total_steps + 2
    positions = np.zeros((T_alloc, num_players, 2), dtype=np.float32)
    ball = np.zeros((T_alloc, 4), dtype=np.float32)
    positions[0] = st["positions"]
    ball[0] = np.asarray(st["ball"], dtype=np.float32)

    with torch.no_grad():
        for t in range(1, T_alloc):
            obs_t = torch.from_numpy(obs).to(args.device)
            logits, _v = policy(obs_t)
            acts = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int32)
            obs, _r, _term, _trunc, _info = env.step(acts)
            st = env.get_state(0)
            positions[t] = st["positions"]
            ball[t] = np.asarray(st["ball"], dtype=np.float32)
    env.close()

    events = find_events(positions, ball, blue_left, ppt)
    print(f"found {len(events)} candidate events")
    for e in events[:20]:
        print(
            f"  t={e['tA_last']:4d}-{e['tB_first']:4d}  team={'blue' if e['team']==0 else 'red'}  "
            f"A={e['A']} -> B={e['B']}  poss={e['possession_steps']} fwd={e['forward_progress']:.1f}  "
            f"score={e['score']:.1f}"
        )
    events = events[: args.max_clips]
    if not events:
        return

    # Pass 2: render selected event windows
    render_clips(
        args.checkpoint,
        events,
        args.output_dir,
        seed=args.seed,
        ppt=ppt,
        total_steps=args.total_steps,
        fps=args.fps,
        pre_steps=args.pre_steps,
        post_steps=args.post_steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
