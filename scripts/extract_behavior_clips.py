"""Detect example soccer behaviors in a self-play rollout and export short
mp4 clips, one per (behavior, instance). Intended to give the human a few
candidates per behavior to pick from.

Behaviors detected:
  - dribble     : same player touches the ball >=3 times in a row (no other touch)
  - pass        : kick by A followed by teammate-B's next touch within window
  - goalie_rot  : for a team, the current goalie-id changes to a new teammate
                  within GOALIE_ROTATION_WINDOW frames of the prior goalie's
                  last active step (self re-entry NOT counted)
  - fwd_vs_def  : ball in team's own third & >=1 of that team's players is
                  past midfield in the attacking direction
  - def_vs_off  : ball in opponent's third & >=1 of that team's players is
                  on own side of midfield

Runs a single env with render_mode="rgb_array", buffers the last N frames,
dumps a window centered on each event to an mp4. At most --max-per-behavior
clips per behavior are saved (picking the first K found).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "train_pufferl_clip", REPO / "scripts" / "train_pufferl.py"
)
_train = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _train
_spec.loader.exec_module(_train)

from puffer_soccer.envs.marl2d import make_puffer_env

FIELD_HALF_X = 50.0
GOAL_HALF_Y = 20.0
GOALIE_BASELINE_EPS = 10.0
BALL_DECAY = 0.85
MAX_BALL_SPEED = 5.0
TOUCH_RADIUS = 4.0
IMPULSE_THRESHOLD = 0.05
KICK_ACTION_MIN = 5
KICK_ACTION_MAX = 12
GOALIE_ROTATION_WINDOW = 20


def is_kick(a: int) -> bool:
    return KICK_ACTION_MIN <= a <= KICK_ACTION_MAX


def detect_touch(prev_ball, cur_ball, positions_t, actions_t):
    """Return (player_idx, is_kick) or None. See teamplay_trace.py for logic."""
    dvx = cur_ball[2] - prev_ball[2] * BALL_DECAY
    dvy = cur_ball[3] - prev_ball[3] * BALL_DECAY
    imp_sq = dvx * dvx + dvy * dvy
    if imp_sq < IMPULSE_THRESHOLD ** 2:
        return None
    if imp_sq > (MAX_BALL_SPEED * 1.1) ** 2:
        return None
    dists = np.hypot(positions_t[:, 0] - prev_ball[0], positions_t[:, 1] - prev_ball[1])
    mask = dists < TOUCH_RADIUS
    if not mask.any():
        return None
    within = np.where(mask)[0]
    closest = int(within[int(np.argmin(dists[within]))])
    return (closest, bool(is_kick(int(actions_t[closest]))))


def goalie_id_for_team(positions_t, team_idx, blue_left, ppt):
    """Return the goalie player-idx for a team at step t, or -1."""
    blue_baseline_x = -FIELD_HALF_X if blue_left else FIELD_HALF_X
    red_baseline_x = FIELD_HALF_X if blue_left else -FIELD_HALF_X
    baseline_x = blue_baseline_x if team_idx == 0 else red_baseline_x
    idx = np.arange(2 * ppt)[np.array([0] * ppt + [1] * ppt) == team_idx]
    pos = positions_t[idx]  # (ppt, 2)
    at_baseline = np.abs(pos[:, 0] - baseline_x) < GOALIE_BASELINE_EPS
    in_goal_y = np.abs(pos[:, 1]) < GOAL_HALF_Y
    mask = at_baseline & in_goal_y
    if not mask.any():
        return -1
    dist = np.abs(pos[:, 0] - baseline_x) + 1e-3 * np.abs(pos[:, 1])
    dist[~mask] = np.inf
    return int(idx[int(np.argmin(dist))])


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


def write_clip(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8) as w:
        for f in frames:
            w.append_data(f)
    print(f"wrote {out_path}  ({len(frames)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--total-steps", type=int, default=2400,
                        help="how many env steps to search in (multiple 400-step games)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-per-behavior", type=int, default=5)
    parser.add_argument("--pre-steps", type=int, default=15,
                        help="frames of lead-in before the event")
    parser.add_argument("--post-steps", type=int, default=15,
                        help="frames of trailing context after the event")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ppt = args.players_per_team
    num_players = 2 * ppt

    env = make_puffer_env(
        players_per_team=ppt,
        action_mode="discrete",
        game_length=400,
        render_mode="rgb_array",
        seed=args.seed,
    )
    policy = load_policy(args.checkpoint, env, args.device)

    obs, _ = env.reset(seed=args.seed)

    # make_puffer_env returns the native single-env wrapper; get_state(0)
    # exposes positions/ball/etc.
    def vec_state():
        return env.get_state(0)

    st = vec_state()
    positions = np.zeros((args.total_steps + 2, num_players, 2), dtype=np.float32)
    ball = np.zeros((args.total_steps + 2, 4), dtype=np.float32)
    actions_log = np.zeros((args.total_steps + 2, num_players), dtype=np.int32)
    positions[0] = st["positions"]
    ball[0] = np.asarray(st["ball"], dtype=np.float32)
    blue_left = bool(st["blue_left"])

    # rolling frame buffer: we need up to pre_steps frames of lead-in.
    # We keep all frames (memory cost ~2.2MB*steps) unless steps is huge.
    # For 2400 steps that's ~5GB. Instead, cap and drop old.
    max_buf = args.pre_steps + args.post_steps + 5
    # We actually need per-event windows. To keep memory bounded, we do
    # a single-pass: maintain a ring buffer of the most recent max_buf
    # frames. When an event is detected with its start within max_buf,
    # we capture the prefix then continue capturing frames for
    # post_steps more, then save and move on. Multiple overlapping events
    # are fine.
    frame_buf: deque[np.ndarray] = deque(maxlen=max_buf)
    frame0 = env.render()
    frame_buf.append(np.asarray(frame0))

    # event tracking state
    events: dict[str, list[dict]] = {k: [] for k in ["dribble", "pass", "goalie_rot", "fwd_vs_def", "def_vs_off"]}

    def _count(behavior: str) -> int:
        return len(events[behavior]) + sum(1 for e in pending if e["behavior"] == behavior)
    # dribble state
    touch_streak_player = -1
    touch_streak_len = 0
    touch_streak_start = -1
    last_touch = None  # (step, player, kicked)
    # goalie rotation state
    g_last_id = {0: -1, 1: -1}
    g_last_step = {0: -1, 1: -1}
    # pending "record-until" events — we keep writing frames into buf;
    # when t reaches end_step+post_steps we slice the buffer (which should
    # still contain start_step..cur) and emit.
    pending: list[dict] = []  # each dict: {behavior, start, end, flush_step}

    team = np.zeros(num_players, dtype=np.int32)
    team[ppt:] = 1

    blue_attack_sign = +1.0 if blue_left else -1.0
    red_attack_sign = -blue_attack_sign
    third = FIELD_HALF_X / 3.0

    # "sparse" state windows for fwd_vs_def / def_vs_off — record an event only
    # at the first step of an uninterrupted run of the condition, and only once
    # per run.
    state_active = {"fwd_vs_def": {0: False, 1: False}, "def_vs_off": {0: False, 1: False}}
    state_start = {"fwd_vs_def": {0: -1, 1: -1}, "def_vs_off": {0: -1, 1: -1}}

    done_behaviors = set()

    with torch.no_grad():
        for t in range(1, args.total_steps + 2):
            obs_t = torch.from_numpy(obs).to(args.device)
            logits, _vals = policy(obs_t)
            acts = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int32)
            actions_log[t - 1] = acts
            obs, _r, _term, _trunc, _info = env.step(acts)
            st = vec_state()
            positions[t] = st["positions"]
            ball[t] = np.asarray(st["ball"], dtype=np.float32)

            # render + buffer
            f = np.asarray(env.render())
            frame_buf.append(f)

            # --- touch-based behaviors: dribble / pass
            touch = detect_touch(ball[t - 1], ball[t], positions[t - 1], actions_log[t - 1])
            if touch is not None:
                p, kicked = touch
                # dribble: same player again
                if last_touch is not None and last_touch[1] == p:
                    if touch_streak_player == p:
                        touch_streak_len += 1
                    else:
                        touch_streak_player = p
                        touch_streak_len = 2
                        touch_streak_start = last_touch[0]
                    if touch_streak_len == 3 and "dribble" not in done_behaviors:
                        if _count("dribble") < args.max_per_behavior:
                            pending.append({
                                "behavior": "dribble",
                                "start": max(0, touch_streak_start - args.pre_steps),
                                "end": t,
                                "label": f"p{p}x3",
                            })
                else:
                    # pass: prior touch by teammate with kick
                    if (
                        last_touch is not None
                        and last_touch[1] != p
                        and last_touch[2]  # kicker kicked
                        and int(team[last_touch[1]]) == int(team[p])
                    ):
                        if _count("pass") < args.max_per_behavior:
                            pending.append({
                                "behavior": "pass",
                                "start": max(0, last_touch[0] - args.pre_steps),
                                "end": t,
                                "label": f"p{last_touch[1]}_to_p{p}",
                            })
                    touch_streak_player = p
                    touch_streak_len = 1
                    touch_streak_start = t
                last_touch = (t, p, kicked)

            # --- goalie rotation
            for tm in (0, 1):
                gid = goalie_id_for_team(positions[t], tm, blue_left, ppt)
                if gid == -1:
                    continue
                if (
                    g_last_id[tm] != -1
                    and gid != g_last_id[tm]
                    and (t - g_last_step[tm]) <= GOALIE_ROTATION_WINDOW
                    and _count("goalie_rot") < args.max_per_behavior
                ):
                    pending.append({
                        "behavior": "goalie_rot",
                        "start": max(0, g_last_step[tm] - args.pre_steps),
                        "end": t,
                        "label": f"{'blue' if tm == 0 else 'red'}_{g_last_id[tm]}_to_{gid}",
                    })
                g_last_id[tm] = gid
                g_last_step[tm] = t

            # --- formation-vs-ball metrics (use the ball's current zone)
            ball_x = ball[t, 0]
            if blue_left:
                ball_blue_third = ball_x < -third
                ball_red_third = ball_x > third
            else:
                ball_blue_third = ball_x > third
                ball_red_third = ball_x < -third
            for tm, own_third, attack_sign in (
                (0, ball_blue_third, blue_attack_sign),
                (1, ball_red_third, red_attack_sign),
            ):
                idx = np.arange(num_players)[team == tm]
                team_x = positions[t, idx, 0]
                n_forward = int((team_x * attack_sign > 0.0).sum())
                active = own_third and n_forward >= 1
                if active and not state_active["fwd_vs_def"][tm]:
                    state_active["fwd_vs_def"][tm] = True
                    state_start["fwd_vs_def"][tm] = t
                elif (not active) and state_active["fwd_vs_def"][tm]:
                    start = state_start["fwd_vs_def"][tm]
                    length = t - start
                    if length >= 8 and _count("fwd_vs_def") < args.max_per_behavior:
                        pending.append({
                            "behavior": "fwd_vs_def",
                            "start": max(0, start - args.pre_steps),
                            "end": t,
                            "label": f"{'blue' if tm == 0 else 'red'}_len{length}",
                        })
                    state_active["fwd_vs_def"][tm] = False

                opp_third = ball_red_third if tm == 0 else ball_blue_third
                n_backward = int((team_x * attack_sign < 0.0).sum())
                active_d = opp_third and n_backward >= 1
                if active_d and not state_active["def_vs_off"][tm]:
                    state_active["def_vs_off"][tm] = True
                    state_start["def_vs_off"][tm] = t
                elif (not active_d) and state_active["def_vs_off"][tm]:
                    start = state_start["def_vs_off"][tm]
                    length = t - start
                    if length >= 8 and _count("def_vs_off") < args.max_per_behavior:
                        pending.append({
                            "behavior": "def_vs_off",
                            "start": max(0, start - args.pre_steps),
                            "end": t,
                            "label": f"{'blue' if tm == 0 else 'red'}_len{length}",
                        })
                    state_active["def_vs_off"][tm] = False

            # flush any pending events whose post-buffer has fully filled
            still_pending = []
            for ev in pending:
                if t >= ev["end"] + args.post_steps:
                    # slice buffer: we have the most-recent max_buf frames.
                    # Their step indices span [t - (len(buf)-1), t].
                    buf_len = len(frame_buf)
                    buf_start_step = t - (buf_len - 1)
                    clip_start = max(ev["start"], buf_start_step)
                    clip_end = ev["end"] + args.post_steps
                    if clip_end > t:
                        clip_end = t
                    # convert to buffer indices
                    i0 = clip_start - buf_start_step
                    i1 = clip_end - buf_start_step + 1
                    frames = list(frame_buf)[i0:i1]
                    if not frames:
                        continue
                    events[ev["behavior"]].append(ev)
                    idx_in_behavior = len(events[ev["behavior"]])
                    out_path = (
                        args.output_dir / ev["behavior"]
                        / f"{idx_in_behavior:02d}_step{ev['start']:04d}_{ev['label']}.mp4"
                    )
                    write_clip(frames, out_path, fps=args.fps)
                    # also write a short summary txt alongside
                    meta = out_path.with_suffix(".txt")
                    meta.write_text(
                        f"behavior={ev['behavior']}\nstart_step={ev['start']}\n"
                        f"end_step={ev['end']}\nlabel={ev['label']}\n"
                        f"n_frames={len(frames)}\nfps={args.fps}\n"
                    )
                else:
                    still_pending.append(ev)
            pending = still_pending

            if all(len(events[k]) >= args.max_per_behavior for k in events):
                print("all behaviors saturated; stopping early")
                break

    # flush whatever's still pending at the end
    for ev in pending:
        buf_len = len(frame_buf)
        last_step_in_buf = t
        buf_start_step = last_step_in_buf - (buf_len - 1)
        clip_start = max(ev["start"], buf_start_step)
        i0 = clip_start - buf_start_step
        i1 = buf_len
        frames = list(frame_buf)[i0:i1]
        if frames:
            events[ev["behavior"]].append(ev)
            idx_in_behavior = len(events[ev["behavior"]])
            out_path = (
                args.output_dir / ev["behavior"]
                / f"{idx_in_behavior:02d}_step{ev['start']:04d}_{ev['label']}_truncated.mp4"
            )
            write_clip(frames, out_path, fps=args.fps)

    env.close()

    for k, v in events.items():
        print(f"{k}: {len(v)} clips")


if __name__ == "__main__":
    main()
