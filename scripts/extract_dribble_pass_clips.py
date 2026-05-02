"""Cherry-picked dribble and long-pass clips.

Dribble: same player touches the ball >=DRIBBLE_MIN_TOUCHES times with no
intervening teammate/opposing touch, and at least MIN_CENTER_FRAC of the
dribble's ball positions sit inside the central box
(|x| <= CENTER_HALF_X, |y| <= CENTER_HALF_Y).

Pass: kicker A at step tA, receiver B (teammate) at step tB; pass distance
= ||ball@tA+1 .. ball@tB||-style Euclidean between touch locations, must
exceed PASS_MIN_DIST. No opposing touch may occur in (tA, tB).

Two-pass implementation: state-only rollout to detect touches & events,
then a deterministic re-run with rgb_array rendering writing only frames
inside selected event windows.
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
    "train_pufferl_dp", REPO / "scripts" / "train_pufferl.py"
)
_train = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _train
_spec.loader.exec_module(_train)

from puffer_soccer.envs.marl2d import make_puffer_env

BALL_DECAY = 0.85
MAX_BALL_SPEED = 5.0
TOUCH_RADIUS = 4.0
IMPULSE_THRESHOLD = 0.05
KICK_ACTION_MIN = 5
KICK_ACTION_MAX = 12

DRIBBLE_MIN_TOUCHES = 5
CENTER_HALF_X = 30.0
CENTER_HALF_Y = 20.0
MIN_CENTER_FRAC = 0.70

PASS_MIN_DIST = 15.0
PASS_MAX_GAP_STEPS = 60  # reject "next touch" that arrives after ball has long-since stopped


def is_kick(a: int) -> bool:
    return KICK_ACTION_MIN <= a <= KICK_ACTION_MAX


def detect_touches(positions, ball, actions):
    """Return list of (step, player_idx, is_kick)."""
    T = positions.shape[0]
    num_players = positions.shape[1]
    touches: list[tuple[int, int, bool]] = []
    for t in range(1, T):
        dvx = ball[t, 2] - ball[t - 1, 2] * BALL_DECAY
        dvy = ball[t, 3] - ball[t - 1, 3] * BALL_DECAY
        imp_sq = dvx * dvx + dvy * dvy
        if imp_sq < IMPULSE_THRESHOLD ** 2 or imp_sq > (MAX_BALL_SPEED * 1.1) ** 2:
            continue
        dists = np.hypot(positions[t - 1, :, 0] - ball[t - 1, 0],
                         positions[t - 1, :, 1] - ball[t - 1, 1])
        if (dists < TOUCH_RADIUS).sum() == 0:
            continue
        p = int(np.argmin(np.where(dists < TOUCH_RADIUS, dists, np.inf)))
        touches.append((t, p, bool(is_kick(int(actions[t - 1, p])))))
    return touches


def find_dribbles(touches, ball, ppt) -> list[dict]:
    """Return list of dribble events dict(start, end, player, n_touches, center_frac, score)."""
    events: list[dict] = []
    # Group into runs of consecutive same-player touches.
    runs: list[list[tuple[int, int, bool]]] = []
    for touch in touches:
        if runs and runs[-1][-1][1] == touch[1]:
            runs[-1].append(touch)
        else:
            runs.append([touch])
    for run in runs:
        if len(run) < DRIBBLE_MIN_TOUCHES:
            continue
        start_step = run[0][0]
        end_step = run[-1][0]
        # Ball positions during the dribble window
        pos = ball[start_step : end_step + 1, :2]
        in_center = (np.abs(pos[:, 0]) <= CENTER_HALF_X) & (np.abs(pos[:, 1]) <= CENTER_HALF_Y)
        center_frac = float(in_center.mean())
        if center_frac < MIN_CENTER_FRAC:
            continue
        # Score: longer dribble preferred; center_frac kicker
        events.append({
            "behavior": "dribble",
            "start": int(start_step),
            "end": int(end_step),
            "player": int(run[0][1]),
            "n_touches": len(run),
            "center_frac": center_frac,
            "score": len(run) + 2.0 * center_frac,
        })
    events.sort(key=lambda e: -e["score"])
    return events


def find_long_passes(touches, ball, ppt) -> list[dict]:
    """Return list of (step_start, step_end, kicker, receiver, distance, score).
    A pass = kicker A at step tA, receiver B (teammate, not A) at next touch
    with no intervening opposing-team touch. Distance between ball positions.
    """
    events: list[dict] = []
    for i, (t0, p0, kicked0) in enumerate(touches):
        if not kicked0:
            continue
        team0 = 0 if p0 < ppt else 1
        # find next touch
        if i + 1 >= len(touches):
            continue
        t1, p1, _ = touches[i + 1]
        team1 = 0 if p1 < ppt else 1
        if p1 == p0 or team1 != team0:
            continue
        if (t1 - t0) > PASS_MAX_GAP_STEPS:
            continue
        # measure pass distance
        pa = ball[t0, :2]
        pb = ball[t1, :2]
        dist = float(np.linalg.norm(pa - pb))
        if dist < PASS_MIN_DIST:
            continue
        events.append({
            "behavior": "long_pass",
            "start": int(t0),
            "end": int(t1),
            "A": int(p0),
            "B": int(p1),
            "dist": dist,
            "gap_steps": int(t1 - t0),
            "score": dist,
        })
    events.sort(key=lambda e: -e["score"])
    return events


def load_policy(ckpt_path: Path, env, device: str):
    state = _train.load_checkpoint_state_dict(ckpt_path)
    if "state_dict" in state and "format_version" in state:
        state = state["state_dict"]
    _train._USE_LSTM = any(
        k.startswith("lstm.") or k.startswith("cell.") or k.startswith("policy.")
        for k in state.keys()
    )
    policy = _train.build_policy(env).to(device)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy


def rollout_state(ckpt_path, seed, ppt, total_steps, device):
    env = make_puffer_env(
        players_per_team=ppt,
        action_mode="discrete",
        game_length=400,
        render_mode="rgb_array",
        seed=seed,
    )
    policy = load_policy(ckpt_path, env, device)
    obs, _ = env.reset(seed=seed)
    st = env.get_state(0)
    T_alloc = total_steps + 2
    num_players = 2 * ppt
    positions = np.zeros((T_alloc, num_players, 2), dtype=np.float32)
    ball = np.zeros((T_alloc, 4), dtype=np.float32)
    actions = np.zeros((T_alloc, num_players), dtype=np.int32)
    positions[0] = st["positions"]
    ball[0] = np.asarray(st["ball"], dtype=np.float32)
    with torch.no_grad():
        for t in range(1, T_alloc):
            o = torch.from_numpy(obs).to(device)
            acts = torch.argmax(policy(o)[0], dim=-1).cpu().numpy().astype(np.int32)
            actions[t - 1] = acts
            obs, *_ = env.step(acts)
            st = env.get_state(0)
            positions[t] = st["positions"]
            ball[t] = np.asarray(st["ball"], dtype=np.float32)
    env.close()
    return positions, ball, actions


def render_selected(ckpt_path, events, output_dir, seed, ppt, total_steps, pre_steps, post_steps, fps, device):
    clip_ranges = []
    for i, e in enumerate(events):
        cs = max(0, int(e["start"]) - pre_steps)
        ce = min(total_steps, int(e["end"]) + post_steps)
        clip_ranges.append((i, cs, ce))
    active: dict[int, list[int]] = {}
    for i, cs, ce in clip_ranges:
        for t in range(cs, ce + 1):
            active.setdefault(t, []).append(i)
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[int, "imageio.core.Format.Writer"] = {}
    def writer_for(i: int):
        w = writers.get(i)
        if w is None:
            e = events[i]
            if e["behavior"] == "dribble":
                name = (f"{i+1:02d}_step{e['start']:04d}_p{e['player']}"
                        f"_n{e['n_touches']}_cf{int(e['center_frac']*100):02d}.mp4")
                meta_text = (
                    f"behavior=dribble\nplayer={e['player']}\n"
                    f"n_touches={e['n_touches']}\ncenter_frac={e['center_frac']:.3f}\n"
                    f"start_step={e['start']}\nend_step={e['end']}\nfps={fps}\n"
                )
            else:
                name = (f"{i+1:02d}_step{e['start']:04d}_p{e['A']}_to_p{e['B']}"
                        f"_dist{int(e['dist']):03d}_gap{e['gap_steps']}.mp4")
                meta_text = (
                    f"behavior=long_pass\nkicker={e['A']}\nreceiver={e['B']}\n"
                    f"distance={e['dist']:.2f}\ngap_steps={e['gap_steps']}\n"
                    f"start_step={e['start']}\nend_step={e['end']}\nfps={fps}\n"
                )
            path = output_dir / name
            w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
            path.with_suffix(".txt").write_text(meta_text)
            writers[i] = w
            print(f"opening {path.name}")
        return writers[i]

    env = make_puffer_env(
        players_per_team=ppt,
        action_mode="discrete",
        game_length=400,
        render_mode="rgb_array",
        seed=seed,
    )
    policy = load_policy(ckpt_path, env, device)
    obs, _ = env.reset(seed=seed)
    f0 = np.asarray(env.render())
    if 0 in active:
        for i in active[0]:
            writer_for(i).append_data(f0)
    with torch.no_grad():
        for t in range(1, total_steps + 2):
            o = torch.from_numpy(obs).to(device)
            a = torch.argmax(policy(o)[0], dim=-1).cpu().numpy().astype(np.int32)
            obs, *_ = env.step(a)
            if t not in active:
                continue
            f = np.asarray(env.render())
            for i in active[t]:
                writer_for(i).append_data(f)
    for w in writers.values():
        w.close()
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="root clips dir; dribble/ and pass/ written underneath")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--total-steps", type=int, default=3200)
    parser.add_argument("--max-dribbles", type=int, default=5)
    parser.add_argument("--max-passes", type=int, default=5)
    parser.add_argument("--pre-steps", type=int, default=15)
    parser.add_argument("--post-steps", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    positions, ball, actions = rollout_state(
        args.checkpoint, args.seed, args.players_per_team, args.total_steps, args.device
    )
    touches = detect_touches(positions, ball, actions)
    print(f"detected {len(touches)} touches across {args.total_steps} steps")

    dribbles = find_dribbles(touches, ball, args.players_per_team)[: args.max_dribbles]
    passes = find_long_passes(touches, ball, args.players_per_team)[: args.max_passes]
    print(f"kept {len(dribbles)} dribbles, {len(passes)} long passes")
    for e in dribbles:
        print(f"  dribble  step {e['start']}-{e['end']}  player={e['player']}  "
              f"n_touches={e['n_touches']}  center_frac={e['center_frac']:.2f}")
    for e in passes:
        print(f"  pass     step {e['start']}-{e['end']}  p{e['A']}->p{e['B']}  "
              f"dist={e['dist']:.1f}  gap={e['gap_steps']}")

    if dribbles:
        render_selected(
            args.checkpoint, dribbles, args.output_dir / "dribble",
            seed=args.seed, ppt=args.players_per_team, total_steps=args.total_steps,
            pre_steps=args.pre_steps, post_steps=args.post_steps, fps=args.fps, device=args.device,
        )
    if passes:
        render_selected(
            args.checkpoint, passes, args.output_dir / "pass",
            seed=args.seed, ppt=args.players_per_team, total_steps=args.total_steps,
            pre_steps=args.pre_steps, post_steps=args.post_steps, fps=args.fps, device=args.device,
        )


if __name__ == "__main__":
    main()
