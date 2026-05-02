"""Teamplay emergence stats across a repl run's past-iterate checkpoints.

For each matching checkpoint we play self-play (same policy for blue and red)
in the gallant env (this worktree's env code at commit 1f266f4), record every
step's positions + ball + actions, infer which agent touched the ball each
step via ball-velocity impulse detection, and then compute a battery of
team-behavior statistics. Each metric is computed per team and averaged.

Output: one JSON per checkpoint in <output_dir>/stats_epoch_<epoch>.json.

The env does not log a "last touch" player id, so we infer contacts from the
step-to-step change in ball velocity. The env decays ball velocity by
BALL_DECAY each step, so the expected ball velocity next step (absent any
touch) is decay * prev_v. A touch is anything that changes ball velocity by
more than IMPULSE_THRESHOLD beyond that decay baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location(
    "train_pufferl_trace", REPO / "scripts" / "train_pufferl.py"
)
_train = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _train
_spec.loader.exec_module(_train)

from puffer_soccer.envs.marl2d import make_native_vec_env

# env constants that match binding.c at 1f266f4
BALL_RADIUS = 1.0
AGENT_RADIUS = 1.0
LEG_LENGTH = 3.0
BALL_DECAY = 0.85
MAX_BALL_SPEED = 5.0
TOUCH_RADIUS = 4.0
IMPULSE_THRESHOLD = 0.05

KICK_ACTION_MIN = 5
KICK_ACTION_MAX = 12

FIELD_HALF_X = 50.0
FIELD_HALF_Y = 35.0
GOAL_HALF_Y = 20.0
GOALIE_BASELINE_EPS = 10.0
GOALIE_ROTATION_WINDOW = 20  # frames; handoff must happen within this span

MODEL_NAME_RE = re.compile(r"model_(\d+)\.pt$")


def is_kick(action_id: int) -> bool:
    return KICK_ACTION_MIN <= action_id <= KICK_ACTION_MAX


def detect_touches_per_env(
    positions_t: np.ndarray,
    prev_ball: np.ndarray,
    cur_ball: np.ndarray,
    actions_t: np.ndarray,
) -> list[tuple[int, bool]]:
    """Return list of (player_idx, is_kick) touches for one env at one step."""
    expected_vx = prev_ball[2] * BALL_DECAY
    expected_vy = prev_ball[3] * BALL_DECAY
    dvx = cur_ball[2] - expected_vx
    dvy = cur_ball[3] - expected_vy
    impulse_sq = dvx * dvx + dvy * dvy
    if impulse_sq < IMPULSE_THRESHOLD * IMPULSE_THRESHOLD:
        return []
    # reject episode-reset teleports: any impulse larger than MAX_BALL_SPEED
    # can't come from one touch under the env's physics.
    if impulse_sq > (MAX_BALL_SPEED * 1.1) ** 2:
        return []
    bx, by = prev_ball[0], prev_ball[1]
    dists = np.hypot(positions_t[:, 0] - bx, positions_t[:, 1] - by)
    mask = dists < TOUCH_RADIUS
    if not mask.any():
        return []
    within = np.where(mask)[0]
    closest = within[int(np.argmin(dists[within]))]
    kicked = bool(is_kick(int(actions_t[closest])))
    return [(int(closest), kicked)]


def compute_game_stats(
    positions: np.ndarray,
    ball: np.ndarray,
    actions: np.ndarray,
    blue_left: bool,
    ppt: int,
) -> dict[str, float]:
    """Compute all teamplay stats for one full game trajectory.

    positions: (T, 2*ppt, 2)
    ball: (T, 4)  ->  (bx, by, bvx, bvy)
    actions: (T, 2*ppt)
    """

    T = positions.shape[0]
    num_players = 2 * ppt
    team = np.zeros(num_players, dtype=np.int32)
    team[ppt:] = 1  # blue=0, red=1

    # ------------------------------------------------------------ touches
    touches: list[tuple[int, int, bool]] = []  # (t, player_idx, is_kick)
    for t in range(1, T):
        for p, kicked in detect_touches_per_env(
            positions[t - 1], ball[t - 1], ball[t], actions[t - 1]
        ):
            touches.append((t, p, kicked))

    # per-team accumulators
    passes = {0: 0, 1: 0}
    pass_lengths = {0: [], 1: []}
    double_passes = {0: 0, 1: 0}
    triple_passes = {0: 0, 1: 0}
    dribbles = {0: 0, 1: 0}

    # possession: contiguous touches by same team; resets on opposing touch.
    # track pass chain length inside each possession.
    cur_poss_team = -1
    poss_pass_count = 0
    last_touch_player = -1
    last_touch_step = -1
    last_touch_pos = None

    for step, p, kicked in touches:
        p_team = int(team[p])

        # possession start / continue
        if p_team != cur_poss_team:
            # new possession - finalize previous counts happens naturally
            cur_poss_team = p_team
            poss_pass_count = 0

        # dribble: same player touches ball again with no intervening touch
        if last_touch_player == p and p_team == cur_poss_team:
            dribbles[p_team] += 1

        # pass: previous touch by teammate (not self) via kick
        if (
            last_touch_player != -1
            and last_touch_player != p
            and int(team[last_touch_player]) == p_team
        ):
            # kicker must have kicked; receiver's action irrelevant
            if last_touch_kick:
                passes[p_team] += 1
                if last_touch_pos is not None:
                    cur_pos = positions[step - 1, p]
                    pass_lengths[p_team].append(
                        float(np.hypot(cur_pos[0] - last_touch_pos[0], cur_pos[1] - last_touch_pos[1]))
                    )
                poss_pass_count += 1
                if poss_pass_count == 2:
                    double_passes[p_team] += 1
                if poss_pass_count == 3:
                    triple_passes[p_team] += 1

        last_touch_player = p
        last_touch_step = step
        last_touch_pos = positions[step - 1, p].copy()
        last_touch_kick = kicked  # noqa: F841  used above on next iter

    # ------------------------------------------------------------ goalie
    # A goalie exists for a team at step t iff at least one teammate is within
    # GOALIE_BASELINE_EPS of that team's own baseline and within the goalpost
    # y-interval. Team own baselines: blue_left True -> blue own at -half_x,
    # red own at +half_x.
    blue_baseline_x = -FIELD_HALF_X if blue_left else FIELD_HALF_X
    red_baseline_x = FIELD_HALF_X if blue_left else -FIELD_HALF_X

    goalie_frames = {0: 0, 1: 0}
    # per-step goalie player-idx per team (-1 if no one is goalie).
    # If multiple teammates satisfy the goalie condition we pick the one
    # closest to the own baseline (tie-breaker = smallest |y|).
    goalie_id = {0: np.full(T, -1, dtype=np.int32), 1: np.full(T, -1, dtype=np.int32)}
    for tm, baseline_x in ((0, blue_baseline_x), (1, red_baseline_x)):
        idx = np.arange(num_players)[team == tm]
        pos = positions[:, idx]  # (T, ppt, 2)
        at_baseline = np.abs(pos[:, :, 0] - baseline_x) < GOALIE_BASELINE_EPS
        in_goal_y = np.abs(pos[:, :, 1]) < GOAL_HALF_Y
        mask = at_baseline & in_goal_y  # (T, ppt)
        goalie_steps = np.any(mask, axis=1)
        goalie_frames[tm] = int(goalie_steps.sum())
        # rank teammates per step: distance to baseline, break ties on |y|.
        dist = np.abs(pos[:, :, 0] - baseline_x) + 1e-3 * np.abs(pos[:, :, 1])
        dist_masked = np.where(mask, dist, np.inf)
        best = np.argmin(dist_masked, axis=1)  # (T,)
        has_any = goalie_steps
        chosen = np.where(has_any, idx[best], -1)
        goalie_id[tm] = chosen.astype(np.int32)

    # goalie rotation: while some player is the current goalie, another
    # teammate replaces them within GOALIE_ROTATION_WINDOW frames. Counts
    # an event when the active goalie-id changes to a NEW teammate whose
    # onset is within W steps of the previous goalie's last active frame.
    # A -> none -> A (self re-entry) is NOT a rotation; A -> (any gap ≤ W) -> B is.
    rotations = {0: 0, 1: 0}
    for tm in (0, 1):
        seq = goalie_id[tm]
        last_id = -1
        last_step = -GOALIE_ROTATION_WINDOW - 1
        for t in range(T):
            cur = int(seq[t])
            if cur == -1:
                continue
            if last_id != -1 and cur != last_id and (t - last_step) <= GOALIE_ROTATION_WINDOW:
                rotations[tm] += 1
            last_id = cur
            last_step = t

    # -------------------------- attacking-stayers while defending / vice versa
    # For team T: "own third" = ball in the third closest to team T's baseline.
    # Third boundaries at x = -half/3 and x=+half/3.
    third = FIELD_HALF_X / 3.0
    ball_x = ball[:, 0]

    # For each step, is ball in blue's third / red's third / middle?
    if blue_left:
        ball_in_blue_third = ball_x < -third
        ball_in_red_third = ball_x > third
    else:
        ball_in_blue_third = ball_x > third
        ball_in_red_third = ball_x < -third

    # "offensive player while on defense for team T" =
    # while ball in T's own third, at least one of T's players is across midfield.
    # For blue (attacks +x if blue_left), "across midfield" = x > 0.
    off_def_counts = {0: 0.0, 1: 0.0}
    off_def_steps = {0: 0, 1: 0}
    def_off_counts = {0: 0.0, 1: 0.0}  # defensive players while on offense
    def_off_steps = {0: 0, 1: 0}
    off_def_has_attacker_steps = {0: 0, 1: 0}
    def_off_has_defender_steps = {0: 0, 1: 0}

    # blue offensive direction
    blue_attack_sign = +1.0 if blue_left else -1.0
    red_attack_sign = -blue_attack_sign

    for tm, own_third_mask, attack_sign in (
        (0, ball_in_blue_third, blue_attack_sign),
        (1, ball_in_red_third, red_attack_sign),
    ):
        idx = np.arange(num_players)[team == tm]
        team_pos_x = positions[:, idx, 0]  # (T, ppt)
        # An "offensive player" (one left forward while defending) is one past midfield
        # in the attacking direction.
        forward = team_pos_x * attack_sign > 0.0
        n_forward = forward.sum(axis=1)  # (T,)
        # "offensive player while on defense"
        frames = int(own_third_mask.sum())
        off_def_steps[tm] = frames
        if frames > 0:
            off_def_counts[tm] = float(n_forward[own_third_mask].mean())
            off_def_has_attacker_steps[tm] = int((n_forward[own_third_mask] >= 1).sum())

        # "defensive player while on offense" — ball in opponent's third;
        # count team players on own half (negative attack_sign direction).
        opp_third_mask = ball_in_red_third if tm == 0 else ball_in_blue_third
        backward = team_pos_x * attack_sign < 0.0
        n_backward = backward.sum(axis=1)
        frames_off = int(opp_third_mask.sum())
        def_off_steps[tm] = frames_off
        if frames_off > 0:
            def_off_counts[tm] = float(n_backward[opp_third_mask].mean())
            def_off_has_defender_steps[tm] = int((n_backward[opp_third_mask] >= 1).sum())

    # ------------------------------------------------- velocity toward ball
    # per-step velocity by finite differencing positions; project onto
    # unit vector pointing from agent to ball at step t.
    vel = positions[1:] - positions[:-1]  # (T-1, num_players, 2)
    to_ball = ball[:-1, :2][:, None, :] - positions[:-1]  # (T-1, num_players, 2)
    to_ball_norm = np.linalg.norm(to_ball, axis=2, keepdims=True) + 1e-6
    to_ball_unit = to_ball / to_ball_norm
    # signed speed toward ball per (t, player)
    toward = (vel * to_ball_unit).sum(axis=2)  # (T-1, num_players)
    vtb_blue = float(toward[:, team == 0].mean())
    vtb_red = float(toward[:, team == 1].mean())

    # --------------------------------- formation compactness + centroid y
    compact_blue = float(positions[:, team == 0].std(axis=1).mean())
    compact_red = float(positions[:, team == 1].std(axis=1).mean())
    cy_blue_signed = float(positions[:, team == 0, 1].mean())
    cy_red_signed = float(positions[:, team == 1, 1].mean())
    # centroid y sign-flipped so "forward" is always positive for each team
    # (this is useful only if you want a single scalar; we save both signed
    # and sign-normalized). Use attack_sign semantics on the x axis instead
    # for "how far up the field": mean of team's x multiplied by attack_sign.
    cf_blue = float((positions[:, team == 0, 0] * blue_attack_sign).mean())
    cf_red = float((positions[:, team == 1, 0] * red_attack_sign).mean())

    # --------------------------------- ball x histogram entropy (over time)
    nbins = 10
    ball_hist, _ = np.histogram(ball_x, bins=nbins, range=(-FIELD_HALF_X, FIELD_HALF_X))
    p = ball_hist.astype(np.float64) / max(1.0, float(ball_hist.sum()))
    p = p[p > 0]
    ball_x_entropy = float(-(p * np.log(p)).sum())

    # ------------------------------ per-team stats -> averaged single values
    def avg(d):
        return 0.5 * (d[0] + d[1])

    def list_avg(lst):
        return float(np.mean(lst)) if lst else 0.0

    # pass totals averaged across teams
    n_pass = avg(passes)
    n_double = avg(double_passes)
    n_triple = avg(triple_passes)
    n_dribble = avg(dribbles)
    mean_pass_len = 0.5 * (list_avg(pass_lengths[0]) + list_avg(pass_lengths[1]))

    goalie_frac_blue = goalie_frames[0] / max(1, T)
    goalie_frac_red = goalie_frames[1] / max(1, T)
    goalie_frac = 0.5 * (goalie_frac_blue + goalie_frac_red)
    goalie_rotations = 0.5 * (rotations[0] + rotations[1])

    def frac(num_steps, denom_steps):
        return (num_steps / denom_steps) if denom_steps > 0 else 0.0

    off_while_def_frac = 0.5 * (
        frac(off_def_has_attacker_steps[0], off_def_steps[0])
        + frac(off_def_has_attacker_steps[1], off_def_steps[1])
    )
    off_while_def_count = avg(off_def_counts)
    def_while_off_frac = 0.5 * (
        frac(def_off_has_defender_steps[0], def_off_steps[0])
        + frac(def_off_has_defender_steps[1], def_off_steps[1])
    )
    def_while_off_count = avg(def_off_counts)

    vel_toward_ball = 0.5 * (vtb_blue + vtb_red)
    compactness = 0.5 * (compact_blue + compact_red)
    forward_press = 0.5 * (cf_blue + cf_red)

    return {
        "n_passes": n_pass,
        "mean_pass_length": mean_pass_len,
        "n_double_passes": n_double,
        "n_triple_passes": n_triple,
        "n_dribbles": n_dribble,
        "goalie_frac": goalie_frac,
        "goalie_rotations": goalie_rotations,
        "off_while_def_frac": off_while_def_frac,
        "off_while_def_mean_count": off_while_def_count,
        "def_while_off_frac": def_while_off_frac,
        "def_while_off_mean_count": def_while_off_count,
        "velocity_toward_ball": vel_toward_ball,
        "formation_compactness": compactness,
        "mean_forward_press_x": forward_press,
        "ball_x_entropy": ball_x_entropy,
        "num_touches": len(touches),
        "game_length_steps": int(T),
    }


def run_checkpoint(
    ckpt_path: Path,
    players_per_team: int,
    num_envs: int,
    game_length: int,
    seed: int,
    device: str,
    traces_out: Path | None = None,
) -> dict:
    env = make_native_vec_env(
        num_envs=num_envs,
        players_per_team=players_per_team,
        action_mode="discrete",
        game_length=game_length,
        do_team_switch=False,
        render_mode=None,
        log_interval=1,
        seed=seed,
    )

    state = _train.load_checkpoint_state_dict(ckpt_path)
    if "state_dict" in state and "format_version" in state:
        state = state["state_dict"]
    policy = _train.Policy(env).to(device)
    policy.load_state_dict(state, strict=True)
    policy.eval()

    num_players = 2 * players_per_team
    T_alloc = game_length + 2
    positions = np.zeros((num_envs, T_alloc, num_players, 2), dtype=np.float32)
    ball = np.zeros((num_envs, T_alloc, 4), dtype=np.float32)
    actions_log = np.zeros((num_envs, T_alloc, num_players), dtype=np.int32)
    blue_left_per = np.zeros((num_envs,), dtype=bool)

    obs, _ = env.reset(seed=seed)
    for e in range(num_envs):
        st = env.get_state(e)
        positions[e, 0] = st["positions"]
        ball[e, 0] = np.asarray(st["ball"], dtype=np.float32)
        blue_left_per[e] = bool(st["blue_left"])

    with torch.no_grad():
        for t in range(1, T_alloc):
            obs_t = torch.from_numpy(obs).to(device)
            logits, _vals = policy(obs_t)
            acts = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int32)
            # acts shape is (num_envs * num_players,) – reshape per env
            acts_per_env = acts.reshape(num_envs, num_players)
            for e in range(num_envs):
                actions_log[e, t - 1] = acts_per_env[e]
            obs, _rew, term, trunc, _info = env.step(acts)
            for e in range(num_envs):
                st = env.get_state(e)
                positions[e, t] = st["positions"]
                ball[e, t] = np.asarray(st["ball"], dtype=np.float32)
            # If this rolled into a new episode, we still keep collecting
            # (post-reset states are valid; ball-impulse at the reset step
            # will be filtered by the IMPULSE_THRESHOLD since positions
            # teleport and no single closest agent is within TOUCH_RADIUS).

    all_stats = []
    for e in range(num_envs):
        stats = compute_game_stats(
            positions[e], ball[e], actions_log[e], bool(blue_left_per[e]), players_per_team
        )
        all_stats.append(stats)

    keys = list(all_stats[0].keys())
    agg = {}
    for k in keys:
        vals = [s[k] for s in all_stats]
        agg[k] = float(np.mean(vals))
        agg[k + "_std"] = float(np.std(vals))

    agg["num_games"] = num_envs
    agg["game_length"] = int(game_length)
    agg["checkpoint"] = str(ckpt_path)
    agg["epoch"] = int(MODEL_NAME_RE.search(ckpt_path.name).group(1))

    if traces_out is not None:
        traces_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            traces_out,
            positions=positions.astype(np.float32),
            ball=ball.astype(np.float32),
            actions=actions_log.astype(np.int16),
            blue_left=blue_left_per,
            epoch=np.int32(agg["epoch"]),
            game_length=np.int32(game_length),
            players_per_team=np.int32(players_per_team),
        )

    env.close()
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=5, help="take every Nth checkpoint")
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--game-length", type=int, default=400)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-run checkpoints that already have a JSON",
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=None,
        help="also save raw per-env positions/ball/actions as npz per checkpoint",
    )
    parser.add_argument(
        "--traces-keep-last-only",
        action="store_true",
        help="only save traces for the highest-epoch checkpoint (saves disk)",
    )
    args = parser.parse_args()

    ckpts = sorted(
        p for p in args.checkpoint_dir.glob("model_*.pt") if MODEL_NAME_RE.search(p.name)
    )
    ckpts = ckpts[:: args.stride]
    print(f"Processing {len(ckpts)} checkpoints (stride={args.stride}).")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    last_epoch = int(MODEL_NAME_RE.search(ckpts[-1].name).group(1)) if ckpts else -1
    for i, ckpt in enumerate(ckpts):
        epoch = int(MODEL_NAME_RE.search(ckpt.name).group(1))
        out = args.output_dir / f"stats_epoch_{epoch:06d}.json"
        if out.exists() and not args.overwrite:
            print(f"[{i + 1}/{len(ckpts)}] epoch={epoch}  skip (exists)")
            continue

        trace_path = None
        if args.traces_dir is not None:
            if (not args.traces_keep_last_only) or epoch == last_epoch:
                trace_path = args.traces_dir / f"trace_epoch_{epoch:06d}.npz"

        print(f"[{i + 1}/{len(ckpts)}] epoch={epoch}  running...", flush=True)
        stats = run_checkpoint(
            ckpt,
            players_per_team=args.players_per_team,
            num_envs=args.num_envs,
            game_length=args.game_length,
            seed=args.seed + epoch,
            device=args.device,
            traces_out=trace_path,
        )
        with open(out, "w") as f:
            json.dump(stats, f, indent=2)
        print(
            f"    passes={stats['n_passes']:.1f} dribbles={stats['n_dribbles']:.1f} "
            f"goalie={stats['goalie_frac']:.2f} rot={stats['goalie_rotations']:.1f} "
            f"vtb={stats['velocity_toward_ball']:.3f} "
            f"touches={stats['num_touches']:.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
