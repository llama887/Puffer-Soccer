"""Quick behavioral trace: load a v9 self-play checkpoint and play episodes
while logging per-step ball/player positions and scoring events.

Reports:
  - Goals per episode (mean, total)
  - Mean ball speed (proxy for kick activity)
  - Ball possession fraction by team (proxy: which half the ball is on)
  - Mean distance from every agent to the ball
  - Fraction of steps where any player is within kick range of the ball
  - Team spread (std of team positions)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path("/scratch/fyy2003/repos/Puffer-Soccer/scripts").resolve()))

import importlib.util
script = Path("/scratch/fyy2003/repos/Puffer-Soccer/scripts/train_pufferl.py")
spec = importlib.util.spec_from_file_location("train_pufferl", script)
train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train
spec.loader.exec_module(train)

from puffer_soccer.envs.marl2d import make_puffer_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    state = train.load_checkpoint_state_dict(args.checkpoint)
    if "state_dict" in state and "format_version" in state:
        state = state["state_dict"]
    ck_obs = state.get("net.0.weight", state.get("policy.net.0.weight"))
    expose_stats = ck_obs is not None and ck_obs.shape[1] == (16 + 20 * args.players_per_team)
    env = make_puffer_env(
        players_per_team=args.players_per_team,
        action_mode="discrete",
        game_length=400,
        render_mode=None,
        seed=args.seed,
        expose_stats_in_obs=expose_stats,
    )
    # Auto-detect LSTM checkpoints from state_dict keys (LSTMWrapper prefixes
    # everything with "lstm." or "cell." and keeps inner Policy under
    # "policy.*"). Wrap the base policy if so.
    has_lstm = any(k.startswith("lstm.") or k.startswith("cell.") or k.startswith("policy.") for k in state.keys())
    train._USE_LSTM = has_lstm
    policy = train.build_policy(env).to("cpu")
    policy.load_state_dict(state, strict=True)
    policy.eval()

    total_blue_goals = 0
    total_red_goals = 0
    ball_speed_sum = 0.0
    ball_speed_steps = 0
    blue_half_steps = 0
    red_half_steps = 0
    agent_to_ball_sum = 0.0
    agent_to_ball_count = 0
    kick_range_steps = 0
    total_steps = 0
    blue_team_spread_sum = 0.0
    red_team_spread_sum = 0.0
    kick_range_threshold = 4.0
    # Action count histogram: [NOOP, MOVE_FWD, MOVE_BWD, ROT_L, ROT_R, kick x 8]
    action_counts = np.zeros(13, dtype=np.int64)

    is_lstm = has_lstm
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state0 = env.get_state()
        prev_goals = state0["goals"]
        # Per-episode LSTM hidden state, zero-init each ep
        lstm_state = None
        if is_lstm:
            n_agents = obs.shape[0]
            hidden = int(getattr(policy, "hidden_size", 256))
            lstm_state = {
                "lstm_h": torch.zeros(n_agents, hidden),
                "lstm_c": torch.zeros(n_agents, hidden),
            }
        for step in range(400):
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                if is_lstm:
                    logits, _ = policy.forward_eval(obs_t, lstm_state)
                else:
                    logits, _ = policy.forward(obs_t)
                actions = torch.distributions.Categorical(logits=logits).sample().numpy().astype(np.int32)
            for a in actions:
                if 0 <= a < action_counts.shape[0]:
                    action_counts[a] += 1
            obs, reward, term, trunc, info = env.step(actions)
            s = env.get_state()
            positions = s["positions"]
            bx, by, bvx, bvy = s["ball"]
            goals_blue, goals_red = s["goals"]

            total_blue_goals += goals_blue - prev_goals[0]
            total_red_goals += goals_red - prev_goals[1]
            prev_goals = (goals_blue, goals_red)

            ball_speed_sum += float(np.hypot(bvx, bvy))
            ball_speed_steps += 1

            if s["blue_left"]:
                if bx < 0: blue_half_steps += 1
                else: red_half_steps += 1
            else:
                if bx < 0: red_half_steps += 1
                else: blue_half_steps += 1

            dists = np.sqrt((positions[:, 0] - bx) ** 2 + (positions[:, 1] - by) ** 2)
            agent_to_ball_sum += float(dists.mean())
            agent_to_ball_count += 1
            if dists.min() < kick_range_threshold:
                kick_range_steps += 1

            blue_pos = positions[: args.players_per_team]
            red_pos = positions[args.players_per_team :]
            blue_team_spread_sum += float(blue_pos.std(axis=0).mean())
            red_team_spread_sum += float(red_pos.std(axis=0).mean())
            total_steps += 1

    n_ep = args.episodes
    print(f"=== behavioral trace over {n_ep} episodes ({total_steps} steps total) ===")
    print(f"goals_blue={total_blue_goals}, goals_red={total_red_goals}")
    print(f"mean_goals_per_episode: blue={total_blue_goals/n_ep:.2f}, red={total_red_goals/n_ep:.2f}")
    print(f"mean_ball_speed={ball_speed_sum/max(1,ball_speed_steps):.3f}  (max possible = 5.0)")
    print(f"ball_possession_fraction: blue={blue_half_steps/total_steps:.3f}  red={red_half_steps/total_steps:.3f}")
    print(f"mean_agent_to_ball_dist={agent_to_ball_sum/max(1,agent_to_ball_count):.2f} (base field half-diag ~61)")
    print(f"kick_range_frac (some player within {kick_range_threshold} units of ball): {kick_range_steps/total_steps:.3f}")
    print(f"mean_team_spread: blue={blue_team_spread_sum/total_steps:.2f}  red={red_team_spread_sum/total_steps:.2f}")
    total_actions = max(1, int(action_counts.sum()))
    action_names = ["NOOP", "MOVE_F", "MOVE_B", "ROT_L", "ROT_R", "KICK1", "KICK2", "KICK3", "KICK4", "KICK5", "KICK6", "KICK7", "KICK8"]
    action_fractions = action_counts / total_actions
    print("action distribution:")
    for name, frac in zip(action_names, action_fractions):
        print(f"  {name:7s} {frac:.3f}")
    # Summary groups
    noop_frac = action_fractions[0]
    move_frac = action_fractions[1] + action_fractions[2]
    rot_frac = action_fractions[3] + action_fractions[4]
    kick_frac = action_fractions[5:].sum()
    print(f"grouped: noop={noop_frac:.3f}  move={move_frac:.3f}  rot={rot_frac:.3f}  kick={kick_frac:.3f}")


if __name__ == "__main__":
    main()
