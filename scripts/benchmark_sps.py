from __future__ import annotations

import argparse
import time

import numpy as np

from puffer_soccer.envs.marl2d import make_puffer_env


def run(num_envs: int, players_per_team: int, seconds: int, action_mode: str) -> int:
    env = make_puffer_env(
        num_envs=num_envs,
        players_per_team=players_per_team,
        action_mode=action_mode,
        game_length=400,
    )
    env.reset(seed=0)

    steps = 0
    cache = 1024
    if action_mode == "discrete":
        actions = np.random.randint(0, 9, size=(cache, env.num_agents), dtype=np.int32)
    else:
        actions = np.random.uniform(-1, 1, size=(cache, env.num_agents, 2)).astype(np.float32)

    i = 0
    start = time.time()
    while time.time() - start < seconds:
        env.step(actions[i % cache])
        i += 1
        steps += env.num_agents

    env.close()
    return int(steps / max(time.time() - start, 1e-6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--action-mode", type=str, default="discrete", choices=["discrete", "continuous"])
    args = parser.parse_args()

    for n in (1, 5, 11):
        sps = run(args.num_envs, n, args.seconds, args.action_mode)
        print(f"{n}v{n}: {sps} SPS")


if __name__ == "__main__":
    main()
