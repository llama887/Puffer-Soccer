from __future__ import annotations

import numpy as np

from puffer_soccer.envs.marl2d import make_parallel_env


def main():
    env = make_parallel_env(players_per_team=2, action_mode="discrete", render_mode="human")
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        actions = {agent: np.random.randint(0, 9) for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        env.render()
        done = all(terms.values()) or all(truncs.values())
    env.close()


if __name__ == "__main__":
    main()
