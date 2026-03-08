from __future__ import annotations

import numpy as np

from puffer_soccer.envs.marl2d import make_puffer_env


def main():
    env = make_puffer_env(
        players_per_team=2, action_mode="discrete", render_mode="human"
    )
    env.reset(seed=0)
    done = False
    while not done:
        actions = np.random.randint(0, 9, size=(env.num_agents,), dtype=np.int32)
        _, _, terms, truncs, _ = env.step(actions)
        env.render()
        done = bool(terms.all() or truncs.all())
    env.close()


if __name__ == "__main__":
    main()
