from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import psutil
import pufferlib.vector

from puffer_soccer.envs.marl2d import make_native_vec_env, make_puffer_env


@dataclass(frozen=True)
class VecEnvConfig:
    backend: str = "multiprocessing"
    shard_num_envs: int = 2
    num_shards: int = 1
    num_workers: int | None = None
    batch_size: int | None = None
    zero_copy: bool = True
    overwork: bool = False


def physical_cpu_count() -> int:
    return max(1, psutil.cpu_count(logical=False) or 1)


def logical_cpu_count() -> int:
    return max(1, psutil.cpu_count(logical=True) or 1)


def make_sharded_puffer_env(
    players_per_team: int,
    action_mode: str,
    game_length: int,
    do_team_switch: bool = False,
    render_mode: str | None = None,
    log_interval: int = 128,
    buf=None,
    seed: int = 0,
):
    return make_puffer_env(
        players_per_team=players_per_team,
        action_mode=action_mode,
        game_length=game_length,
        do_team_switch=do_team_switch,
        render_mode=render_mode,
        log_interval=log_interval,
        buf=buf,
        seed=seed,
    )


def make_soccer_vecenv(
    *,
    players_per_team: int,
    action_mode: str,
    game_length: int,
    do_team_switch: bool = False,
    render_mode: str | None,
    seed: int,
    vec: VecEnvConfig,
    log_interval: int = 128,
):
    if vec.backend == "native":
        return make_native_vec_env(
            num_envs=total_sim_envs(vec),
            players_per_team=players_per_team,
            action_mode=action_mode,
            game_length=game_length,
            do_team_switch=do_team_switch,
            render_mode=render_mode,
            log_interval=log_interval,
            seed=seed,
        )

    env_creator = partial(
        make_sharded_puffer_env,
        players_per_team=players_per_team,
        action_mode=action_mode,
        game_length=game_length,
        do_team_switch=do_team_switch,
        render_mode=render_mode,
        log_interval=log_interval,
    )

    backend_cls = {
        "serial": pufferlib.vector.Serial,
        "multiprocessing": pufferlib.vector.Multiprocessing,
    }[vec.backend]
    return pufferlib.vector.make(
        env_creator,
        backend=backend_cls,
        num_envs=total_sim_envs(vec),
        num_workers=vec.num_workers,
        batch_size=vec.batch_size,
        zero_copy=vec.zero_copy,
        overwork=vec.overwork,
        seed=seed,
    )


def total_sim_envs(vec: VecEnvConfig) -> int:
    if vec.backend == "native":
        return vec.shard_num_envs
    return vec.shard_num_envs * vec.num_shards
