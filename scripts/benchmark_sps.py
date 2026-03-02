from __future__ import annotations

import argparse
import time

import numpy as np

from puffer_soccer.vector_env import VecEnvConfig, make_soccer_vecenv, physical_cpu_count


def parse_int_list(raw: str) -> list[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def search_candidates(max_value: int) -> list[int]:
    if max_value < 1:
        raise ValueError("max_value must be positive")

    values = {1}
    current = 1
    while current < max_value:
        values.add(current)
        if current < 4:
            current += 1
        elif current < 16:
            current += 2
        elif current < 64:
            current += 4
        else:
            current += 8
    values.add(max_value)
    return sorted(v for v in values if v <= max_value)


def refine_candidates(best_value: int, max_value: int) -> list[int]:
    radius = max(2, best_value // 4)
    start = max(1, best_value - radius)
    end = min(max_value, best_value + radius)
    return list(range(start, end + 1))


def action_cache(cache: int, num_agents: int, action_mode: str) -> np.ndarray:
    if action_mode == "discrete":
        return np.random.randint(0, 9, size=(cache, num_agents), dtype=np.int32)
    return np.random.uniform(-1, 1, size=(cache, num_agents, 2)).astype(np.float32)


def run_benchmark(
    *,
    backend: str,
    shard_num_envs: int,
    num_shards: int,
    batch_size: int | None,
    players_per_team: int,
    seconds: int,
    action_mode: str,
) -> int:
    if backend != "native":
        if batch_size is None:
            raise ValueError("batch_size must be set for multiprocessing benchmarks")
        if batch_size > num_shards or num_shards % batch_size != 0:
            raise ValueError("for zero_copy multiprocessing, num_shards must be divisible by batch_size")

    vecenv = make_soccer_vecenv(
        players_per_team=players_per_team,
        action_mode=action_mode,
        game_length=400,
        render_mode=None,
        seed=0,
        vec=VecEnvConfig(
            backend=backend,
            shard_num_envs=shard_num_envs,
            num_shards=num_shards,
            num_workers=None if backend == "native" else num_shards,
            batch_size=batch_size if backend != "native" else None,
        ),
    )

    try:
        if backend == "native":
            vecenv.reset(seed=0)
            cached_actions = action_cache(1024, vecenv.num_agents, action_mode)
            steps = 0
            idx = 0
            start = time.time()
            while time.time() - start < seconds:
                vecenv.step(cached_actions[idx % 1024])
                idx += 1
                steps += vecenv.num_agents
            elapsed = max(time.time() - start, 1e-6)
            return int(steps / elapsed)

        vecenv.async_reset(seed=0)
        _, _, _, _, _, _, masks = vecenv.recv()
        batch_agents = int(np.asarray(masks).size)
        cached_actions = action_cache(1024, batch_agents, action_mode)
        steps = 0
        idx = 0
        start = time.time()
        while time.time() - start < seconds:
            vecenv.send(cached_actions[idx % 1024])
            _, _, _, _, _, _, masks = vecenv.recv()
            idx += 1
            steps += int(np.asarray(masks).sum())
        elapsed = max(time.time() - start, 1e-6)
        return int(steps / elapsed)
    finally:
        vecenv.close()


def autotune_native(players_per_team: int, seconds: int, action_mode: str, max_num_envs: int) -> tuple[int, int]:
    coarse = []
    for shard_num_envs in search_candidates(max_num_envs):
        sps = run_benchmark(
            backend="native",
            shard_num_envs=shard_num_envs,
            num_shards=1,
            batch_size=None,
            players_per_team=players_per_team,
            seconds=seconds,
            action_mode=action_mode,
        )
        coarse.append((shard_num_envs, sps))
        print(f"AUTOTUNE\tnative-coarse\t{players_per_team}v{players_per_team}\t{shard_num_envs} envs\t{sps} SPS")

    best_num_envs, best_sps = max(coarse, key=lambda item: item[1])
    fine = []
    for shard_num_envs in refine_candidates(best_num_envs, max_num_envs):
        if any(existing == shard_num_envs for existing, _ in coarse):
            continue
        sps = run_benchmark(
            backend="native",
            shard_num_envs=shard_num_envs,
            num_shards=1,
            batch_size=None,
            players_per_team=players_per_team,
            seconds=seconds,
            action_mode=action_mode,
        )
        fine.append((shard_num_envs, sps))
        print(f"AUTOTUNE\tnative-fine\t{players_per_team}v{players_per_team}\t{shard_num_envs} envs\t{sps} SPS")

    if fine:
        best_num_envs, best_sps = max(coarse + fine, key=lambda item: item[1])
    return best_num_envs, best_sps


def batch_candidates(num_shards: int) -> list[int]:
    values = set()
    for candidate in (1, 2, 4, num_shards, max(1, num_shards // 2), max(1, num_shards // 4)):
        if candidate <= num_shards and num_shards % candidate == 0:
            values.add(candidate)
    return sorted(values)


def autotune_multiprocessing(
    players_per_team: int,
    seconds: int,
    action_mode: str,
    max_shard_num_envs: int,
    max_num_shards: int,
) -> tuple[int, int, int]:
    coarse_results: list[tuple[int, int, int, int]] = []
    for shard_num_envs in search_candidates(max_shard_num_envs):
        for num_shards in search_candidates(max_num_shards):
            for batch_size in batch_candidates(num_shards):
                sps = run_benchmark(
                    backend="multiprocessing",
                    shard_num_envs=shard_num_envs,
                    num_shards=num_shards,
                    batch_size=batch_size,
                    players_per_team=players_per_team,
                    seconds=seconds,
                    action_mode=action_mode,
                )
                coarse_results.append((shard_num_envs, num_shards, batch_size, sps))
                print(
                    "AUTOTUNE\tmp-coarse\t"
                    f"{players_per_team}v{players_per_team}\t"
                    f"shard_envs={shard_num_envs}\tshards={num_shards}\tbatch={batch_size}\t{sps} SPS"
                )

    best_shard_num_envs, best_num_shards, best_batch_size, best_sps = max(coarse_results, key=lambda item: item[3])

    fine_results: list[tuple[int, int, int, int]] = []
    for shard_num_envs in refine_candidates(best_shard_num_envs, max_shard_num_envs):
        for num_shards in refine_candidates(best_num_shards, max_num_shards):
            for batch_size in batch_candidates(num_shards):
                if any(
                    existing_shard_envs == shard_num_envs
                    and existing_shards == num_shards
                    and existing_batch == batch_size
                    for existing_shard_envs, existing_shards, existing_batch, _ in coarse_results
                ):
                    continue
                sps = run_benchmark(
                    backend="multiprocessing",
                    shard_num_envs=shard_num_envs,
                    num_shards=num_shards,
                    batch_size=batch_size,
                    players_per_team=players_per_team,
                    seconds=seconds,
                    action_mode=action_mode,
                )
                fine_results.append((shard_num_envs, num_shards, batch_size, sps))
                print(
                    "AUTOTUNE\tmp-fine\t"
                    f"{players_per_team}v{players_per_team}\t"
                    f"shard_envs={shard_num_envs}\tshards={num_shards}\tbatch={batch_size}\t{sps} SPS"
                )

    if fine_results:
        best_shard_num_envs, best_num_shards, best_batch_size, best_sps = max(
            coarse_results + fine_results,
            key=lambda item: item[3],
        )
    return best_shard_num_envs, best_num_shards, best_batch_size, best_sps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="native", choices=["native", "multiprocessing"])
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-envs-list", type=str, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--num-shards-list", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-size-list", type=str, default=None)
    parser.add_argument("--shard-num-envs", type=int, default=2)
    parser.add_argument("--shard-num-envs-list", type=str, default=None)
    parser.add_argument("--players-per-team", type=int, default=None)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--max-num-envs", type=int, default=None)
    parser.add_argument("--max-num-shards", type=int, default=None)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--action-mode", type=str, default="discrete", choices=["discrete", "continuous"])
    args = parser.parse_args()

    player_counts = [args.players_per_team] if args.players_per_team is not None else [1, 5, 11]
    if args.autotune:
        best = None
        if args.backend == "native":
            max_num_envs = args.max_num_envs or max(1, min(256, physical_cpu_count() * 8))
            for players_per_team in player_counts:
                num_envs, sps = autotune_native(players_per_team, args.seconds, args.action_mode, max_num_envs)
                print(f"BEST\tnative\t{players_per_team}v{players_per_team}\tenvs={num_envs}\t{sps} SPS")
                candidate = (players_per_team, num_envs, 1, sps)
                if best is None or sps > best[3]:
                    best = candidate
            assert best is not None
            print(f"OVERALL_BEST\tnative\t{best[0]}v{best[0]}\tenvs={best[1]}\t{best[3]} SPS")
            return

        max_shard_num_envs = args.max_num_envs or 256
        max_num_shards = args.max_num_shards or physical_cpu_count()
        for players_per_team in player_counts:
            shard_num_envs, num_shards, batch_size, sps = autotune_multiprocessing(
                players_per_team,
                args.seconds,
                args.action_mode,
                max_shard_num_envs=max_shard_num_envs,
                max_num_shards=max_num_shards,
            )
            print(
                f"BEST\tmultiprocessing\t{players_per_team}v{players_per_team}\t"
                f"shard_envs={shard_num_envs}\tshards={num_shards}\tbatch={batch_size}\t{sps} SPS"
            )
            candidate = (players_per_team, shard_num_envs, num_shards, batch_size, sps)
            if best is None or sps > best[4]:
                best = candidate

        assert best is not None
        print(
            f"OVERALL_BEST\tmultiprocessing\t{best[0]}v{best[0]}\t"
            f"shard_envs={best[1]}\tshards={best[2]}\tbatch={best[3]}\t{best[4]} SPS"
        )
        return

    if args.backend == "native":
        if args.num_envs_list is None:
            for players_per_team in player_counts:
                sps = run_benchmark(
                    backend="native",
                    shard_num_envs=args.num_envs,
                    num_shards=1,
                    batch_size=None,
                    players_per_team=players_per_team,
                    seconds=args.seconds,
                    action_mode=args.action_mode,
                )
                print(f"{players_per_team}v{players_per_team}: {sps} SPS @ {args.num_envs} envs")
            return

        env_counts = parse_int_list(args.num_envs_list)
        best = None
        for players_per_team in player_counts:
            for num_envs in env_counts:
                sps = run_benchmark(
                    backend="native",
                    shard_num_envs=num_envs,
                    num_shards=1,
                    batch_size=None,
                    players_per_team=players_per_team,
                    seconds=args.seconds,
                    action_mode=args.action_mode,
                )
                print(f"{players_per_team}v{players_per_team}\t{num_envs} envs\t{sps} SPS")
                candidate = (players_per_team, num_envs, sps)
                if best is None or sps > best[2]:
                    best = candidate
        assert best is not None
        print(f"BEST\tnative\t{best[0]}v{best[0]}\tenvs={best[1]}\t{best[2]} SPS")
        return

    num_shards = args.num_shards or physical_cpu_count()
    batch_size = args.batch_size or num_shards
    shard_num_envs_values = (
        parse_int_list(args.shard_num_envs_list) if args.shard_num_envs_list is not None else [args.shard_num_envs]
    )
    num_shard_values = parse_int_list(args.num_shards_list) if args.num_shards_list is not None else [num_shards]
    batch_size_values = parse_int_list(args.batch_size_list) if args.batch_size_list is not None else [batch_size]
    best = None
    for players_per_team in player_counts:
        for shard_num_envs in shard_num_envs_values:
            for shards in num_shard_values:
                for batch in batch_size_values:
                    if batch > shards or shards % batch != 0:
                        continue
                    sps = run_benchmark(
                        backend="multiprocessing",
                        shard_num_envs=shard_num_envs,
                        num_shards=shards,
                        batch_size=batch,
                        players_per_team=players_per_team,
                        seconds=args.seconds,
                        action_mode=args.action_mode,
                    )
                    print(
                        f"{players_per_team}v{players_per_team}\t"
                        f"shard_envs={shard_num_envs}\tshards={shards}\tbatch={batch}\t{sps} SPS"
                    )
                    candidate = (players_per_team, shard_num_envs, shards, batch, sps)
                    if best is None or sps > best[4]:
                        best = candidate

    assert best is not None
    print(
        f"BEST\tmultiprocessing\t{best[0]}v{best[0]}\t"
        f"shard_envs={best[1]}\tshards={best[2]}\tbatch={best[3]}\t{best[4]} SPS"
    )


if __name__ == "__main__":
    main()
