from __future__ import annotations

import argparse

from puffer_soccer.autotune import (
    DEFAULT_AUTOTUNE_SAMPLE_SECONDS,
    autotune_multiprocessing,
    autotune_native,
    autotune_vecenv,
    format_benchmark_result,
    run_benchmark,
)
from puffer_soccer.vector_env import physical_cpu_count


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default="native",
        choices=["native", "multiprocessing", "auto"],
    )
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
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument(
        "--action-mode",
        type=str,
        default="discrete",
        choices=["discrete", "continuous"],
    )
    args = parser.parse_args()

    player_counts = (
        [args.players_per_team] if args.players_per_team is not None else [1, 5, 11]
    )
    if args.autotune:
        autotune_seconds = args.seconds or DEFAULT_AUTOTUNE_SAMPLE_SECONDS
        best = None
        for players_per_team in player_counts:
            if args.backend == "native":
                outcome = autotune_native(
                    players_per_team,
                    args.action_mode,
                    max_num_envs=args.max_num_envs
                    or max(1, min(256, physical_cpu_count() * 8)),
                    seconds=autotune_seconds,
                    reporter=print,
                )
            elif args.backend == "multiprocessing":
                outcome = autotune_multiprocessing(
                    players_per_team,
                    args.action_mode,
                    max_num_envs=args.max_num_envs
                    or max(1, min(256, physical_cpu_count() * 8)),
                    max_num_shards=args.max_num_shards or physical_cpu_count(),
                    seconds=autotune_seconds,
                    reporter=print,
                )
            else:
                outcome = autotune_vecenv(
                    players_per_team=players_per_team,
                    seconds=autotune_seconds,
                    action_mode=args.action_mode,
                    backend="auto",
                    max_num_envs=args.max_num_envs,
                    max_num_shards=args.max_num_shards,
                    reporter=print,
                )

            print(
                f"BEST\t{outcome.best.backend}\t{players_per_team}v{players_per_team}\t"
                f"{format_benchmark_result(outcome.best)}\t{outcome.selection_reason}"
            )
            candidate = (players_per_team, outcome.best)
            if best is None or outcome.best.sps > best[1].sps:
                best = candidate

        assert best is not None
        print(
            f"OVERALL_BEST\t{best[1].backend}\t{best[0]}v{best[0]}\t"
            f"{format_benchmark_result(best[1])}"
        )
        return

    if args.backend == "auto":
        raise ValueError(
            "manual benchmarking requires --backend native or multiprocessing"
        )

    benchmark_seconds = args.seconds or 10.0

    if args.backend == "native":
        if args.num_envs_list is None:
            for players_per_team in player_counts:
                result = run_benchmark(
                    backend="native",
                    shard_num_envs=args.num_envs,
                    num_shards=1,
                    batch_size=None,
                    players_per_team=players_per_team,
                    seconds=benchmark_seconds,
                    action_mode=args.action_mode,
                )
                print(
                    f"{players_per_team}v{players_per_team}\t{format_benchmark_result(result)}"
                )
            return

        env_counts = parse_int_list(args.num_envs_list)
        best = None
        for players_per_team in player_counts:
            for num_envs in env_counts:
                result = run_benchmark(
                    backend="native",
                    shard_num_envs=num_envs,
                    num_shards=1,
                    batch_size=None,
                    players_per_team=players_per_team,
                    seconds=benchmark_seconds,
                    action_mode=args.action_mode,
                )
                print(
                    f"{players_per_team}v{players_per_team}\t{format_benchmark_result(result)}"
                )
                candidate = (players_per_team, result)
                if best is None or result.sps > best[1].sps:
                    best = candidate
        assert best is not None
        print(f"BEST\tnative\t{best[0]}v{best[0]}\t{format_benchmark_result(best[1])}")
        return

    num_shards = args.num_shards or physical_cpu_count()
    batch_size = args.batch_size or num_shards
    shard_num_envs_values = (
        parse_int_list(args.shard_num_envs_list)
        if args.shard_num_envs_list is not None
        else [args.shard_num_envs]
    )
    num_shard_values = (
        parse_int_list(args.num_shards_list)
        if args.num_shards_list is not None
        else [num_shards]
    )
    batch_size_values = (
        parse_int_list(args.batch_size_list)
        if args.batch_size_list is not None
        else [batch_size]
    )
    best = None
    for players_per_team in player_counts:
        for shard_num_envs in shard_num_envs_values:
            for shards in num_shard_values:
                for batch in batch_size_values:
                    if batch > shards or shards % batch != 0:
                        continue
                    result = run_benchmark(
                        backend="multiprocessing",
                        shard_num_envs=shard_num_envs,
                        num_shards=shards,
                        batch_size=batch,
                        players_per_team=players_per_team,
                        seconds=benchmark_seconds,
                        action_mode=args.action_mode,
                    )
                    print(
                        f"{players_per_team}v{players_per_team}\t{format_benchmark_result(result)}"
                    )
                    candidate = (players_per_team, result)
                    if best is None or result.sps > best[1].sps:
                        best = candidate

    assert best is not None
    print(
        f"BEST\tmultiprocessing\t{best[0]}v{best[0]}\t{format_benchmark_result(best[1])}"
    )


if __name__ == "__main__":
    main()
