from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from puffer_soccer.utilization import UtilizationMonitor

if TYPE_CHECKING:
    from puffer_soccer.vector_env import VecEnvConfig


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    shard_num_envs: int
    num_shards: int
    batch_size: int | None
    players_per_team: int
    action_mode: str
    sps: int
    cpu_avg: float | None
    cpu_peak: float | None

    @property
    def total_envs(self) -> int:
        if self.backend == "native":
            return self.shard_num_envs
        return self.shard_num_envs * self.num_shards


@dataclass(frozen=True)
class SearchConfig:
    backend: str
    shard_num_envs: int
    num_shards: int
    batch_size: int | None


@dataclass(frozen=True)
class AutotuneOutcome:
    best: BenchmarkResult
    best_saturated: BenchmarkResult | None
    all_results: tuple[BenchmarkResult, ...]
    selection_reason: str


DEFAULT_AUTOTUNE_SAMPLE_SECONDS = 0.75
DEFAULT_CPU_AVG_TARGET = 97.0
DEFAULT_CPU_PEAK_TARGET = 99.0


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


def batch_candidates(num_shards: int, shard_num_envs: int) -> list[int]:
    """Return valid multiprocessing batch sizes measured in environments.

    PufferLib's multiprocessing vectorizer groups whole worker payloads into a batch. In
    this project each worker owns `shard_num_envs` logical environments, so a valid batch
    size must be an integer number of workers times that shard size. Generating raw worker
    counts here keeps the search focused on a few simple, high-value options while still
    guaranteeing that every returned batch size is accepted by the runtime.

    This helper is separate from the main search loop because the batch-size constraint is
    easy to get subtly wrong: the value must divide the total number of environments and it
    must also be divisible by the number of environments assigned to each worker. Encoding
    those rules once prevents the autotuner from repeatedly probing impossible layouts.
    """
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if shard_num_envs < 1:
        raise ValueError("shard_num_envs must be positive")

    worker_batch_sizes = set()
    for candidate in (
        1,
        2,
        4,
        num_shards,
        max(1, num_shards // 2),
        max(1, num_shards // 4),
    ):
        if candidate <= num_shards and num_shards % candidate == 0:
            worker_batch_sizes.add(candidate)

    return sorted(worker_count * shard_num_envs for worker_count in worker_batch_sizes)


def action_cache(cache: int, num_agents: int, action_mode: str) -> np.ndarray:
    if action_mode == "discrete":
        return np.random.randint(0, 9, size=(cache, num_agents), dtype=np.int32)
    return np.random.uniform(-1, 1, size=(cache, num_agents, 2)).astype(np.float32)


def is_cpu_saturated(
    result: BenchmarkResult,
    cpu_avg_target: float = DEFAULT_CPU_AVG_TARGET,
    cpu_peak_target: float = DEFAULT_CPU_PEAK_TARGET,
) -> bool:
    if result.cpu_avg is not None and result.cpu_avg >= cpu_avg_target:
        return True
    if result.cpu_peak is not None and result.cpu_peak >= cpu_peak_target:
        return True
    return False


def choose_best_result(
    results: Iterable[BenchmarkResult],
    cpu_avg_target: float = DEFAULT_CPU_AVG_TARGET,
    cpu_peak_target: float = DEFAULT_CPU_PEAK_TARGET,
) -> AutotuneOutcome:
    collected = tuple(results)
    if not collected:
        raise ValueError("expected at least one benchmark result")

    saturated = [
        result
        for result in collected
        if is_cpu_saturated(
            result, cpu_avg_target=cpu_avg_target, cpu_peak_target=cpu_peak_target
        )
    ]
    if saturated:
        best = max(saturated, key=lambda item: item.sps)
        return AutotuneOutcome(
            best=best,
            best_saturated=best,
            all_results=collected,
            selection_reason=(
                f"selected highest SPS among {len(saturated)} CPU-saturated candidates"
            ),
        )

    best = max(collected, key=lambda item: item.sps)
    return AutotuneOutcome(
        best=best,
        best_saturated=None,
        all_results=collected,
        selection_reason=(
            "no candidate reached near-100% CPU usage; selected highest SPS overall"
        ),
    )


def format_benchmark_result(result: BenchmarkResult) -> str:
    cpu_avg = "n/a" if result.cpu_avg is None else f"{result.cpu_avg:.1f}%"
    cpu_peak = "n/a" if result.cpu_peak is None else f"{result.cpu_peak:.1f}%"
    if result.backend == "native":
        return f"envs={result.shard_num_envs}\t{sps_label(result)}\tcpu_avg={cpu_avg}\tcpu_peak={cpu_peak}"
    return (
        f"total_envs={result.total_envs}\tshard_envs={result.shard_num_envs}\t"
        f"shards={result.num_shards}\tbatch={result.batch_size}\t{sps_label(result)}\t"
        f"cpu_avg={cpu_avg}\tcpu_peak={cpu_peak}"
    )


def sps_label(result: BenchmarkResult) -> str:
    return f"{result.sps} SPS"


def run_benchmark(
    *,
    backend: str,
    shard_num_envs: int,
    num_shards: int,
    batch_size: int | None,
    players_per_team: int,
    seconds: float,
    action_mode: str,
    sample_interval_s: float = 0.25,
) -> BenchmarkResult:
    from puffer_soccer.vector_env import VecEnvConfig, make_soccer_vecenv

    if backend != "native":
        total_envs = shard_num_envs * num_shards
        if batch_size is None:
            raise ValueError("batch_size must be set for multiprocessing benchmarks")
        if batch_size > total_envs or total_envs % batch_size != 0:
            raise ValueError(
                "for zero_copy multiprocessing, total envs must be divisible by batch_size"
            )
        if batch_size % shard_num_envs != 0:
            raise ValueError("batch_size must be divisible by (num_envs / num_workers)")

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
    monitor = UtilizationMonitor(sample_interval_s=sample_interval_s)
    monitor.start()

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
        else:
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
    finally:
        util = monitor.stop()
        vecenv.close()

    return BenchmarkResult(
        backend=backend,
        shard_num_envs=shard_num_envs,
        num_shards=num_shards,
        batch_size=batch_size,
        players_per_team=players_per_team,
        action_mode=action_mode,
        sps=int(steps / elapsed),
        cpu_avg=util["cpu_avg"],
        cpu_peak=util["cpu_peak"],
    )


def native_search_space(max_num_envs: int) -> list[tuple[int, list[SearchConfig]]]:
    return [
        (
            num_envs,
            [
                SearchConfig(
                    backend="native",
                    shard_num_envs=num_envs,
                    num_shards=1,
                    batch_size=None,
                )
            ],
        )
        for num_envs in search_candidates(max_num_envs)
    ]


def multiprocessing_configs_for_total_envs(
    total_envs: int, max_num_shards: int
) -> list[SearchConfig]:
    configs: list[SearchConfig] = []
    for num_shards in search_candidates(min(total_envs, max_num_shards)):
        if total_envs % num_shards != 0:
            continue
        shard_num_envs = total_envs // num_shards
        for batch_size in batch_candidates(num_shards, shard_num_envs):
            configs.append(
                SearchConfig(
                    backend="multiprocessing",
                    shard_num_envs=shard_num_envs,
                    num_shards=num_shards,
                    batch_size=batch_size,
                )
            )
    return configs


def _report(reporter: Callable[[str], None] | None, message: str) -> None:
    if reporter is not None:
        reporter(message)


def should_stop_autotune(
    *,
    level_best_sps: float,
    best_sps_so_far: float,
    saturated_seen: bool,
    plateau_count: int,
    plateau_tolerance: float,
    plateau_patience: int,
) -> tuple[float, int, bool]:
    if level_best_sps > best_sps_so_far * (1.0 + plateau_tolerance):
        return level_best_sps, 0, False
    if saturated_seen:
        plateau_count += 1
        return best_sps_so_far, plateau_count, plateau_count >= plateau_patience
    return best_sps_so_far, plateau_count, False


def _evaluate_search_space(
    *,
    backend: str,
    players_per_team: int,
    seconds: float,
    action_mode: str,
    coarse_levels: list[tuple[int, list[SearchConfig]]],
    fine_level_builder: Callable[[int], list[tuple[int, list[SearchConfig]]]],
    reporter: Callable[[str], None] | None,
    cpu_avg_target: float,
    cpu_peak_target: float,
    plateau_patience: int = 2,
    plateau_tolerance: float = 0.01,
) -> AutotuneOutcome:
    results: list[BenchmarkResult] = []
    evaluated_keys: set[tuple[str, int, int, int | None]] = set()

    def evaluate_levels(
        stage: str, levels: list[tuple[int, list[SearchConfig]]], allow_early_stop: bool
    ) -> None:
        best_sps_so_far = 0.0
        plateau_count = 0
        saturated_seen = False
        for total_envs, configs in levels:
            level_results: list[BenchmarkResult] = []
            for config in configs:
                key = (
                    config.backend,
                    config.shard_num_envs,
                    config.num_shards,
                    config.batch_size,
                )
                if key in evaluated_keys:
                    continue
                evaluated_keys.add(key)
                try:
                    result = run_benchmark(
                        backend=config.backend,
                        shard_num_envs=config.shard_num_envs,
                        num_shards=config.num_shards,
                        batch_size=config.batch_size,
                        players_per_team=players_per_team,
                        seconds=seconds,
                        action_mode=action_mode,
                    )
                except Exception as exc:
                    _report(
                        reporter,
                        "AUTOTUNE\t"
                        f"{backend}-{stage}\t{players_per_team}v{players_per_team}\t"
                        f"skip backend={config.backend}\t"
                        f"shard_envs={config.shard_num_envs}\t"
                        f"shards={config.num_shards}\t"
                        f"batch={config.batch_size}\t{exc}",
                    )
                    continue
                results.append(result)
                level_results.append(result)
                _report(
                    reporter,
                    f"AUTOTUNE\t{backend}-{stage}\t{players_per_team}v{players_per_team}\t{format_benchmark_result(result)}",
                )

            if not allow_early_stop or not level_results:
                continue

            if any(
                is_cpu_saturated(
                    result,
                    cpu_avg_target=cpu_avg_target,
                    cpu_peak_target=cpu_peak_target,
                )
                for result in level_results
            ):
                saturated_seen = True

            level_best_sps = max(result.sps for result in level_results)
            best_sps_so_far, plateau_count, should_stop = should_stop_autotune(
                level_best_sps=level_best_sps,
                best_sps_so_far=best_sps_so_far,
                saturated_seen=saturated_seen,
                plateau_count=plateau_count,
                plateau_tolerance=plateau_tolerance,
                plateau_patience=plateau_patience,
            )
            if should_stop:
                _report(
                    reporter,
                    f"AUTOTUNE\t{backend}-{stage}\t{players_per_team}v{players_per_team}\tstop total_envs={total_envs}",
                )
                break

    evaluate_levels("coarse", coarse_levels, allow_early_stop=True)
    if not results:
        raise RuntimeError(f"no valid {backend} vector configurations completed")
    coarse_choice = choose_best_result(
        results,
        cpu_avg_target=cpu_avg_target,
        cpu_peak_target=cpu_peak_target,
    )
    evaluate_levels(
        "fine",
        fine_level_builder(coarse_choice.best.total_envs),
        allow_early_stop=False,
    )
    return choose_best_result(
        results,
        cpu_avg_target=cpu_avg_target,
        cpu_peak_target=cpu_peak_target,
    )


def autotune_native(
    players_per_team: int,
    action_mode: str,
    max_num_envs: int,
    seconds: float = DEFAULT_AUTOTUNE_SAMPLE_SECONDS,
    reporter: Callable[[str], None] | None = None,
    cpu_avg_target: float = DEFAULT_CPU_AVG_TARGET,
    cpu_peak_target: float = DEFAULT_CPU_PEAK_TARGET,
) -> AutotuneOutcome:
    coarse_levels = native_search_space(max_num_envs)

    def fine_levels(best_total_envs: int) -> list[tuple[int, list[SearchConfig]]]:
        return [
            (
                num_envs,
                [
                    SearchConfig(
                        backend="native",
                        shard_num_envs=num_envs,
                        num_shards=1,
                        batch_size=None,
                    )
                ],
            )
            for num_envs in refine_candidates(best_total_envs, max_num_envs)
        ]

    return _evaluate_search_space(
        backend="native",
        players_per_team=players_per_team,
        seconds=seconds,
        action_mode=action_mode,
        coarse_levels=coarse_levels,
        fine_level_builder=fine_levels,
        reporter=reporter,
        cpu_avg_target=cpu_avg_target,
        cpu_peak_target=cpu_peak_target,
    )


def autotune_multiprocessing(
    players_per_team: int,
    action_mode: str,
    max_num_envs: int,
    max_num_shards: int,
    seconds: float = DEFAULT_AUTOTUNE_SAMPLE_SECONDS,
    reporter: Callable[[str], None] | None = None,
    cpu_avg_target: float = DEFAULT_CPU_AVG_TARGET,
    cpu_peak_target: float = DEFAULT_CPU_PEAK_TARGET,
) -> AutotuneOutcome:
    coarse_levels = [
        (total_envs, multiprocessing_configs_for_total_envs(total_envs, max_num_shards))
        for total_envs in search_candidates(max_num_envs)
    ]

    def fine_levels(best_total_envs: int) -> list[tuple[int, list[SearchConfig]]]:
        return [
            (
                total_envs,
                multiprocessing_configs_for_total_envs(total_envs, max_num_shards),
            )
            for total_envs in refine_candidates(best_total_envs, max_num_envs)
        ]

    return _evaluate_search_space(
        backend="multiprocessing",
        players_per_team=players_per_team,
        seconds=seconds,
        action_mode=action_mode,
        coarse_levels=coarse_levels,
        fine_level_builder=fine_levels,
        reporter=reporter,
        cpu_avg_target=cpu_avg_target,
        cpu_peak_target=cpu_peak_target,
    )


def autotune_vecenv(
    *,
    players_per_team: int,
    seconds: float = DEFAULT_AUTOTUNE_SAMPLE_SECONDS,
    action_mode: str,
    backend: str = "auto",
    max_num_envs: int | None = None,
    max_num_shards: int | None = None,
    reporter: Callable[[str], None] | None = None,
    cpu_avg_target: float = DEFAULT_CPU_AVG_TARGET,
    cpu_peak_target: float = DEFAULT_CPU_PEAK_TARGET,
) -> AutotuneOutcome:
    from puffer_soccer.vector_env import physical_cpu_count

    resolved_max_num_envs = max_num_envs or max(1, min(256, physical_cpu_count() * 8))
    resolved_max_num_shards = max_num_shards or physical_cpu_count()

    if backend == "native":
        return autotune_native(
            players_per_team,
            action_mode,
            max_num_envs=resolved_max_num_envs,
            seconds=seconds,
            reporter=reporter,
            cpu_avg_target=cpu_avg_target,
            cpu_peak_target=cpu_peak_target,
        )
    if backend == "multiprocessing":
        return autotune_multiprocessing(
            players_per_team,
            action_mode,
            max_num_envs=resolved_max_num_envs,
            max_num_shards=resolved_max_num_shards,
            seconds=seconds,
            reporter=reporter,
            cpu_avg_target=cpu_avg_target,
            cpu_peak_target=cpu_peak_target,
        )
    if backend != "auto":
        raise ValueError(f"unsupported backend: {backend}")

    outcomes: list[AutotuneOutcome] = []
    backend_failures: list[str] = []

    try:
        outcomes.append(
            autotune_native(
                players_per_team,
                action_mode,
                max_num_envs=resolved_max_num_envs,
                seconds=seconds,
                reporter=reporter,
                cpu_avg_target=cpu_avg_target,
                cpu_peak_target=cpu_peak_target,
            )
        )
    except RuntimeError as exc:
        backend_failures.append(f"native: {exc}")
        _report(
            reporter,
            f"AUTOTUNE\tauto\t{players_per_team}v{players_per_team}\tskip native\t{exc}",
        )

    try:
        outcomes.append(
            autotune_multiprocessing(
                players_per_team,
                action_mode,
                max_num_envs=resolved_max_num_envs,
                max_num_shards=resolved_max_num_shards,
                seconds=seconds,
                reporter=reporter,
                cpu_avg_target=cpu_avg_target,
                cpu_peak_target=cpu_peak_target,
            )
        )
    except RuntimeError as exc:
        backend_failures.append(f"multiprocessing: {exc}")
        _report(
            reporter,
            f"AUTOTUNE\tauto\t{players_per_team}v{players_per_team}\tskip multiprocessing\t{exc}",
        )

    if not outcomes:
        raise RuntimeError(
            "; ".join(backend_failures) or "no vector backends completed"
        )

    combined = choose_best_result(
        [outcome.best for outcome in outcomes],
        cpu_avg_target=cpu_avg_target,
        cpu_peak_target=cpu_peak_target,
    )
    return AutotuneOutcome(
        best=combined.best,
        best_saturated=combined.best_saturated,
        all_results=tuple(
            result for outcome in outcomes for result in outcome.all_results
        ),
        selection_reason=(
            combined.selection_reason
            if not backend_failures
            else f"{combined.selection_reason}; skipped {'; '.join(backend_failures)}"
        ),
    )


def vec_config_from_benchmark(result: BenchmarkResult) -> VecEnvConfig:
    from puffer_soccer.vector_env import VecEnvConfig

    return VecEnvConfig(
        backend=result.backend,
        shard_num_envs=result.shard_num_envs,
        num_shards=result.num_shards,
        num_workers=None if result.backend == "native" else result.num_shards,
        batch_size=result.batch_size,
    )
