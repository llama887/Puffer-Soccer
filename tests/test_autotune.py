from puffer_soccer.autotune import (
    BenchmarkResult,
    choose_best_result,
    multiprocessing_configs_for_total_envs,
    should_stop_autotune,
)


def make_result(
    *,
    backend: str = "native",
    shard_num_envs: int = 1,
    num_shards: int = 1,
    batch_size: int | None = None,
    sps: int,
    cpu_avg: float | None,
    cpu_peak: float | None,
) -> BenchmarkResult:
    return BenchmarkResult(
        backend=backend,
        shard_num_envs=shard_num_envs,
        num_shards=num_shards,
        batch_size=batch_size,
        players_per_team=5,
        action_mode="discrete",
        sps=sps,
        cpu_avg=cpu_avg,
        cpu_peak=cpu_peak,
    )


def test_choose_best_result_prefers_cpu_saturated_candidate() -> None:
    unsaturated_faster = make_result(sps=1200, cpu_avg=72.0, cpu_peak=80.0)
    saturated = make_result(sps=1100, cpu_avg=98.0, cpu_peak=99.0)

    outcome = choose_best_result([unsaturated_faster, saturated])

    assert outcome.best == saturated
    assert outcome.best_saturated == saturated
    assert "CPU-saturated" in outcome.selection_reason


def test_choose_best_result_falls_back_to_highest_sps() -> None:
    slower = make_result(sps=800, cpu_avg=60.0, cpu_peak=70.0)
    faster = make_result(sps=1000, cpu_avg=75.0, cpu_peak=89.0)

    outcome = choose_best_result([slower, faster])

    assert outcome.best == faster
    assert outcome.best_saturated is None
    assert "highest SPS overall" in outcome.selection_reason


def test_multiprocessing_configs_for_total_envs_respect_constraints() -> None:
    configs = multiprocessing_configs_for_total_envs(total_envs=12, max_num_shards=6)

    assert configs
    for config in configs:
        assert config.num_shards <= 6
        assert 12 % config.num_shards == 0
        assert config.shard_num_envs * config.num_shards == 12
        assert config.batch_size is not None
        assert 12 % config.batch_size == 0
        assert config.batch_size % config.shard_num_envs == 0


def test_multiprocessing_configs_avoid_invalid_single_worker_batch_sizes() -> None:
    configs = multiprocessing_configs_for_total_envs(total_envs=8, max_num_shards=4)

    observed = {
        (config.shard_num_envs, config.num_shards, config.batch_size)
        for config in configs
    }

    assert (8, 1, 1) not in observed
    assert (4, 2, 1) not in observed
    assert (4, 2, 2) not in observed
    assert (8, 1, 8) in observed
    assert (4, 2, 4) in observed


def test_should_stop_autotune_waits_for_saturation_before_plateau_stop() -> None:
    best_sps, plateau_count, should_stop = should_stop_autotune(
        level_best_sps=1000.0,
        best_sps_so_far=1010.0,
        saturated_seen=False,
        plateau_count=1,
        plateau_tolerance=0.01,
        plateau_patience=2,
    )

    assert best_sps == 1010.0
    assert plateau_count == 1
    assert should_stop is False


def test_should_stop_autotune_stops_after_saturated_plateau() -> None:
    best_sps, plateau_count, should_stop = should_stop_autotune(
        level_best_sps=1000.0,
        best_sps_so_far=1010.0,
        saturated_seen=True,
        plateau_count=1,
        plateau_tolerance=0.01,
        plateau_patience=2,
    )

    assert best_sps == 1010.0
    assert plateau_count == 2
    assert should_stop is True
