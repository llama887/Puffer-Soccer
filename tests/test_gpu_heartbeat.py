"""Tests for the GPU heartbeat control policy.

These tests focus on the pure burst-selection helper rather than the full infinite runtime
loop. That keeps the checks deterministic and runnable on machines without CUDA while still
covering the policy choice that decides whether the heartbeat will undershoot, overshoot, or
roughly track the requested utilization target.
"""

from __future__ import annotations
# pylint: disable=wrong-import-position

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbatch.gpu_heartbeat import (
    HeartbeatConfig,
    compute_burst_seconds,
    resolve_matmul_dtype,
)


def test_compute_burst_seconds_returns_zero_inside_target_deadband() -> None:
    """Avoid filler work when the observed GPU utilization is already close enough.

    The heartbeat is meant to top up idle gaps, not to fight the training job for every last
    percentage point. This test locks in the deadband behavior so readings at or just below the
    target remain untouched.
    """

    config = HeartbeatConfig(
        target_utilization=70,
        utilization_tolerance=3,
        check_interval=0.5,
        matrix_size=4096,
        min_compute_seconds=0.05,
        max_compute_seconds=0.60,
        compute_gain_seconds=0.015,
        matmuls_per_chunk=4,
        dtype_name="bfloat16",
    )

    assert compute_burst_seconds(70, config=config) == 0.0
    assert compute_burst_seconds(67, config=config) == 0.0


def test_compute_burst_seconds_scales_up_with_larger_utilization_gap() -> None:
    """Launch longer bursts when the GPU is much farther below the requested target.

    The control loop is intended to act like a simple proportional controller. A bigger gap
    should produce more background work than a smaller gap so the helper can recover from deep
    idle periods more aggressively.
    """

    config = HeartbeatConfig(
        target_utilization=70,
        utilization_tolerance=3,
        check_interval=0.5,
        matrix_size=4096,
        min_compute_seconds=0.05,
        max_compute_seconds=0.60,
        compute_gain_seconds=0.015,
        matmuls_per_chunk=4,
        dtype_name="bfloat16",
    )
    smaller_gap = compute_burst_seconds(60, config=config)
    larger_gap = compute_burst_seconds(30, config=config)

    assert smaller_gap > 0.0
    assert larger_gap > smaller_gap


def test_compute_burst_seconds_respects_configured_caps() -> None:
    """Clamp the controller output so bad readings do not create runaway filler work.

    We expect occasional noisy utilization readings on shared systems. This test makes sure the
    helper stays inside the configured safety envelope even when the measured utilization is far
    below the target.
    """

    config = HeartbeatConfig(
        target_utilization=70,
        utilization_tolerance=3,
        check_interval=0.5,
        matrix_size=4096,
        min_compute_seconds=0.10,
        max_compute_seconds=0.50,
        compute_gain_seconds=0.05,
        matmuls_per_chunk=4,
        dtype_name="bfloat16",
    )

    assert compute_burst_seconds(0, config=config) == 0.50


def test_resolve_matmul_dtype_supports_tensor_core_friendly_aliases() -> None:
    """Map the user-facing dtype names onto the torch dtype objects we expect.

    The heartbeat now defaults to Tensor Core friendly dtypes because that is the simplest way
    to create meaningful filler load on L40S-class GPUs. This test protects the small parser
    that turns environment-variable strings into the actual torch dtype constants.
    """

    class _FakeTorch:  # pylint: disable=too-few-public-methods
        float16 = "float16"
        bfloat16 = "bfloat16"
        float32 = "float32"

    fake_torch = _FakeTorch()

    assert resolve_matmul_dtype(fake_torch, "bf16") == "bfloat16"
    assert resolve_matmul_dtype(fake_torch, "fp16") == "float16"
    assert resolve_matmul_dtype(fake_torch, "fp32") == "float32"
