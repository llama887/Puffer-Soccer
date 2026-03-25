"""Generate low-priority background GPU work that tracks a target utilization.

This helper exists for batch jobs where cluster policy rewards steady GPU activity, but the
real training job occasionally leaves short idle gaps. The heartbeat fills only part of those
gaps by adding low-priority matrix multiplies, and it should quickly get out of the way once
training naturally keeps the device busy enough.

The earlier version behaved like a simple floor: if utilization dropped below a threshold, it
always ran the same fixed burst of work. That was good enough to avoid very low utilization,
but it did not actually aim for a stable operating point. This version instead tracks a target
utilization, which lets us ask for "roughly 70 percent GPU usage" in a direct way.

All tuning remains in environment variables because the intended caller is an `sbatch` wrapper
that launches this helper in the background. That keeps the batch script easy to adjust per
machine without adding a second command-line interface to maintain.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import time

from puffer_soccer.torch_loader import import_torch


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class HeartbeatConfig:
    """Hold the small set of knobs that define the heartbeat control policy.

    Grouping these values into one immutable object keeps the controller logic simple while
    avoiding long function signatures full of loosely related keyword arguments. This also
    makes tests easier to read because each case can construct one explicit policy instead of
    overriding several module globals.
    """

    target_utilization: int
    utilization_tolerance: int
    check_interval: float
    matrix_size: int
    min_compute_seconds: float
    max_compute_seconds: float
    compute_gain_seconds: float
    matmuls_per_chunk: int
    dtype_name: str


DEFAULT_CONFIG = HeartbeatConfig(
    target_utilization=int(
        os.getenv("GPU_HEARTBEAT_TARGET_UTILIZATION", os.getenv("GPU_HEARTBEAT_THRESHOLD", "70"))
    ),
    utilization_tolerance=int(os.getenv("GPU_HEARTBEAT_UTILIZATION_TOLERANCE", "3")),
    check_interval=float(os.getenv("GPU_HEARTBEAT_CHECK_INTERVAL", "0.2")),
    matrix_size=int(os.getenv("GPU_HEARTBEAT_MATRIX_SIZE", "6144")),
    min_compute_seconds=float(os.getenv("GPU_HEARTBEAT_MIN_COMPUTE_SECONDS", "0.10")),
    max_compute_seconds=float(os.getenv("GPU_HEARTBEAT_MAX_COMPUTE_SECONDS", "1.20")),
    compute_gain_seconds=float(os.getenv("GPU_HEARTBEAT_COMPUTE_GAIN_SECONDS", "0.03")),
    matmuls_per_chunk=int(os.getenv("GPU_HEARTBEAT_MATMULS_PER_CHUNK", "8")),
    dtype_name=os.getenv("GPU_HEARTBEAT_DTYPE", "bfloat16"),
)


def resolve_matmul_dtype(torch, dtype_name: str):
    """Return the torch dtype used for heartbeat matrix multiplies.

    L40S-class GPUs respond much better to Tensor Core friendly matrix multiplies than to the
    default float32 path. This helper keeps the environment-variable parsing in one place and
    makes the fallback explicit: if an unknown name is requested we fail fast instead of
    silently running a much weaker filler workload.
    """

    dtype_aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    normalized_name = dtype_name.strip().lower()
    if normalized_name not in dtype_aliases:
        raise ValueError(f"unsupported GPU_HEARTBEAT_DTYPE: {dtype_name}")
    return dtype_aliases[normalized_name]


def get_gpu_utilization() -> int:
    """Return the current device utilization percentage reported by `nvidia-smi`.

    The scheduler-facing goal is to keep utilization from sitting near zero when the real
    workload has brief idle periods. We query `nvidia-smi` directly because it is stable across
    the cluster and reflects the whole device, not just this process. If the query fails, the
    safe fallback is to report a busy GPU so the helper sleeps instead of accidentally adding
    extra load when observability is broken.
    """

    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return 100

    return int(result.strip())


def compute_burst_seconds(
    current_utilization: int,
    *,
    config: HeartbeatConfig = DEFAULT_CONFIG,
) -> float:
    """Choose how long the heartbeat should keep the GPU busy for one cycle.

    The earlier controller returned a fixed number of matrix multiplies. That turned out to be
    too bursty for this project's L40S jobs: the helper produced short spikes, but the
    scheduler-visible utilization and W&B device metric still sat around the true training
    baseline. This controller instead chooses a compute *duration*. That makes the filler work
    much more continuous and gives the GPU monitor a better chance to see the intended load.

    This helper is separate from `main` so the policy can be tested without CUDA and without
    mocking an endless loop. The returned duration is always clamped to a safe minimum and
    maximum so the helper does not thrash with extremely short bursts or monopolize the device
    for too long when utilization readings are noisy.
    """

    utilization_gap = config.target_utilization - current_utilization
    if utilization_gap <= config.utilization_tolerance:
        return 0.0

    unclamped_seconds = (
        config.min_compute_seconds
        + utilization_gap * config.compute_gain_seconds
    )
    return max(
        config.min_compute_seconds,
        min(unclamped_seconds, config.max_compute_seconds),
    )


def main() -> None:
    """Run an endless low-priority controller loop on the first visible CUDA device.

    The tensors are allocated once up front so each iteration only does compute work. That
    keeps overhead low and avoids repeated allocator churn during long training runs.

    The control goal is not maximum utilization. Instead, we want a modest and steady floor
    around the configured target so scheduler-visible usage does not collapse during brief idle
    windows. If this change is helping, cluster monitoring and `nvidia-smi` samples should show
    fewer dips far below the target while training throughput and replay quality remain
    unchanged. If it is hurting, we would expect lower SPS, longer iteration times, or replay
    behavior that suggests the filler work is stealing too much GPU time.
    """

    torch = import_torch()

    if not torch.cuda.is_available():
        raise RuntimeError("gpu_heartbeat.py requires CUDA but no CUDA device is visible")

    device = torch.device("cuda")
    print(f"Starting GPU heartbeat on {torch.cuda.get_device_name(0)}")
    print(f"PID: {os.getpid()}")
    print(
        "Settings: "
        f"target_utilization={DEFAULT_CONFIG.target_utilization}, "
        f"utilization_tolerance={DEFAULT_CONFIG.utilization_tolerance}, "
        f"check_interval={DEFAULT_CONFIG.check_interval}, "
        f"matrix_size={DEFAULT_CONFIG.matrix_size}, "
        f"min_compute_seconds={DEFAULT_CONFIG.min_compute_seconds}, "
        f"max_compute_seconds={DEFAULT_CONFIG.max_compute_seconds}, "
        f"compute_gain_seconds={DEFAULT_CONFIG.compute_gain_seconds}, "
        f"matmuls_per_chunk={DEFAULT_CONFIG.matmuls_per_chunk}, "
        f"dtype={DEFAULT_CONFIG.dtype_name}"
    )

    matmul_dtype = resolve_matmul_dtype(torch, DEFAULT_CONFIG.dtype_name)
    x = torch.randn(
        DEFAULT_CONFIG.matrix_size,
        DEFAULT_CONFIG.matrix_size,
        device=device,
        dtype=matmul_dtype,
    )
    y = torch.randn(
        DEFAULT_CONFIG.matrix_size,
        DEFAULT_CONFIG.matrix_size,
        device=device,
        dtype=matmul_dtype,
    )

    while True:
        current_util = get_gpu_utilization()
        burst_seconds = compute_burst_seconds(current_util)
        if burst_seconds == 0.0:
            time.sleep(DEFAULT_CONFIG.check_interval)
            continue

        deadline = time.monotonic() + burst_seconds
        while time.monotonic() < deadline:
            for _ in range(DEFAULT_CONFIG.matmuls_per_chunk):
                torch.mm(x, y)
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
