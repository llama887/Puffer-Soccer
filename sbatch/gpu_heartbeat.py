"""Generate low-priority background GPU work when training under-utilizes the device.

This helper is designed for batch jobs where scheduler policy or utilization checks reward
steady GPU activity. The actual training process should remain the primary workload, so this
script only injects matrix multiplies when the observed GPU utilization falls below a chosen
threshold. As soon as real training pushes utilization above that threshold, the helper backs
off and sleeps.

The implementation keeps all tuning in environment variables rather than CLI flags because the
main use case is an `sbatch` wrapper that starts the helper in the background. That keeps the
launcher simple while still allowing per-GPU overrides. The default values are intentionally
conservative enough to work as a starting point on L40S-class hardware, but they should be
re-tuned for other accelerators.
"""

from __future__ import annotations

import os
import subprocess
import time

import torch


THRESHOLD = int(os.getenv("GPU_HEARTBEAT_THRESHOLD", "50"))
CHECK_INTERVAL = float(os.getenv("GPU_HEARTBEAT_CHECK_INTERVAL", "0.5"))
MATRIX_SIZE = int(os.getenv("GPU_HEARTBEAT_MATRIX_SIZE", "2048"))
BURST_MATMULS = int(os.getenv("GPU_HEARTBEAT_BURST_MATMULS", "15"))


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


def main() -> None:
    """Run an endless low-priority utilization guard loop on the first visible CUDA device.

    The tensors are allocated once up front so the loop itself only performs compute work,
    which avoids repeating allocator overhead and reduces interference with the real training
    process. This script is different from a benchmark or stress test: it intentionally leaves
    the GPU alone whenever training is already active enough, and it is expected to be started
    and later terminated by an outer batch launcher.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("gpu_heartbeat.py requires CUDA but no CUDA device is visible")

    device = torch.device("cuda")
    print(f"Starting GPU heartbeat on {torch.cuda.get_device_name(0)}")
    print(f"PID: {os.getpid()}")
    print(
        "Settings: "
        f"threshold={THRESHOLD}, "
        f"check_interval={CHECK_INTERVAL}, "
        f"matrix_size={MATRIX_SIZE}, "
        f"burst_matmuls={BURST_MATMULS}"
    )

    x = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    y = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)

    while True:
        current_util = get_gpu_utilization()
        if current_util < THRESHOLD:
            for _ in range(BURST_MATMULS):
                torch.mm(x, y)
            torch.cuda.synchronize()
        else:
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
