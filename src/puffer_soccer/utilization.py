from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

import numpy as np
import psutil


def query_nvidia_smi() -> dict[str, float] | None:
    if shutil.which("nvidia-smi") is None:
        return None

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None

    try:
        util, mem_used, mem_total = result.stdout.strip().splitlines()[0].split(",")
        return {
            "utilization": float(util.strip()),
            "memory_used_mb": float(mem_used.strip()),
            "memory_total_mb": float(mem_total.strip()),
        }
    except (ValueError, IndexError):
        return None


class UtilizationMonitor:
    def __init__(self, sample_interval_s: float = 0.25):
        self.sample_interval_s = sample_interval_s
        self._cpu_samples: list[float] = []
        self._gpu_samples: list[float] = []
        self._gpu_mem_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        num_cores = (
            psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
        )
        self._cpu_capacity = float(num_cores * 100)
        self._root_process = psutil.Process(os.getpid())
        self._process_cache: dict[int, psutil.Process] = {
            self._root_process.pid: self._root_process
        }

    def start(self) -> None:
        self._prime_process_tree()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

        return {
            "cpu_avg": float(np.mean(self._cpu_samples)) if self._cpu_samples else None,
            "cpu_peak": float(np.max(self._cpu_samples)) if self._cpu_samples else None,
            "gpu_avg": float(np.mean(self._gpu_samples)) if self._gpu_samples else None,
            "gpu_peak": float(np.max(self._gpu_samples)) if self._gpu_samples else None,
            "gpu_mem_peak_mb": float(np.max(self._gpu_mem_samples))
            if self._gpu_mem_samples
            else None,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            time.sleep(self.sample_interval_s)
            self._cpu_samples.append(self._sample_process_tree_cpu())
            gpu_stats = query_nvidia_smi()
            if gpu_stats is not None:
                self._gpu_samples.append(gpu_stats["utilization"])
                self._gpu_mem_samples.append(gpu_stats["memory_used_mb"])

    def _iter_process_tree(self) -> list[psutil.Process]:
        try:
            descendants = self._root_process.children(recursive=True)
        except psutil.Error:
            descendants = []
        active_pids = {self._root_process.pid}
        processes = [self._root_process]
        for proc in descendants:
            active_pids.add(proc.pid)
            cached = self._process_cache.get(proc.pid)
            if cached is None:
                self._process_cache[proc.pid] = proc
                cached = proc
            processes.append(cached)

        stale_pids = [pid for pid in self._process_cache if pid not in active_pids]
        for pid in stale_pids:
            del self._process_cache[pid]

        return processes

    def _prime_process_tree(self) -> None:
        for proc in self._iter_process_tree():
            try:
                proc.cpu_percent(interval=None)
            except psutil.Error:
                continue

    def _sample_process_tree_cpu(self) -> float:
        total = 0.0
        for proc in self._iter_process_tree():
            try:
                total += proc.cpu_percent(interval=None)
            except psutil.Error:
                continue
        return min(100.0, 100.0 * total / self._cpu_capacity)
