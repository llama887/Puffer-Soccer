"""Pretune and persist the highest-SPS vecenv layout for the automode Slurm job.

This script is meant to be launched once per fixed cluster architecture, ideally with
`srun`, before running long self-play jobs through `sbatch/train_automode.sbatch`.
It reuses the repo's existing vecenv autotuner, records the winning layout in the same
standardized autoload JSON that the training script already consumes, and preserves any
existing PPO defaults already stored in that file.

The operational goal is simple: benchmark the machine once, save the result, and then let
every later training run reuse that vecenv layout automatically without editing the batch
script or paying the autotune startup cost again.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import socket
import time

import numpy as np

import train_pufferl
from puffer_soccer.autotune import (
    BenchmarkResult,
    autotune_vecenv,
    format_benchmark_result,
    vec_config_from_benchmark,
)
from puffer_soccer.torch_loader import import_torch
from puffer_soccer.utilization import UtilizationMonitor

torch = import_torch()


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI for one-shot SPS pretuning on the current machine.

    The tuning surface here is intentionally narrow because the user already fixed the
    hardware shape in the Slurm script. We only expose the environment properties that
    change SPS materially for the vectorizer search, together with the output path so the
    same command can target either the canonical autoload file or an experiment-local copy.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=Path,
        default=train_pufferl.STANDARD_HYPERPARAMETERS_PATH,
    )
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument(
        "--vec-backend",
        type=str,
        default="auto",
        choices=["native", "multiprocessing", "auto"],
    )
    parser.add_argument("--autotune-seconds", type=float, default=1.0)
    parser.add_argument("--autotune-max-num-envs", type=int, default=None)
    parser.add_argument("--autotune-max-num-shards", type=int, default=None)
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="training_like",
        choices=["env", "training_like"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--training-benchmark-seconds", type=float, default=6.0)
    parser.add_argument("--training-shortlist-per-backend", type=int, default=3)
    parser.add_argument(
        "--source-label",
        type=str,
        default="pretuned_train_automode_vecenv",
    )
    return parser


def load_existing_autoload_record(path: Path) -> dict[str, object]:
    """Load the existing standardized autoload record, returning an empty one when absent.

    The pretune step should be additive. Teams may already rely on the autoload file for
    PPO defaults from a previous sweep, so this helper preserves that state and only updates
    the vecenv-specific portion. Returning a mutable plain dictionary makes the later merge
    logic straightforward while still validating that any on-disk payload is JSON-object
    shaped.
    """

    payload = train_pufferl.read_json_record(path)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected mapping in standardized autoload file: {path}")
    return dict(payload)


def update_autoload_record(
    existing: Mapping[str, object],
    *,
    vecenv_defaults: Mapping[str, object],
    vecenv_benchmark: Mapping[str, object],
    vecenv_training_benchmark: Mapping[str, object] | None,
    selection_reason: str,
    source_label: str,
) -> dict[str, object]:
    """Merge a new vecenv benchmark winner into a standardized autoload JSON record.

    The existing record may already carry train defaults, rollout scaling metadata, and
    provenance about earlier hyperparameter sweeps. This helper updates only the pieces
    related to SPS pretuning while leaving those training defaults untouched. That keeps the
    user's established workflow intact: one shared JSON file can now carry both PPO defaults
    and the hardware-specific vecenv layout for the automode batch job.

    The extra `vecenv_tuning` block is stored alongside the defaults because long-running
    RL jobs benefit from simple postmortem evidence. When someone later asks why a certain
    layout was chosen, the saved hostname, timestamp, and autotuner selection reason are
    already in the same file.
    """

    merged = dict(existing)
    merged["format_version"] = 1
    merged["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    merged["vecenv_defaults"] = train_pufferl.standardized_vecenv_defaults(
        vecenv_defaults
    )
    merged["vecenv_benchmark"] = dict(vecenv_benchmark)
    merged["vecenv_training_benchmark"] = (
        {} if vecenv_training_benchmark is None else dict(vecenv_training_benchmark)
    )
    merged["vecenv_tuning"] = {
        "selection_reason": selection_reason,
        "source_label": source_label,
        "hostname": socket.gethostname(),
        "saved_at_utc": merged["generated_at_utc"],
    }
    return merged


def shortlist_training_like_candidates(
    results: tuple[BenchmarkResult, ...], top_per_backend: int
) -> tuple[BenchmarkResult, ...]:
    """Return a small backend-balanced shortlist for the slower training-like rerank.

    The environment-only autotune can cheaply evaluate a wide search space, but the follow-up
    training-like check is intentionally more expensive because it includes policy forward
    passes, rollout copies, and synthetic learner work. We therefore keep that second stage
    focused on only a few strong candidates from each backend.

    Balancing the shortlist by backend matters for this project because the env-only metric
    can overstate multiprocessing throughput on some machines while native remains a strong
    end-to-end training option. Taking the top candidates per backend preserves that cross-
    backend comparison instead of letting one backend crowd out the other before reranking.

    Native needs extra care. Its env-only SPS often peaks at very small env counts, but those
    tiny layouts can be poor matches for real PPO throughput once rollout copies and learner
    work are included. We therefore keep the best small native result while also forcing a
    few larger, representative native env counts into the shortlist so the training-like
    rerank can make a fair final decision.
    """

    if top_per_backend < 1:
        raise ValueError("training shortlist must keep at least one candidate per backend")

    def candidate_key(
        result: BenchmarkResult,
    ) -> tuple[str, int, int, int | None]:
        """Return a stable identity tuple for deduplicating vecenv candidates.

        The rerank stores candidates from multiple selection rules. Deduplicating by backend
        and effective vecenv shape keeps the shortlist deterministic and avoids benchmarking
        the exact same layout twice when a candidate is selected by both the small-env and
        large-env native heuristics.
        """

        return (
            result.backend,
            result.shard_num_envs,
            result.num_shards,
            result.batch_size,
        )

    def append_unique(
        destination: list[BenchmarkResult],
        seen_keys: set[tuple[str, int, int, int | None]],
        result: BenchmarkResult,
    ) -> None:
        """Append a candidate once while preserving the shortlist's original order.

        The shortlist combines candidates from a few heuristics. Order matters because the
        rerank logs candidates in the same sequence that they were chosen, so this helper
        keeps the result readable while preventing duplicate work.
        """

        key = candidate_key(result)
        if key in seen_keys:
            return
        seen_keys.add(key)
        destination.append(result)

    def rank_backend_results(backend: str) -> list[BenchmarkResult]:
        """Return backend candidates ordered by the cheap env-only benchmark.

        The env-only score is still a useful first-pass filter. We keep using it to choose
        promising candidates inside each backend before the slower training-like rerank
        decides the final winner.
        """

        return sorted(
            (result for result in results if result.backend == backend),
            key=lambda result: (
                result.sps,
                -1.0 if result.cpu_avg is None else result.cpu_avg,
                -1.0 if result.cpu_peak is None else result.cpu_peak,
            ),
            reverse=True,
        )

    def choose_native_representative(
        ranked_native_results: list[BenchmarkResult], target_num_envs: int
    ) -> BenchmarkResult | None:
        """Pick the strongest native candidate near a representative env-count target.

        The purpose of this helper is to stop the native shortlist from collapsing to only
        tiny env counts such as 2, 3, and 5 envs. We prefer the smallest candidate at or
        above the target because that keeps the representative points easy to reason about.
        When the target is above the search range, we fall back to the largest available
        native layout instead of silently dropping the large-env comparison entirely.
        """

        if not ranked_native_results:
            return None
        eligible = [
            result for result in ranked_native_results if result.total_envs >= target_num_envs
        ]
        pool = eligible if eligible else ranked_native_results
        return min(
            pool,
            key=lambda result: (
                abs(result.total_envs - target_num_envs),
                -result.sps,
            ),
        )

    shortlisted: list[BenchmarkResult] = []
    seen: set[tuple[str, int, int, int | None]] = set()

    native_results = rank_backend_results("native")
    if native_results:
        native_count = 0
        append_unique(shortlisted, seen, native_results[0])
        native_count += 1
        native_target_envs = (64, 128, 256, 512)
        for target_num_envs in native_target_envs[: max(top_per_backend - 1, 0)]:
            representative = choose_native_representative(native_results, target_num_envs)
            if representative is not None:
                before = len(shortlisted)
                append_unique(shortlisted, seen, representative)
                if len(shortlisted) > before:
                    native_count += 1
        for result in native_results:
            if native_count >= top_per_backend:
                break
            before = len(shortlisted)
            append_unique(shortlisted, seen, result)
            if len(shortlisted) > before:
                native_count += 1

    multiprocessing_results = rank_backend_results("multiprocessing")
    for result in multiprocessing_results[:top_per_backend]:
        append_unique(shortlisted, seen, result)

    if not shortlisted:
        raise RuntimeError("no vecenv candidates available for training-like rerank")
    return tuple(shortlisted)


def resolve_benchmark_device(name: str) -> str:
    """Resolve the device used for the training-like rerank.

    The vecenv sweep is meant to choose the layout that is fastest for the actual automode
    training job. That means the rerank should run on the same broad device class as the
    final job when possible, especially because rollout-copy pressure and learner overlap can
    shift the best layout between CPU-only and GPU-backed runs.

    Accepting `auto` keeps the CLI ergonomic for the batch script while still failing fast
    when a caller explicitly asks for CUDA on a machine that does not have it.
    """

    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for vecenv rerank but torch.cuda.is_available() is false")
    return name


def sample_policy_actions(
    policy: torch.nn.Module,
    observations: np.ndarray,
    device: str,
) -> np.ndarray:
    """Run one lightweight policy forward pass and return sampled actions as NumPy.

    The rerank is not trying to learn a good policy. Its purpose is to expose the same high-
    level costs that dominate real training throughput: host-to-device observation transfer,
    policy inference, action sampling, and device-to-host action materialization. Using the
    normal training policy architecture makes those costs much more representative than a raw
    random-action environment benchmark.
    """

    obs_tensor = torch.as_tensor(observations, device=device, dtype=torch.float32)
    with torch.no_grad():
        logits, _ = policy(obs_tensor)
        if hasattr(policy, "discrete") and policy.discrete:
            actions = torch.distributions.Categorical(logits=logits).sample()
        else:
            actions = torch.tanh(logits)
    return actions.detach().cpu().numpy()


def synthetic_update(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_batches: list[np.ndarray],
    action_batches: list[np.ndarray],
    *,
    minibatch_size: int,
    update_epochs: int,
    device: str,
) -> None:
    """Run a small synthetic learner update that approximates rollout-copy pressure.

    The bad automode choice came from ranking vecenvs on simulator stepping alone even though
    the real training loop spends substantial time copying rollout tensors and running learner
    updates. This helper deliberately adds a simplified learner phase so the rerank measures
    the same kind of end-to-end pressure that the actual PPO job experiences.

    The loss itself is intentionally generic. We only need stable work that scales with the
    amount of collected rollout data, not exact PPO semantics. That keeps the rerank small,
    fast, and robust while still penalizing layouts that look amazing in bare env SPS but
    collapse once rollout copies and gradient work are included.
    """

    obs = torch.as_tensor(
        np.concatenate(obs_batches, axis=0), device=device, dtype=torch.float32
    )
    actions_np = np.concatenate(action_batches, axis=0)
    action_dtype = torch.long if getattr(policy, "discrete", False) else torch.float32
    actions = torch.as_tensor(actions_np, device=device, dtype=action_dtype)
    total = int(obs.shape[0])

    for _ in range(update_epochs):
        permutation = torch.randperm(total, device=device)
        for start in range(0, total, minibatch_size):
            idx = permutation[start : start + minibatch_size]
            optimizer.zero_grad(set_to_none=True)
            logits, values = policy(obs[idx])
            if getattr(policy, "discrete", False):
                log_prob = torch.distributions.Categorical(logits=logits).log_prob(
                    actions[idx].long()
                )
            else:
                log_prob = -torch.square(logits - actions[idx]).sum(dim=-1)
            advantages = torch.randn_like(values.squeeze(-1))
            returns = advantages + values.squeeze(-1).detach()
            policy_loss = -(log_prob * advantages).mean()
            value_loss = torch.nn.functional.mse_loss(values.squeeze(-1), returns)
            loss = policy_loss + 0.5 * value_loss
            loss.backward()
            optimizer.step()


def benchmark_training_like_candidate(
    candidate: BenchmarkResult,
    *,
    players_per_team: int,
    seconds: float,
    device: str,
) -> dict[str, float | str | int | None]:
    """Measure one vecenv with a short training-like loop and return comparable metrics.

    The original pretune path measured only simulator stepping. That was not enough for this
    project because the chosen layout also needs to stay fast once real policy inference,
    rollout staging, and learner updates enter the loop. This benchmark mirrors that fuller
    path while staying short enough to run at job startup.

    The returned payload is JSON-safe so it can be stored directly next to the selected
    vecenv defaults. That gives later debugging a concrete record of why one candidate beat
    the others under the more realistic benchmark.
    """

    vec_config = vec_config_from_benchmark(candidate)
    vecenv = train_pufferl.make_soccer_vecenv(
        players_per_team=players_per_team,
        action_mode="discrete",
        game_length=400,
        render_mode=None,
        seed=0,
        vec=vec_config,
    )
    policy = train_pufferl.Policy(vecenv).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    obs, _ = vecenv.reset(seed=0)
    rollout_horizon = 64
    minibatch_size = max(int(np.asarray(obs).shape[0]), (int(np.asarray(obs).shape[0]) * rollout_horizon) // 4)
    rollout_obs: list[np.ndarray] = []
    rollout_actions: list[np.ndarray] = []
    steps = 0
    monitor = UtilizationMonitor()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    monitor.start()
    start = time.time()
    try:
        while time.time() - start < seconds:
            obs_array = np.asarray(obs, dtype=np.float32)
            actions = sample_policy_actions(policy, obs_array, device)
            rollout_obs.append(obs_array.copy())
            rollout_actions.append(actions.copy())
            obs, _, _, _, _ = vecenv.step(actions)
            steps += int(obs_array.shape[0])
            if len(rollout_obs) >= rollout_horizon:
                synthetic_update(
                    policy,
                    optimizer,
                    rollout_obs,
                    rollout_actions,
                    minibatch_size=minibatch_size,
                    update_epochs=1,
                    device=device,
                )
                rollout_obs.clear()
                rollout_actions.clear()

        if rollout_obs:
            synthetic_update(
                policy,
                optimizer,
                rollout_obs,
                rollout_actions,
                minibatch_size=minibatch_size,
                update_epochs=1,
                device=device,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = max(time.time() - start, 1e-6)
        util = monitor.stop()
        return {
            "backend": candidate.backend,
            "num_envs": candidate.total_envs,
            "vec_num_shards": None if candidate.backend == "native" else candidate.num_shards,
            "vec_batch_size": None if candidate.backend == "native" else candidate.batch_size,
            "training_like_sps": float(steps / elapsed),
            "cpu_avg": None if util["cpu_avg"] is None else float(util["cpu_avg"]),
            "cpu_peak": None if util["cpu_peak"] is None else float(util["cpu_peak"]),
            "gpu_avg": None if util["gpu_avg"] is None else float(util["gpu_avg"]),
            "gpu_peak": None if util["gpu_peak"] is None else float(util["gpu_peak"]),
        }
    finally:
        vecenv.close()


def choose_best_training_like_candidate(
    candidates: tuple[BenchmarkResult, ...],
    *,
    players_per_team: int,
    seconds: float,
    device: str,
) -> tuple[BenchmarkResult, dict[str, float | str | int | None]]:
    """Rerank shortlisted vecenvs with a short end-to-end training-style benchmark.

    This second stage is intentionally narrow and conservative. It trusts the cheaper env
    sweep only to provide promising candidates, then makes the final backend decision using a
    benchmark that better matches the real self-play training loop. That is exactly the
    failure mode we saw in automode: a vecenv that looked best in bare env SPS became slow
    once rollout copying and learner work were included.
    """

    best_candidate: BenchmarkResult | None = None
    best_metrics: dict[str, float | str | int | None] | None = None
    for index, candidate in enumerate(candidates, start=1):
        print(
            "TRAIN_LIKE_AUTOTUNE\t"
            f"[{index}/{len(candidates)}]\t"
            f"{format_benchmark_result(candidate)}",
            flush=True,
        )
        metrics = benchmark_training_like_candidate(
            candidate,
            players_per_team=players_per_team,
            seconds=seconds,
            device=device,
        )
        print(
            "TRAIN_LIKE_AUTOTUNE_RESULT\t"
            f"backend={metrics['backend']}\t"
            f"num_envs={metrics['num_envs']}\t"
            f"vec_num_shards={metrics['vec_num_shards']}\t"
            f"vec_batch_size={metrics['vec_batch_size']}\t"
            f"training_like_sps={metrics['training_like_sps']:.1f}\t"
            f"cpu_avg={metrics['cpu_avg']}\t"
            f"gpu_avg={metrics['gpu_avg']}",
            flush=True,
        )
        if (
            best_metrics is None
            or float(metrics["training_like_sps"]) > float(best_metrics["training_like_sps"])
        ):
            best_candidate = candidate
            best_metrics = metrics

    if best_candidate is None or best_metrics is None:
        raise RuntimeError("training-like vecenv rerank did not produce a winner")
    return best_candidate, best_metrics


def main() -> None:
    """Run the vecenv autotuner and persist the winning layout for later Slurm jobs.

    This script intentionally benchmarks only the rollout backend, not the PPO optimizer.
    The resulting file is therefore best verified by comparing later training runs' logged
    SPS against the saved benchmark metadata and by confirming that `train_automode.sbatch`
    now prints the same vecenv layout without rerunning autotune on startup.
    """

    args = build_parser().parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    outcome = autotune_vecenv(
        players_per_team=args.players_per_team,
        seconds=args.autotune_seconds,
        action_mode="discrete",
        backend=args.vec_backend,
        max_num_envs=args.autotune_max_num_envs,
        max_num_shards=args.autotune_max_num_shards,
        reporter=print,
    )
    best = outcome.best
    training_benchmark: dict[str, float | str | int | None] | None = None
    selection_reason = outcome.selection_reason
    if args.selection_mode == "training_like":
        shortlist = shortlist_training_like_candidates(
            outcome.all_results,
            top_per_backend=args.training_shortlist_per_backend,
        )
        benchmark_device = resolve_benchmark_device(args.device)
        best, training_benchmark = choose_best_training_like_candidate(
            shortlist,
            players_per_team=args.players_per_team,
            seconds=args.training_benchmark_seconds,
            device=benchmark_device,
        )
        selection_reason = (
            "selected highest training-like SPS among env-autotune shortlist; "
            f"env-stage reason: {outcome.selection_reason}"
        )
    vecenv_defaults = train_pufferl.vecenv_defaults_from_benchmark(best)
    benchmark = train_pufferl.benchmark_record(best)
    existing = load_existing_autoload_record(output_path)
    merged = update_autoload_record(
        existing,
        vecenv_defaults=vecenv_defaults,
        vecenv_benchmark=benchmark,
        vecenv_training_benchmark=training_benchmark,
        selection_reason=selection_reason,
        source_label=args.source_label,
    )
    train_pufferl.write_json_record(output_path, merged)

    print("Saved pretuned vecenv defaults")
    print(f"  output_path={output_path}")
    print(f"  backend={vecenv_defaults['vec_backend']}")
    print(f"  num_envs={vecenv_defaults['num_envs']}")
    print(f"  vec_num_shards={vecenv_defaults.get('vec_num_shards')}")
    print(f"  vec_batch_size={vecenv_defaults.get('vec_batch_size')}")
    print(f"  benchmark={format_benchmark_result(best)}")
    if training_benchmark is not None:
        print(
            "  training_like_benchmark="
            f"training_like_sps={training_benchmark['training_like_sps']:.1f}, "
            f"cpu_avg={training_benchmark['cpu_avg']}, "
            f"gpu_avg={training_benchmark['gpu_avg']}"
        )
    print(f"  selection_reason={selection_reason}")


if __name__ == "__main__":
    main()
