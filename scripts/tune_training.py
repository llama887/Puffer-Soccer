from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import psutil

import pufferlib
import pufferlib.vector
from pufferlib.emulation import PettingZooPufferEnv

from puffer_soccer.envs.marl2d import make_parallel_env
from puffer_soccer.torch_loader import import_torch
from puffer_soccer.utilization import UtilizationMonitor, query_nvidia_smi

if TYPE_CHECKING:
    import torch

torch = import_torch()


def parse_bool_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def layer_init(layer: torch.nn.Module, std: float = 1.0, bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def format_num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


class Policy(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = env.single_observation_space.shape[0]
        self.discrete = hasattr(env.single_action_space, "n")
        hidden_dim = 256
        self.net = torch.nn.Sequential(
            layer_init(torch.nn.Linear(obs_dim, hidden_dim), std=2**0.5),
            torch.nn.ReLU(),
            layer_init(torch.nn.Linear(hidden_dim, hidden_dim), std=2**0.5),
            torch.nn.ReLU(),
        )
        if self.discrete:
            act_dim = env.single_action_space.n
            self.action_head = torch.nn.Linear(hidden_dim, act_dim)
            self.log_std = None
        else:
            act_dim = env.single_action_space.shape[0]
            self.action_head = torch.nn.Linear(hidden_dim, act_dim)
            self.log_std = torch.nn.Parameter(torch.zeros(act_dim))
        self.value_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, observations):
        hidden = self.net(observations)
        return self.action_head(hidden), self.value_head(hidden).squeeze(-1)

    def act(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = self(observations)
        if self.discrete:
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
        else:
            std = torch.exp(self.log_std).expand_as(logits)
            dist = torch.distributions.Normal(logits, std)
            actions = torch.tanh(dist.sample())
        return actions, values

    def ppoish_loss(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        logits, values = self(observations)
        advantages = torch.randn_like(values)
        returns = advantages + values.detach()
        if self.discrete:
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(actions.long())
            entropy = dist.entropy().mean()
        else:
            std = torch.exp(self.log_std).expand_as(logits)
            dist = torch.distributions.Normal(logits, std)
            unclipped_actions = torch.clamp(actions, -0.999, 0.999)
            log_prob = dist.log_prob(unclipped_actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
        policy_loss = -(log_prob * advantages).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns)
        return policy_loss + 0.5 * value_loss - 0.01 * entropy


@dataclass
class Profile:
    agents_per_env: int
    obs_bytes_per_agent: int
    env_ram_gb: float
    single_env_sps: float
    reset_percent: float
    step_variance: float
    num_cores: int
    max_envs: int


@dataclass
class BenchmarkResult:
    vec_backend: str
    num_envs: int
    num_workers: int | None
    vec_batch_size: int | None
    zero_copy: bool | None
    rollout_horizon: int
    update_epochs: int
    minibatch_size: int
    train_batch_size: int
    device: str
    sps: float
    cpu_avg: float | None
    cpu_peak: float | None
    gpu_avg: float | None
    gpu_peak: float | None
    gpu_mem_peak_mb: float | None
    score: float


@dataclass(frozen=True)
class TrainCandidate:
    rollout_horizon: int
    update_epochs: int
    minibatch_size: int


def build_env_creator(
    players_per_team: int, action_mode: str, game_length: int, seed: int
):
    def env_creator(**kwargs):
        pz = make_parallel_env(
            players_per_team=players_per_team,
            action_mode=action_mode,
            game_length=game_length,
            render_mode=None,
            seed=kwargs.get("seed", seed),
        )
        return PettingZooPufferEnv(env=pz)

    return env_creator


def profile_single_env(
    env_creator, time_per_test: float, max_env_ram_gb: float, max_envs: int
) -> Profile:
    num_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    load_ram = psutil.Process().memory_info().rss
    peak_ram = load_ram

    print(f"Profiling single environment for ~{time_per_test:.1f}s", flush=True)
    env = env_creator()
    env.reset()
    actions = []
    for _ in range(1024):
        action = np.array(
            [env.single_action_space.sample() for _ in range(env.num_agents)]
        )
        actions.append(action)

    obs_space = env.single_observation_space
    steps = 0
    step_times = []
    reset_times = []
    start = time.time()
    while time.time() - start < time_per_test:
        peak_ram = max(peak_ram, psutil.Process().memory_info().rss)
        tick = time.time()
        if env.done:
            env.reset()
            reset_times.append(time.time() - tick)
        else:
            env.step(actions[steps % len(actions)])
            step_times.append(time.time() - tick)
        steps += 1

    env.close()
    elapsed = max(sum(step_times) + sum(reset_times), 1e-6)
    step_mean = max(float(np.mean(step_times)) if step_times else 0.0, 1e-9)
    env_ram_gb = max(float(peak_ram - load_ram) / 1e9, 1e-6)
    max_allowed_by_ram = max(1, int(max_env_ram_gb // env_ram_gb))
    return Profile(
        agents_per_env=env.num_agents,
        obs_bytes_per_agent=int(
            np.prod(obs_space.shape) * np.dtype(obs_space.dtype).itemsize
        ),
        env_ram_gb=env_ram_gb,
        single_env_sps=(steps * env.num_agents) / elapsed,
        reset_percent=100.0 * sum(reset_times) / elapsed,
        step_variance=100.0 * float(np.std(step_times)) / step_mean,
        num_cores=num_cores,
        max_envs=min(max_envs, max_allowed_by_ram),
    )


def estimate_gpu_limits(
    profile: Profile, device: str, max_batch_vram_gb: float
) -> int | None:
    obs_gb_per_step = (profile.obs_bytes_per_agent * profile.agents_per_env) / 1e9
    if obs_gb_per_step <= 0:
        return None

    limit = max(1, int(max_batch_vram_gb // obs_gb_per_step))
    if device == "cuda" and torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        runtime_limit = max(1, int(((free_bytes / 1e9) * 0.7) // obs_gb_per_step))
        limit = min(limit, runtime_limit)
    return limit


def vector_config_candidates(profile: Profile, max_envs: int) -> list[dict]:
    num_cores = profile.num_cores
    capped_envs = min(max_envs, profile.max_envs)
    if capped_envs < 1:
        capped_envs = 1

    aligned_envs = (
        capped_envs - (capped_envs % num_cores)
        if capped_envs >= num_cores
        else capped_envs
    )
    if aligned_envs >= num_cores:
        capped_envs = aligned_envs

    candidates: list[dict] = []
    seen: set[tuple] = set()

    def add(config: dict) -> None:
        key = tuple(
            sorted((k, v.__name__ if k == "backend" else v) for k, v in config.items())
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(config)

    for num_envs in sorted(
        {1, min(2, capped_envs), min(4, capped_envs), min(8, capped_envs)}
    ):
        add({"num_envs": num_envs, "backend": pufferlib.vector.Serial})

    if capped_envs <= 1:
        return candidates

    preferred_workers = sorted(
        {1, min(num_cores // 2 or 1, capped_envs), min(num_cores, capped_envs)}
    )
    for num_workers in preferred_workers:
        if num_workers < 1:
            continue
        for envs_per_worker in (1, 2, 4):
            num_envs = num_workers * envs_per_worker
            if num_envs > capped_envs:
                continue
            for batch_size in sorted(
                {envs_per_worker, max(1, num_envs // 2), num_envs}
            ):
                if num_envs % batch_size != 0:
                    continue
                if batch_size % envs_per_worker != 0:
                    continue
                add(
                    {
                        "num_envs": num_envs,
                        "num_workers": num_workers,
                        "batch_size": batch_size,
                        "backend": pufferlib.vector.Multiprocessing,
                    }
                )
                if batch_size != num_envs:
                    add(
                        {
                            "num_envs": num_envs,
                            "num_workers": num_workers,
                            "batch_size": batch_size,
                            "zero_copy": False,
                            "backend": pufferlib.vector.Multiprocessing,
                        }
                    )

    candidates.sort(
        key=lambda cfg: (
            0 if cfg["backend"] is pufferlib.vector.Multiprocessing else 1,
            -cfg.get("num_workers", 0),
            -cfg["num_envs"],
            -cfg.get("batch_size", cfg["num_envs"]),
            0 if cfg.get("zero_copy", True) else 1,
        )
    )
    return candidates


def training_param_candidates(
    vecenv,
    device: str,
    max_train_batch_agents: int | None,
    quick: bool,
) -> list[TrainCandidate]:
    base_horizons = [64, 128, 256] if device == "cuda" else [32, 64, 128]
    epoch_options = (2, 4)
    divisor_options = (2, 4, 8)
    if quick:
        base_horizons = [64, 128] if device == "cuda" else [32, 64]
        epoch_options = (2,)
        divisor_options = (2, 4)

    candidates: list[TrainCandidate] = []
    seen: set[TrainCandidate] = set()
    agents_per_batch = vecenv.agents_per_batch
    for horizon in base_horizons:
        train_batch_size = agents_per_batch * horizon
        if (
            max_train_batch_agents is not None
            and train_batch_size > max_train_batch_agents
        ):
            continue
        for update_epochs in epoch_options:
            for divisor in divisor_options:
                minibatch_size = max(agents_per_batch, train_batch_size // divisor)
                if train_batch_size % minibatch_size != 0:
                    continue
                config = TrainCandidate(horizon, update_epochs, minibatch_size)
                if config not in seen:
                    seen.add(config)
                    candidates.append(config)
    return candidates


def sample_actions(policy: Policy, obs: np.ndarray, device: str) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
    with torch.no_grad():
        actions, _ = policy.act(obs_tensor)
    return actions.detach().cpu().numpy()


def synthetic_update(
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    obs_batches: list[np.ndarray],
    action_batches: list[np.ndarray],
    minibatch_size: int,
    update_epochs: int,
    device: str,
) -> None:
    obs = torch.as_tensor(
        np.concatenate(obs_batches, axis=0), device=device, dtype=torch.float32
    )
    actions_np = np.concatenate(action_batches, axis=0)
    action_dtype = torch.long if policy.discrete else torch.float32
    actions = torch.as_tensor(actions_np, device=device, dtype=action_dtype)

    total = obs.shape[0]
    for _ in range(update_epochs):
        permutation = torch.randperm(total, device=device)
        for start in range(0, total, minibatch_size):
            idx = permutation[start : start + minibatch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = policy.ppoish_loss(obs[idx], actions[idx])
            loss.backward()
            optimizer.step()


def score_result(sps: float, cpu_avg: float | None, gpu_avg: float | None) -> float:
    cpu_term = (cpu_avg or 0.0) / 100.0
    gpu_term = (gpu_avg or 0.0) / 100.0
    return sps * (1.0 + 0.35 * cpu_term + 0.35 * gpu_term)


def print_result(result: BenchmarkResult) -> None:
    details = [
        f"backend={result.vec_backend}",
        f"num_envs={result.num_envs}",
        f"num_workers={result.num_workers if result.num_workers is not None else 'n/a'}",
        f"vec_batch={result.vec_batch_size if result.vec_batch_size is not None else 'n/a'}",
        f"zero_copy={result.zero_copy if result.zero_copy is not None else 'n/a'}",
        f"horizon={result.rollout_horizon}",
        f"epochs={result.update_epochs}",
        f"minibatch={result.minibatch_size}",
        f"sps={result.sps:.1f}",
        f"cpu_avg={format_num(result.cpu_avg)}%",
        f"gpu_avg={format_num(result.gpu_avg)}%",
        f"score={result.score:.1f}",
    ]
    print("  " + " | ".join(details), flush=True)


def benchmark_config(
    env_creator,
    vec_config: dict,
    device: str,
    seconds: float,
    seed: int,
    train_limit: int | None,
    quick: bool,
    config_index: int,
    total_configs: int,
) -> BenchmarkResult | None:
    print(
        f"[{config_index}/{total_configs}] starting "
        f"{vec_config['backend'].__name__} num_envs={vec_config['num_envs']} "
        f"num_workers={vec_config.get('num_workers', 'n/a')} "
        f"vec_batch={vec_config.get('batch_size', 'n/a')} "
        f"zero_copy={vec_config.get('zero_copy', 'n/a')}",
        flush=True,
    )

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        with pufferlib.Suppress():
            vecenv = pufferlib.vector.make(env_creator, seed=seed, **vec_config)
    except Exception as exc:
        print(f"skip vec config {vec_config}: {exc}", flush=True)
        return None

    policy = Policy(vecenv.driver_env).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    train_candidates = training_param_candidates(vecenv, device, train_limit, quick)
    if not train_candidates:
        vecenv.close()
        print(f"skip vec config {vec_config}: no valid train params")
        return None

    obs, _ = vecenv.reset(seed=seed)
    print(
        f"[{config_index}/{total_configs}] benchmarking "
        f"{vec_config['backend'].__name__} num_envs={vec_config['num_envs']} "
        f"num_workers={vec_config.get('num_workers', 'n/a')} "
        f"vec_batch={vec_config.get('batch_size', 'n/a')} "
        f"zero_copy={vec_config.get('zero_copy', 'n/a')} "
        f"with {len(train_candidates)} learner configs",
        flush=True,
    )

    results: list[BenchmarkResult] = []
    for train_index, candidate in enumerate(train_candidates, start=1):
        rollout_horizon = candidate.rollout_horizon
        update_epochs = candidate.update_epochs
        minibatch_size = candidate.minibatch_size
        print(
            f"    learner [{train_index}/{len(train_candidates)}] "
            f"horizon={rollout_horizon} epochs={update_epochs} minibatch={minibatch_size}",
            flush=True,
        )
        rollout_obs: list[np.ndarray] = []
        rollout_actions: list[np.ndarray] = []
        steps = 0
        monitor = UtilizationMonitor()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        monitor.start()
        start = time.time()
        report_interval = max(2.0, seconds / 2.0)
        next_report = start + report_interval
        while time.time() - start < seconds:
            actions = sample_actions(policy, obs, device)
            rollout_obs.append(np.asarray(obs, dtype=np.float32).copy())
            rollout_actions.append(actions.copy())
            obs, _, _, _, _ = vecenv.step(actions)
            steps += vecenv.agents_per_batch
            if len(rollout_obs) >= rollout_horizon:
                synthetic_update(
                    policy,
                    optimizer,
                    rollout_obs,
                    rollout_actions,
                    minibatch_size,
                    update_epochs,
                    device,
                )
                rollout_obs.clear()
                rollout_actions.clear()
            now = time.time()
            if seconds >= 2.0 and now >= next_report:
                print(
                    f"      progress elapsed={now - start:.1f}s "
                    f"steps={steps} sps~={steps / max(now - start, 1e-6):.1f}",
                    flush=True,
                )
                next_report = now + report_interval

        if rollout_obs:
            synthetic_update(
                policy,
                optimizer,
                rollout_obs,
                rollout_actions,
                minibatch_size,
                update_epochs,
                device,
            )
        elapsed = max(time.time() - start, 1e-6)
        util = monitor.stop()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            util["gpu_mem_peak_mb"] = max(
                util["gpu_mem_peak_mb"] or 0.0, gpu_mem_peak_mb
            )

        result = BenchmarkResult(
            vec_backend=vec_config["backend"].__name__,
            num_envs=vec_config["num_envs"],
            num_workers=vec_config.get("num_workers"),
            vec_batch_size=vec_config.get("batch_size"),
            zero_copy=vec_config.get("zero_copy"),
            rollout_horizon=rollout_horizon,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            train_batch_size=vecenv.num_agents * rollout_horizon,
            device=device,
            sps=steps / elapsed,
            cpu_avg=util["cpu_avg"],
            cpu_peak=util["cpu_peak"],
            gpu_avg=util["gpu_avg"],
            gpu_peak=util["gpu_peak"],
            gpu_mem_peak_mb=util["gpu_mem_peak_mb"],
            score=score_result(steps / elapsed, util["cpu_avg"], util["gpu_avg"]),
        )
        results.append(result)
        print_result(result)

    vecenv.close()
    return max(results, key=lambda item: item.score) if results else None


def print_profile(profile: Profile, max_batch_vram_gb: float, device: str) -> None:
    throughput_gb_s = (
        profile.obs_bytes_per_agent * profile.single_env_sps * profile.num_cores
    ) / 1e9
    print("Single-env profile")
    print(f"  agents/env: {profile.agents_per_env}")
    print(f"  obs bytes/agent: {profile.obs_bytes_per_agent}")
    print(f"  single-env SPS: {profile.single_env_sps:.1f}")
    print(f"  reset share: {profile.reset_percent:.2f}%")
    print(f"  step variance: {profile.step_variance:.2f}%")
    print(f"  RAM per env: {profile.env_ram_gb * 1024:.1f} MB")
    print(f"  physical cores: {profile.num_cores}")
    print(f"  capped max envs: {profile.max_envs}")
    print(f"  estimated obs throughput ceiling: {throughput_gb_s:.3f} GB/s")
    if device == "cuda" and torch.cuda.is_available():
        max_train_steps = estimate_gpu_limits(profile, device, max_batch_vram_gb)
        print(f"  capped train batch by VRAM: {max_train_steps} agent-steps")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument(
        "--action-mode", choices=["discrete", "continuous"], default="discrete"
    )
    parser.add_argument("--game-length", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--profile-seconds", type=float, default=4.0)
    parser.add_argument("--benchmark-seconds", type=float, default=6.0)
    parser.add_argument("--max-envs", type=int, default=None)
    parser.add_argument("--max-env-ram-gb", type=float, default=32.0)
    parser.add_argument("--max-batch-vram-gb", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--quick", type=parse_bool_arg, default=True)
    parser.add_argument("--max-configs", type=int, default=12)
    args = parser.parse_args()

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device == "auto":
        device = "cpu"

    env_creator = build_env_creator(
        players_per_team=args.players_per_team,
        action_mode=args.action_mode,
        game_length=args.game_length,
        seed=args.seed,
    )

    max_envs = args.max_envs or max(8, (psutil.cpu_count(logical=False) or 1) * 4)
    profile = profile_single_env(
        env_creator,
        time_per_test=args.profile_seconds,
        max_env_ram_gb=args.max_env_ram_gb,
        max_envs=max_envs,
    )
    print_profile(profile, args.max_batch_vram_gb, device)

    train_limit = (
        estimate_gpu_limits(profile, device, args.max_batch_vram_gb)
        if device == "cuda"
        else None
    )
    vec_candidates = vector_config_candidates(profile, max_envs)
    if args.max_configs is not None:
        vec_candidates = vec_candidates[: max(1, args.max_configs)]
    per_vec_train_candidates = 4 if args.quick else (18 if device == "cpu" else 18)
    estimated_minutes = (
        len(vec_candidates) * per_vec_train_candidates * args.benchmark_seconds
    ) / 60.0
    print(
        f"Benchmarking {len(vec_candidates)} vector configs on device={device}\n",
        flush=True,
    )
    print(
        f"Quick mode: {args.quick}. "
        f"Estimated benchmark floor: ~{estimated_minutes:.1f} minutes plus setup overhead.\n",
        flush=True,
    )

    results: list[BenchmarkResult] = []
    for index, vec_config in enumerate(vec_candidates, start=1):
        result = benchmark_config(
            env_creator,
            vec_config,
            device=device,
            seconds=args.benchmark_seconds,
            seed=args.seed,
            train_limit=train_limit,
            quick=args.quick,
            config_index=index,
            total_configs=len(vec_candidates),
        )
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError("No valid configurations completed")

    results.sort(key=lambda item: item.score, reverse=True)
    print("\nTop results")
    for result in results[: args.top_k]:
        print_result(result)

    best = results[0]
    print("\nRecommended training settings")
    print(f"  num_envs={best.num_envs}")
    print(f"  backend={best.vec_backend}")
    if best.num_workers is not None:
        print(f"  num_workers={best.num_workers}")
    if best.vec_batch_size is not None:
        print(f"  vector_batch_size={best.vec_batch_size}")
    if best.zero_copy is not None:
        print(f"  zero_copy={best.zero_copy}")
    print(f"  rollout_horizon={best.rollout_horizon}")
    print(f"  train_batch_size={best.train_batch_size}")
    print(f"  minibatch_size={best.minibatch_size}")
    print(f"  update_epochs={best.update_epochs}")
    print(f"  device={best.device}")
    print(f"  score={best.score:.1f}")
    print("\nMap these to scripts/train_pufferl.py like this:")
    print(f"  --num-envs {best.num_envs}")
    print(f"  cfg['train']['batch_size'] = {best.train_batch_size}")
    print(f"  cfg['train']['bptt_horizon'] = {best.rollout_horizon}")
    print(f"  cfg['train']['minibatch_size'] = {best.minibatch_size}")
    print(f"  cfg['train']['device'] = '{best.device}'")


if __name__ == "__main__":
    main()
