# Puffer Soccer

Native C-backed MARL 2D soccer environment for PufferLib, with:

- `make_puffer_env(...)` high-throughput native env
- `make_parallel_env(...)` PettingZoo `ParallelEnv` API
- centralized critic state in `infos[agent]["global_state"]`
- discrete and continuous action modes
- PuffeRL training script and SPS benchmark

## Install

```bash
uv sync --extra dev
```

## Quick demo

```bash
uv run python main.py
```

## Train (PuffeRL PPO baseline)

```bash
uv run python scripts/train_pufferl.py --players-per-team 5 --num-envs 8 --total-timesteps 200000
```

This writes a self-play video at `experiments/self_play.mp4` after training.
W&B logging is enabled by default with the `robot-soccer` project and logs the generated self-play video to the same run.

## Benchmark

```bash
uv run python scripts/benchmark_sps.py --num-envs 64 --seconds 10 --action-mode discrete
```

## Tests

```bash
uv run pytest -q
```

`tests/test_parity.py` compares against `third-party/MARL2DFootball` when its dependencies are available.
