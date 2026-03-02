# Puffer Soccer

Native C-backed MARL 2D soccer environment for PufferLib, with:

- `make_puffer_env(...)` high-throughput native env
- direct rendering from the native env
- centralized critic state in `env.global_states`
- discrete and continuous action modes
- PuffeRL training script and SPS benchmark

PettingZoo compatibility has been removed intentionally to keep training on the native C-batched path.

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
uv run python scripts/train_pufferl.py --players-per-team 5 --num-envs 8 --ppo-iterations 1000
```

This writes a self-play video at `experiments/self_play.mp4` after training.
W&B logging is enabled by default with the `robot-soccer` project and logs the generated self-play video to the same run.

The training path runs directly on `MARL2DPufferEnv` without a PettingZoo wrapper or Python serial vectorizer.

For higher CPU throughput, use Puffer's multiprocessing vecenv with small native shards per worker:

```bash
uv run python scripts/train_pufferl.py \
  --players-per-team 5 \
  --ppo-iterations 1000 \
  --vec-backend multiprocessing \
  --vec-num-shards 16 \
  --shard-num-envs 192 \
  --vec-batch-size 1
```

## Benchmark

```bash
uv run python scripts/benchmark_sps.py --num-envs 64 --seconds 10 --action-mode discrete
```

Autotune the native env's internal parallel env count on the current machine:

```bash
uv run python scripts/benchmark_sps.py --backend native --players-per-team 5 --autotune --seconds 3 --action-mode discrete
```

Benchmark the Puffer multiprocessing layout directly:

```bash
uv run python scripts/benchmark_sps.py \
  --backend multiprocessing \
  --players-per-team 5 \
  --shard-num-envs-list 160,192,224 \
  --num-shards-list 16 \
  --batch-size-list 1,2,4 \
  --seconds 3
```

## Tests

```bash
uv run pytest -q
```

`tests/test_parity.py` compares against `third-party/MARL2DFootball` when its dependencies are available.
