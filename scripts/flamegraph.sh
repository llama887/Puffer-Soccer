#!/usr/bin/env bash
# Profile the training loop with py-spy and write a flame graph SVG.
#
# Environment variable overrides:
#   PLAYERS    players per team (default: 5)
#   NUM_ENVS   parallel environments (default: 8)
#   PPO_ITERS  PPO iterations to profile (default: 30)
#   OUTPUT     output SVG path (default: experiments/flame.svg)
#   RATE       py-spy sample rate in Hz (default: 100)
#   NATIVE     set to 1 to include C-extension frames (requires debug build)
set -euo pipefail

PLAYERS=${PLAYERS:-5}
NUM_ENVS=${NUM_ENVS:-8}
PPO_ITERS=${PPO_ITERS:-30}
OUTPUT=${OUTPUT:-experiments/flame.svg}
RATE=${RATE:-100}
NATIVE=${NATIVE:-0}

mkdir -p "$(dirname "$OUTPUT")"

PY_SPY_FLAGS=(record -o "$OUTPUT" --rate "$RATE" --subprocesses)
if [ "$NATIVE" = "1" ]; then
    PY_SPY_FLAGS+=(--native)
fi

echo "Profiling training loop → $OUTPUT"
echo "  players-per-team=$PLAYERS  num-envs=$NUM_ENVS  ppo-iterations=$PPO_ITERS"

uv run py-spy "${PY_SPY_FLAGS[@]}" -- python scripts/train_pufferl.py \
    --players-per-team "$PLAYERS" \
    --num-envs "$NUM_ENVS" \
    --ppo-iterations "$PPO_ITERS" \
    --vec-backend native \
    --no-wandb \
    --no-export-videos

echo "Flame graph saved to $OUTPUT"
echo "Open in a browser: file://$(realpath "$OUTPUT")"
