.PHONY: build test train benchmark demo

build:
	uv sync --extra dev

test:
	uv run pytest -q

train:
	uv run python scripts/train_pufferl.py --players-per-team 5 --num-envs 8 --ppo-iterations 1000

benchmark:
	uv run python scripts/benchmark_sps.py --num-envs 64 --seconds 10 --action-mode discrete

demo:
	uv run python main.py
