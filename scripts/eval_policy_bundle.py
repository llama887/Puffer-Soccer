# pylint: disable=duplicate-code
"""Run head-to-head evaluation between saved policy bundles and/or raw checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any

from puffer_soccer.policy_bundle import load_policy_module_from_bundle
from puffer_soccer.vector_env import VecEnvConfig, make_soccer_vecenv


def _load_train_module():
    script_path = Path(__file__).resolve().with_name("train_pufferl.py")
    spec = importlib.util.spec_from_file_location("train_pufferl_eval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_pufferl = _load_train_module()


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI for head-to-head evaluation between two policy artifacts."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--players-per-team", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--game-length", type=int, default=400)
    parser.add_argument("--eval-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--left-bundle-dir", type=str, default=None)
    parser.add_argument("--left-checkpoint-path", type=str, default=None)
    parser.add_argument("--right-bundle-dir", type=str, default=None)
    parser.add_argument("--right-checkpoint-path", type=str, default=None)
    return parser


def load_policy_runner(
    *,
    bundle_dir: str | None,
    checkpoint_path: str | None,
    players_per_team: int,
    device: str,
) -> tuple[Any, Any]:
    """Load one evaluation-side policy as either a frozen bundle module or a live policy.

    The standalone evaluator supports two artifact formats because training still keeps raw
    checkpoints for backward compatibility. Bundles are preferred when available because they
    do not depend on the current in-repo policy class shape.
    """

    if bundle_dir is not None:
        module, manifest = load_policy_module_from_bundle(Path(bundle_dir), device=device)
        return module, manifest
    if checkpoint_path is None:
        raise RuntimeError("each side must provide either a bundle dir or a checkpoint path")

    env = make_soccer_vecenv(
        players_per_team=players_per_team,
        action_mode="discrete",
        game_length=400,
        render_mode=None,
        seed=0,
        opponents_enabled=True,
        vec=VecEnvConfig(backend="native", shard_num_envs=1, num_shards=1),
    )
    policy = train_pufferl.Policy(env).to(device)
    state_dict = train_pufferl.load_checkpoint_state_dict(
        train_pufferl.resolve_checkpoint_file(Path(checkpoint_path))
    )
    policy.load_state_dict(state_dict, strict=True)
    return policy, env


def main() -> None:
    """Load two policy artifacts and print head-to-head metrics from left's perspective."""

    args = build_parser().parse_args()
    device = train_pufferl.resolve_device(args.device)
    left_runner, left_resource = load_policy_runner(
        bundle_dir=args.left_bundle_dir,
        checkpoint_path=args.left_checkpoint_path,
        players_per_team=args.players_per_team,
        device=device,
    )
    right_runner, right_resource = load_policy_runner(
        bundle_dir=args.right_bundle_dir,
        checkpoint_path=args.right_checkpoint_path,
        players_per_team=args.players_per_team,
        device=device,
    )

    evaluator = train_pufferl.HeadToHeadEvaluator(
        players_per_team=args.players_per_team,
        game_length=args.game_length,
        vec_config=VecEnvConfig(backend="native", shard_num_envs=args.eval_envs, num_shards=1),
        device=device,
    )
    try:
        metrics = evaluator.evaluate(
            left_runner,
            right_runner,
            num_games=args.games,
            seed=args.seed,
        )
        print(
            "Head-to-head evaluation: "
            f"games={int(metrics['games'])}, "
            f"win_rate={metrics['win_rate']:.3f}, "
            f"score_diff={metrics['score_diff']:.3f}"
        )
    finally:
        evaluator.close()
        if hasattr(left_resource, "close"):
            left_resource.close()
        if hasattr(right_resource, "close"):
            right_resource.close()


if __name__ == "__main__":
    main()
