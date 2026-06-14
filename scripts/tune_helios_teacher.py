"""Random-search tunable HeliosTeacher constants against a saved policy."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from typing import Any

from puffer_soccer.helios_teacher import HeliosTeacherConfig

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_policy_vs_helios import (  # pylint: disable=wrong-import-position,wrong-import-order
    PolicyVsHeliosEvaluator,
    load_checkpoint_policy,
    train_pufferl,
)


def sample_config(rng: random.Random) -> HeliosTeacherConfig:
    """Sample one plausible Helios teacher configuration for evaluation.

    The search space is deliberately small and soccer-shaped rather than fully generic. These
    knobs control the parts of the scripted teacher that most directly affect strength in our
    environment: how readily it claims ball control, how aggressively it chases, how strict pass
    reach timing is, how far defenders step into blocking lanes, and which discrete kick
    strengths are used for passes, clears, and dribbles. The objective is to reduce the learned
    policy's win rate and score difference, which means a stronger Helios approximation.
    """

    pass_kick = rng.randint(3, 5)
    clear_kick = rng.randint(max(pass_kick + 1, 5), 7)
    return HeliosTeacherConfig(
        ball_control_radius=rng.uniform(3.6, 5.2),
        chase_margin=rng.uniform(4.0, 12.0),
        target_angle_tolerance=rng.uniform(0.22, 0.50),
        kick_angle_tolerance=rng.uniform(0.35, 0.75),
        shot_kick_index=7,
        pass_kick_index=pass_kick,
        clear_kick_index=clear_kick,
        dribble_kick_index=rng.randint(0, 2),
        receiver_reach_buffer=rng.uniform(-0.25, 1.25),
        opponent_reach_buffer=rng.uniform(-0.25, 0.75),
        defensive_block_distance=rng.uniform(4.0, 14.0),
    )


def score_trial(metrics: dict[str, float]) -> float:
    """Return the scalar objective minimized during tuning.

    Win rate is the primary signal because the teacher is an opponent. Score difference is a
    useful secondary signal when most short trials produce the same win rate. Lower values mean
    the policy did worse, so the sampled Helios configuration was stronger.
    """

    return float(metrics["win_rate"]) + 0.05 * float(metrics["score_diff"])


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for Helios teacher tuning."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("experiments/dx9z6tf0/model_057400.pt"),
    )
    parser.add_argument("--players-per-team", type=int, default=11)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--games-per-trial", type=int, default=32)
    parser.add_argument("--eval-envs", type=int, default=16)
    parser.add_argument("--game-length", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("experiments/helios_teacher_tuning.jsonl"),
    )
    return parser


def main() -> None:
    """Run random-search trials and print the best configuration found."""

    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    device = train_pufferl.resolve_device(args.device)
    policy = load_checkpoint_policy(
        checkpoint_path=args.checkpoint_path,
        players_per_team=args.players_per_team,
        device=device,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    best_record: dict[str, Any] | None = None
    best_objective = float("inf")
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for trial_idx in range(args.trials):
            config = sample_config(rng)
            evaluator_kwargs = {
                "players_per_team": args.players_per_team,
                "game_length": args.game_length,
                "eval_envs": args.eval_envs,
                "device": device,
                "teacher_config": config,
            }
            evaluator = PolicyVsHeliosEvaluator(**evaluator_kwargs)
            try:
                metrics = evaluator.evaluate(
                    policy,
                    games=args.games_per_trial,
                    seed=args.seed + trial_idx * 10_000,
                )
            finally:
                evaluator.close()

            record: dict[str, Any] = {
                "trial": trial_idx,
                "objective": score_trial(metrics),
                "metrics": metrics,
                "config": asdict(config),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            objective = float(record["objective"])
            if objective < best_objective:
                best_record = record
                best_objective = objective
            print(
                f"trial={trial_idx} objective={record['objective']:.4f} "
                f"win_rate={metrics['win_rate']:.3f} score_diff={metrics['score_diff']:.3f}"
            )

    if best_record is None:
        raise RuntimeError("no tuning trials were run")
    print("Best Helios config:")
    print(json.dumps(best_record, indent=2, sort_keys=True))
    print(f"results={args.output_jsonl}")


if __name__ == "__main__":
    main()
