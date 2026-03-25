"""Aggregate confirmed no-opponent runs and render one repeated-scoring video.

This helper exists to close the loop on the sparse-reward no-opponent curriculum experiment.
The sweep finds a strong configuration, separate confirmation runs train the same configuration
from fresh random seeds, and this script then combines those results into one machine-readable
record and one local video artifact.

Keeping this logic in a normal script is simpler and safer than embedding a large Python
snippet inside nested Slurm and Apptainer shell commands. It also makes the finalization step
re-runnable if we later add more confirmation summaries or want to regenerate the video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, cast

import train_pufferl as tp


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI for result aggregation and video export.

    The script operates purely on existing run summaries and checkpoints. Callers pass one or
    more summary paths, an output directory, and optional rollout limits for the video-seed
    scan and the rendered video itself.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-path",
        action="append",
        required=True,
        help="Path to one training run summary JSON. Repeat for multiple confirmed runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where aggregate JSON files and the final video will be written.",
    )
    parser.add_argument(
        "--seed-scan-count",
        type=int,
        default=32,
        help="How many environment seeds to scan when picking a clean video rollout seed.",
    )
    parser.add_argument(
        "--video-max-steps",
        type=int,
        default=600,
        help="Maximum number of environment steps to render into the exported video.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=15,
        help="Frames per second for the exported MP4 video.",
    )
    return parser


def load_payloads(summary_paths: list[Path]) -> list[dict[str, Any]]:
    """Read the requested run summaries from disk.

    The summary files come from `train_pufferl.py` and contain both the objective metrics and
    the saved checkpoint path. Finalization needs both pieces: metrics for aggregation and the
    checkpoint for policy replay.
    """

    return [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]


def aggregate_metrics(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute one compact consistency record across multiple confirmed runs.

    The goal is to answer a simple question: does one tuner-selected configuration reliably
    train repeated-scoring policies from fresh seeds? The aggregate therefore reports mean full
    map scoring metrics across all supplied runs and keeps the raw payloads for traceability.
    """

    metrics = [cast(dict[str, Any], payload["objective_metrics"]) for payload in payloads]
    return {
        "mean_goal_rate": mean(metric["goal_rate"] for metric in metrics),
        "mean_multi_goal_rate": mean(metric["multi_goal_rate"] for metric in metrics),
        "mean_goals_scored": mean(metric["mean_goals_scored"] for metric in metrics),
        "mean_first_goal_step": mean(metric["mean_first_goal_step"] for metric in metrics),
        "mean_own_goal_rate": mean(metric["own_goal_rate"] for metric in metrics),
        "runs": payloads,
    }


def choose_best_payload(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the strongest confirmed run for checkpoint replay.

    The ordering matches the tuning objective priorities: repeated scoring first, then total
    goals, then whether the agent scores at all, then own-goal avoidance, and finally speed to
    the first goal.
    """

    return max(
        payloads,
        key=lambda payload: (
            payload["objective_metrics"]["multi_goal_rate"],
            payload["objective_metrics"]["mean_goals_scored"],
            payload["objective_metrics"]["goal_rate"],
            -payload["objective_metrics"]["own_goal_rate"],
            -payload["objective_metrics"]["mean_first_goal_step"],
        ),
    )


def build_policy_from_checkpoint(checkpoint_path: Path, players_per_team: int) -> tp.Policy:
    """Load one saved policy checkpoint onto CPU for evaluation and video export.

    Video rendering does not need GPU throughput, so the policy is restored on CPU to keep the
    finalization step lightweight and compatible with CPU-only Slurm allocations.
    """

    env = tp.make_puffer_env(
        players_per_team=players_per_team,
        action_mode="discrete",
        opponents_enabled=False,
    )
    try:
        policy = tp.Policy(env).to("cpu")
        policy.load_state_dict(tp.load_checkpoint_state_dict(checkpoint_path), strict=True)
        return policy
    finally:
        env.close()


def score_video_seed(
    policy: tp.Policy,
    *,
    players_per_team: int,
    seed: int,
    max_steps: int,
) -> dict[str, float]:
    """Run one greedy no-opponent rollout and summarize its visual quality.

    The exported video should show repeated scoring clearly. This helper scans a small set of
    environment seeds and ranks them by goals scored, whether a goal happens at all, and how
    quickly the first goal arrives.
    """

    rollout = tp.run_no_opponent_rollouts(
        policy,
        players_per_team=players_per_team,
        seed=seed,
        device="cpu",
        num_games=1,
        max_steps=max_steps,
    )[0]
    return {
        "seed": float(seed),
        "goals": rollout["blue_goals"],
        "scored": rollout["goal_rate"],
        "first_goal_step": rollout["mean_first_goal_step"],
    }


def main() -> None:
    """Aggregate confirmed sparse-reward runs and write the final experiment artifacts."""

    args = build_parser().parse_args()
    summary_paths = [Path(path) for path in args.summary_path]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = load_payloads(summary_paths)
    aggregate = aggregate_metrics(payloads)
    aggregate["summary_paths"] = [str(path) for path in summary_paths]
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    best_payload = choose_best_payload(payloads)
    best_checkpoint = Path(cast(str, best_payload["model_path"]))
    players_per_team = 3
    policy = build_policy_from_checkpoint(best_checkpoint, players_per_team)

    seed_candidates = [
        score_video_seed(
            policy,
            players_per_team=players_per_team,
            seed=seed,
            max_steps=args.video_max_steps,
        )
        for seed in range(args.seed_scan_count)
    ]
    seed_candidates.sort(
        key=lambda item: (item["goals"], item["scored"], -item["first_goal_step"]),
        reverse=True,
    )
    (output_dir / "video_seed_scan.json").write_text(
        json.dumps(seed_candidates, indent=2), encoding="utf-8"
    )
    best_seed = int(seed_candidates[0]["seed"])

    video_args = SimpleNamespace(
        players_per_team=players_per_team,
        seed=best_seed,
        no_opponent_team=True,
        video_max_steps=args.video_max_steps,
        video_output=str(output_dir / "best_confirm_video.mp4"),
        video_fps=args.video_fps,
    )
    video_path = tp.save_self_play_video(policy, video_args)

    recipe = {
        "best_checkpoint": str(best_checkpoint),
        "best_seed_for_video": best_seed,
        "video_path": None if video_path is None else str(video_path),
        "aggregate_metrics": {
            "mean_goal_rate": aggregate["mean_goal_rate"],
            "mean_multi_goal_rate": aggregate["mean_multi_goal_rate"],
            "mean_goals_scored": aggregate["mean_goals_scored"],
            "mean_first_goal_step": aggregate["mean_first_goal_step"],
            "mean_own_goal_rate": aggregate["mean_own_goal_rate"],
        },
        "best_effective_hyperparameters": best_payload["effective_hyperparameters"],
    }
    (output_dir / "best_recipe.json").write_text(
        json.dumps(recipe, indent=2), encoding="utf-8"
    )
    print(json.dumps(recipe, indent=2))


if __name__ == "__main__":
    main()
