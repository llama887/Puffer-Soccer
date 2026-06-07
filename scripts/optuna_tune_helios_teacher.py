"""Tune HeliosTeacher constants with Optuna against a saved policy checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import optuna

from puffer_soccer.helios_teacher import HeliosTeacherConfig

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_policy_vs_helios import (  # pylint: disable=wrong-import-position,wrong-import-order
    PolicyVsHeliosEvaluator,
    load_checkpoint_policy,
    train_pufferl,
)


def config_from_trial(trial: optuna.Trial) -> HeliosTeacherConfig:
    """Build one HeliosTeacherConfig from Optuna's suggested parameters.

    The search space focuses on parameters that map cleanly to our environment: ball control,
    chase aggressiveness, body alignment thresholds, discrete kick strengths, pass reach-time
    buffers, and defensive blocking distance. It deliberately does not tune RoboCup-only ideas
    such as stamina, view direction, goalie catch, or tackle because our simulator does not
    expose those actions.
    """

    pass_kick = trial.suggest_int("pass_kick_index", 3, 5)
    clear_kick = trial.suggest_int("clear_kick_index", max(pass_kick + 1, 5), 7)
    return HeliosTeacherConfig(
        ball_control_radius=trial.suggest_float("ball_control_radius", 3.4, 5.4),
        chase_margin=trial.suggest_float("chase_margin", 3.0, 13.0),
        target_angle_tolerance=trial.suggest_float("target_angle_tolerance", 0.20, 0.55),
        kick_angle_tolerance=trial.suggest_float("kick_angle_tolerance", 0.30, 0.80),
        shot_kick_index=7,
        pass_kick_index=pass_kick,
        clear_kick_index=clear_kick,
        dribble_kick_index=trial.suggest_int("dribble_kick_index", 0, 2),
        receiver_reach_buffer=trial.suggest_float("receiver_reach_buffer", -0.5, 1.5),
        opponent_reach_buffer=trial.suggest_float("opponent_reach_buffer", -0.5, 1.0),
        defensive_block_distance=trial.suggest_float("defensive_block_distance", 3.0, 15.0),
    )


def config_from_trial_attrs(trial: optuna.trial.FrozenTrial) -> HeliosTeacherConfig:
    """Rebuild the stored teacher config from an already-completed Optuna trial.

    Optuna exposes live trials and completed trials through different objects. Live trials can
    suggest new parameter values, while completed trials should be treated as immutable records.
    During optimization we build the config with ``config_from_trial`` and store the exact
    dataclass fields in ``user_attrs``. This helper reads that stored record back instead of
    asking Optuna to suggest values again, which keeps the final evaluation tied to the real best
    trial even if the search space is later edited.
    """

    config_data = trial.user_attrs["config"]
    if not isinstance(config_data, dict):
        raise TypeError("Best Optuna trial is missing a dictionary 'config' user attribute")
    return HeliosTeacherConfig(**config_data)


def objective_from_metrics(metrics: dict[str, float]) -> float:
    """Return the minimized tuning objective from the learned policy's perspective.

    Lower is better. Win rate is primary because we want the scripted Helios opponent to win or
    draw more often. Mean score difference breaks ties in noisy short evaluations, so a Helios
    config that loses by fewer goals is preferred even when the win rate is unchanged.
    """

    return float(metrics["win_rate"]) + 0.05 * float(metrics["score_diff"])


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for Optuna-based Helios teacher tuning."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("experiments/dx9z6tf0/model_057400.pt"),
    )
    parser.add_argument("--players-per-team", type=int, default=11)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--games-per-trial", type=int, default=32)
    parser.add_argument("--final-games", type=int, default=128)
    parser.add_argument("--eval-envs", type=int, default=16)
    parser.add_argument("--game-length", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///experiments/helios_teacher_optuna.db",
    )
    parser.add_argument("--study-name", type=str, default="helios_teacher")
    parser.add_argument(
        "--best-config-json",
        type=Path,
        default=Path("experiments/helios_teacher_optuna_best.json"),
    )
    return parser


def evaluate_config(  # pylint: disable=too-many-arguments
    *,
    policy: Any,
    config: HeliosTeacherConfig,
    players_per_team: int,
    game_length: int,
    eval_envs: int,
    device: str,
    games: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate one Helios configuration against the loaded policy."""

    evaluator = PolicyVsHeliosEvaluator(
        players_per_team=players_per_team,
        game_length=game_length,
        eval_envs=eval_envs,
        device=device,
        teacher_config=config,
    )
    try:
        return evaluator.evaluate(policy, games=games, seed=seed)
    finally:
        evaluator.close()


def main() -> None:
    """Run the Optuna study, evaluate the best config, and save the result."""

    args = build_parser().parse_args()
    args.best_config_json.parent.mkdir(parents=True, exist_ok=True)
    device = train_pufferl.resolve_device(args.device)
    policy = load_checkpoint_policy(
        checkpoint_path=args.checkpoint_path,
        players_per_team=args.players_per_team,
        device=device,
    )
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        config = config_from_trial(trial)
        metrics = evaluate_config(
            policy=policy,
            config=config,
            players_per_team=args.players_per_team,
            game_length=args.game_length,
            eval_envs=args.eval_envs,
            device=device,
            games=args.games_per_trial,
            seed=args.seed + trial.number * 10_000,
        )
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("config", asdict(config))
        value = objective_from_metrics(metrics)
        print(
            f"trial={trial.number} objective={value:.4f} "
            f"win_rate={metrics['win_rate']:.3f} score_diff={metrics['score_diff']:.3f}"
        )
        return value

    study.optimize(objective, n_trials=args.trials)

    best_config = config_from_trial_attrs(study.best_trial)
    final_metrics = evaluate_config(
        policy=policy,
        config=best_config,
        players_per_team=args.players_per_team,
        game_length=args.game_length,
        eval_envs=args.eval_envs,
        device=device,
        games=args.final_games,
        seed=args.seed + 9_000_000,
    )
    payload = {
        "best_trial_number": study.best_trial.number,
        "best_trial_value": float(study.best_value),
        "best_trial_metrics": study.best_trial.user_attrs.get("metrics", {}),
        "final_metrics": final_metrics,
        "config": asdict(best_config),
        "checkpoint_path": str(args.checkpoint_path),
    }
    args.best_config_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("Best Optuna Helios config:")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(
        "Final tuned Helios evaluation: "
        f"games={int(final_metrics['games'])}, "
        f"policy_win_rate={final_metrics['win_rate']:.3f}, "
        f"policy_score_diff={final_metrics['score_diff']:.3f}"
    )


if __name__ == "__main__":
    main()
