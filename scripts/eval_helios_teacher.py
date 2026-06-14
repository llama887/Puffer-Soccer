"""Record and score the Helios-inspired scripted teacher in the MARL2D env."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from puffer_soccer.envs.marl2d import make_puffer_env
from puffer_soccer.helios_teacher import HeliosTeacher


def run_teacher_episode(
    *,
    players_per_team: int,
    game_length: int,
    seed: int,
    video_path: Path | None,
    video_fps: int,
) -> tuple[int, int] | None:
    """Run one deterministic teacher self-play episode and optionally save an mp4.

    This helper is intentionally separate from the CLI parser so tests and future experiment
    scripts can reuse the exact same rollout path. The teacher controls both teams from the
    public environment state, so the resulting video answers the first implementation question:
    does the Helios approximation produce recognizable soccer spacing before we spend training
    budget on imitation or scripted-opponent experiments?

    The return value is the completed ``(blue_goals, red_goals)`` score when the native env has
    exposed one at episode end. It can be ``None`` for very short smoke runs that stop before a
    terminal score is available, which keeps the function useful for quick rendering checks.
    """

    env = make_puffer_env(
        players_per_team=players_per_team,
        action_mode="discrete",
        game_length=game_length,
        render_mode="rgb_array" if video_path is not None else None,
        seed=seed,
        opponents_enabled=True,
    )
    teacher = HeliosTeacher(players_per_team)
    frames: list[np.ndarray] = []
    env.reset(seed=seed)
    try:
        for _ in range(game_length):
            if video_path is not None:
                frame = env.render()
                if frame is not None:
                    frames.append(frame.astype(np.uint8, copy=False))
            actions = teacher.actions(env.get_state())
            _, _, terminals, truncations, _ = env.step(actions)
            if bool(np.all(terminals) or np.all(truncations)):
                break

        if video_path is not None:
            frame = env.render()
            if frame is not None:
                frames.append(frame.astype(np.uint8, copy=False))
    finally:
        scores = env.get_last_episode_scores(clear=False)
        env.close()

    if video_path is not None and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=video_fps)  # type: ignore[arg-type]
    return scores


def run_teacher_eval(
    *,
    players_per_team: int,
    game_length: int,
    games: int,
    seed: int,
) -> dict[str, float]:
    """Return aggregate self-play scoring stats for the scripted teacher.

    Teacher self-play is symmetric, so this is not a strength benchmark. It is a regression
    guard and a sanity check: games should run to completion, scores should be finite, and
    future changes to the teacher should have a visible effect on goals per game. The useful
    strategic measurement still comes from replay videos and the existing teamplay trace tools,
    but this cheap scalar summary makes command-line runs easier to compare.
    """

    scores: list[tuple[int, int]] = []
    for game_idx in range(games):
        score = run_teacher_episode(
            players_per_team=players_per_team,
            game_length=game_length,
            seed=seed + game_idx,
            video_path=None,
            video_fps=20,
        )
        if score is not None:
            scores.append(score)

    if not scores:
        return {
            "games": 0.0,
            "blue_goals_per_game": 0.0,
            "red_goals_per_game": 0.0,
            "abs_score_diff_per_game": 0.0,
        }

    score_arr = np.asarray(scores, dtype=np.float32)
    return {
        "games": float(len(scores)),
        "blue_goals_per_game": float(np.mean(score_arr[:, 0])),
        "red_goals_per_game": float(np.mean(score_arr[:, 1])),
        "abs_score_diff_per_game": float(np.mean(np.abs(score_arr[:, 0] - score_arr[:, 1]))),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the teacher evaluation utility.

    The defaults are chosen for a quick local GPU-node or login-node smoke check rather than a
    full research run. The script does not use a neural policy, so no GPU or W&B run is needed.
    It still follows the repository convention that Python commands are launched through
    ``uv run`` by keeping all behavior inside this ordinary script entry point.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players-per-team", type=int, default=11)
    parser.add_argument("--game-length", type=int, default=400)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--video-output",
        type=Path,
        default=Path("videos/helios_teacher_selfplay.mp4"),
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """Run the requested teacher video and scalar evaluation from the command line."""

    args = parse_args()
    video_path = None if args.no_video else args.video_output
    first_score = run_teacher_episode(
        players_per_team=args.players_per_team,
        game_length=args.game_length,
        seed=args.seed,
        video_path=video_path,
        video_fps=args.video_fps,
    )
    metrics = run_teacher_eval(
        players_per_team=args.players_per_team,
        game_length=args.game_length,
        games=args.games,
        seed=args.seed + 10_000,
    )
    print(f"first_episode_score={first_score}")
    for key, value in metrics.items():
        print(f"{key}={value:.6g}")
    if video_path is not None:
        print(f"video={video_path}")


if __name__ == "__main__":
    main()
