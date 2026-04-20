"""GEPA evaluation pipeline for PuzzleScript environments.

Evaluates candidate PuzzleScript environments using a tiered strategy:
  Tier 1: Heuristic pre-filter (sub-millisecond per env)
  Tier 2: A* solver for solvability (10-500ms per env)
  Tier 3: Multi-level evaluation via BatchedEngine rollouts (optional)

Usage:
  # Curriculum mode: automatic 5 -> 10 game progression
  python scripts/gepa_evaluate.py --mode=curriculum

  # Benchmark mode: evaluate all train+val Sokoban games
  python scripts/gepa_evaluate.py --mode=benchmark

  # Evaluate a single game file
  python scripts/gepa_evaluate.py --mode=single --game=sokoban_basic

  # Evaluate a list of game files (simulates GEPA batch)
  python scripts/gepa_evaluate.py --mode=batch --games sokoban_basic sokoban_sanity Swap_Sokoban
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puffer_soccer.puzzle_evaluator import (
    PuzzleScriptEvaluator,
    QualityScore,
)

SCRIPT_DOCTOR_PATH = PROJECT_ROOT.parent / "script-doctor"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Curriculum: automatic phase progression (mirroring llm-desparsifier pattern)
CURRICULUM_PHASE_GAME_COUNTS = (5, 10)
PHASE_SOLVE_RATE_THRESHOLD = 0.80
PHASE_EARLY_STOP_PATIENCE = 3
DEFAULT_MAX_PHASE_ITERATIONS = 10


def load_sokoban_envs() -> dict:
    """Load the train/val Sokoban environment config."""
    config_path = CONFIGS_DIR / "sokoban_envs.json"
    with open(config_path) as f:
        return json.load(f)


def gepa_evaluate_candidates(
    evaluator: PuzzleScriptEvaluator,
    game_texts: dict[str, str],
    heuristic_threshold: float = 0.0,
    heuristic_filter_ratio: float = 0.5,
    solver_max_iters: int = 50_000,
    solver_timeout_ms: int = 5_000,
    max_workers: int = 8,
) -> list[QualityScore]:
    """Evaluate a batch of candidate PuzzleScript games.

    This is the core GEPA evaluation loop. Given N candidate game texts,
    it returns quality scores using a tiered approach:

    1. Compile all games (sequential, JS compiler is not thread-safe)
    2. Compute heuristic for all compiled games
    3. Filter bottom `heuristic_filter_ratio` by heuristic score
    4. Run A* solver on surviving candidates (parallel via threads)
    5. Return quality scores for all candidates

    Args:
        evaluator: PuzzleScriptEvaluator instance.
        game_texts: {name: game_text} dict of candidates.
        heuristic_threshold: Minimum heuristic to attempt solving.
        heuristic_filter_ratio: Fraction of candidates to filter out.
        solver_max_iters: Max iterations for A* solver.
        solver_timeout_ms: Timeout for A* solver.
        max_workers: Thread pool size for parallel solving.

    Returns:
        List of QualityScore for each candidate.
    """
    names = list(game_texts.keys())
    n = len(names)

    # --- Tier 1: Compile and compute heuristics ---
    t0 = time.perf_counter()
    compiled = {}  # name -> json_str
    infos = {}     # name -> game_info
    heuristics = {}  # name -> score

    for name in names:
        try:
            json_str = evaluator.compile_game(game_texts[name])
            compiled[name] = json_str
            infos[name] = evaluator.get_game_info(json_str)
            heuristics[name] = evaluator.compute_heuristic(json_str, level_i=0)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    compile_time = time.perf_counter() - t0
    print(f"  Tier 1: Compiled {len(compiled)}/{n} games in {compile_time:.3f}s")

    # --- Tier 2: Filter by heuristic ---
    if heuristics:
        sorted_by_heuristic = sorted(
            heuristics.items(), key=lambda x: x[1], reverse=True)
        n_keep = max(1, int(len(sorted_by_heuristic) * (1 - heuristic_filter_ratio)))
        survivors = {name for name, _ in sorted_by_heuristic[:n_keep]}
    else:
        survivors = set()

    print(f"  Tier 2: Heuristic filter kept {len(survivors)}/{len(compiled)} candidates")

    # --- Tier 3: Parallel A* solving ---
    t0 = time.perf_counter()
    solvability = {}

    survivor_list = [name for name in names if name in survivors]
    if survivor_list:
        json_strs = [compiled[name] for name in survivor_list]
        results = evaluator.evaluate_batch(
            json_strs,
            level_indices=[0] * len(json_strs),
            algo="astar",
            max_iters=solver_max_iters,
            timeout_ms=solver_timeout_ms,
            max_workers=max_workers,
        )
        for name, result in zip(survivor_list, results):
            solvability[name] = result

    solve_time = time.perf_counter() - t0
    solved_count = sum(1 for r in solvability.values() if r.solved)
    print(f"  Tier 3: Solved {solved_count}/{len(survivor_list)} in {solve_time:.3f}s")

    # --- Assemble quality scores ---
    max_solution_cap = 200
    scores = []
    for name in names:
        if name not in compiled:
            scores.append(QualityScore(
                game_name=name, compiled=False, n_levels=0,
                n_objects=0, grid_w=0, grid_h=0,
                heuristic_score=0.0, solvability=None, fitness=0.0,
            ))
            continue

        info = infos[name]
        h = heuristics.get(name, 0.0)
        solv = solvability.get(name)

        if solv and solv.solved:
            raw = 1.0 - (solv.solution_length / max_solution_cap)
            fitness = max(0.1, min(1.0, raw))
        elif solv and not solv.solved:
            fitness = h * 0.1
        else:
            # Not attempted (filtered by heuristic)
            fitness = h * 0.05

        scores.append(QualityScore(
            game_name=name,
            compiled=True,
            n_levels=info["n_levels"],
            n_objects=info["n_objects"],
            grid_w=info["grid_w"],
            grid_h=info["grid_h"],
            heuristic_score=h,
            solvability=solv,
            fitness=fitness,
        ))

    return scores


def run_benchmark(evaluator: PuzzleScriptEvaluator, n_games: int = 0,
                   split: str = "train"):
    """Benchmark mode: evaluate Sokoban environments.

    Args:
        evaluator: PuzzleScriptEvaluator instance.
        n_games: Number of train games to evaluate (0 = all train+val).
        split: Which split to use when n_games > 0 ("train" or "val").
    """
    config = load_sokoban_envs()

    all_games = {}
    if n_games > 0:
        entries = config[split][:n_games]
        for entry in entries:
            name = entry["name"]
            try:
                path = SCRIPT_DOCTOR_PATH / "data" / "scraped_games" / f"{name}.txt"
                if not path.exists():
                    path = SCRIPT_DOCTOR_PATH / "custom_games" / f"{name}.txt"
                all_games[name] = path.read_text()
            except Exception as e:
                print(f"  Could not load {name}: {e}")
    else:
        for s in ("train", "val"):
            for entry in config[s]:
                name = entry["name"]
                try:
                    path = SCRIPT_DOCTOR_PATH / "data" / "scraped_games" / f"{name}.txt"
                    if not path.exists():
                        path = SCRIPT_DOCTOR_PATH / "custom_games" / f"{name}.txt"
                    all_games[name] = path.read_text()
                except Exception as e:
                    print(f"  Could not load {name}: {e}")

    print(f"\nLoaded {len(all_games)} games for benchmark")
    print("=" * 70)

    # Full GEPA evaluation
    print("\n--- GEPA Full Pipeline (heuristic filter=50%, then A*) ---")
    t0 = time.perf_counter()
    scores = gepa_evaluate_candidates(
        evaluator, all_games,
        heuristic_filter_ratio=0.5,
        solver_max_iters=50_000,
        solver_timeout_ms=5_000,
        max_workers=8,
    )
    total_time = time.perf_counter() - t0

    print(f"\n  Total time: {total_time:.3f}s ({total_time/len(all_games):.3f}s/env)")
    print(f"\n  {'Game':<40} {'Fitness':>7} {'Solved':>7} {'SolLen':>7} {'Heur':>6}")
    print("  " + "-" * 70)
    for s in sorted(scores, key=lambda x: x.fitness, reverse=True):
        solved = "Y" if (s.solvability and s.solvability.solved) else "N"
        sol_len = s.solvability.solution_length if s.solvability else -1
        print(f"  {s.game_name:<40} {s.fitness:>7.3f} {solved:>7} "
              f"{sol_len:>7} {s.heuristic_score:>6.3f}")

    # Compare: sequential A* only (no heuristic filter)
    print("\n--- Comparison: Sequential A* Only ---")
    t0 = time.perf_counter()
    scores_seq = gepa_evaluate_candidates(
        evaluator, all_games,
        heuristic_filter_ratio=0.0,  # no filtering
        max_workers=1,  # sequential
    )
    seq_time = time.perf_counter() - t0
    print(f"  Total time: {seq_time:.3f}s ({seq_time/len(all_games):.3f}s/env)")

    # Compare: heuristic only
    print("\n--- Comparison: Heuristic Only ---")
    t0 = time.perf_counter()
    for name, text in all_games.items():
        try:
            json_str = evaluator.compile_game(text)
            evaluator.compute_heuristic(json_str)
        except Exception:
            pass
    heur_time = time.perf_counter() - t0
    print(f"  Total time: {heur_time:.3f}s ({heur_time/len(all_games):.3f}s/env)")

    print(f"\n--- Speedups vs Sequential A* ---")
    if seq_time > 0:
        print(f"  GEPA Pipeline: {seq_time/total_time:.1f}x faster")
        print(f"  Heuristic Only: {seq_time/heur_time:.0f}x faster")


def _load_game_texts(
    entries: list[dict],
    script_doctor_path: Path,
) -> dict[str, str]:
    """Load game texts from script-doctor for a list of config entries."""
    game_texts = {}
    for entry in entries:
        name = entry["name"]
        for subdir in ("data/scraped_games", "custom_games"):
            path = script_doctor_path / subdir / f"{name}.txt"
            if path.exists():
                game_texts[name] = path.read_text()
                break
        else:
            print(f"  [WARN] Game '{name}' not found in script-doctor")
    return game_texts


def run_curriculum(
    evaluator: PuzzleScriptEvaluator,
    max_phase_iterations: int = DEFAULT_MAX_PHASE_ITERATIONS,
    state_path: Path | None = None,
):
    """Curriculum mode: automatic progression from 5 -> 10 train games.

    Mirrors the llm-desparsifier GEPA curriculum pattern:
      - Phase 1: first 5 train games
      - Phase 2: all 10 train games
      - Advance when solve_rate >= 0.80
      - Early-stop if no improvement for 3 consecutive iterations
      - Final phase runs until iteration cap

    State is checkpointed to disk so the run can be resumed.
    """
    config = load_sokoban_envs()
    all_train_entries = config["train"]
    val_entries = config["val"]

    # Build phase schedule: cumulative slices of the train set
    phase_schedule = []
    for count in CURRICULUM_PHASE_GAME_COUNTS:
        phase_schedule.append(all_train_entries[:count])
    # Ensure the final phase includes all train games
    if len(all_train_entries) > CURRICULUM_PHASE_GAME_COUNTS[-1]:
        phase_schedule.append(all_train_entries)

    total_phases = len(phase_schedule)

    # Load or init curriculum state
    if state_path is None:
        state_path = PROJECT_ROOT / "experiments" / "gepa_curriculum_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if state_path.exists():
        with open(state_path) as f:
            curriculum_state = json.load(f)
        print(f"Resumed curriculum state from {state_path}")
    else:
        curriculum_state = {
            "current_phase": 1,
            "completed_phases": [],
            "phase_records": {},
            "total_phases": total_phases,
            "phase_game_counts": [len(p) for p in phase_schedule],
            "global_iteration": 0,
            "stop_reason": None,
        }

    current_phase = curriculum_state["current_phase"]
    global_iteration = curriculum_state["global_iteration"]

    print("=" * 70)
    print("GEPA Curriculum")
    print(f"  Phases: {[len(p) for p in phase_schedule]} games")
    print(f"  Solve-rate threshold: {PHASE_SOLVE_RATE_THRESHOLD}")
    print(f"  Early-stop patience: {PHASE_EARLY_STOP_PATIENCE}")
    print(f"  Max iterations/phase: {max_phase_iterations}")
    print(f"  Resuming at phase {current_phase}, global iter {global_iteration}")
    print("=" * 70)

    stop_reason = curriculum_state["stop_reason"]

    while stop_reason is None and current_phase <= total_phases:
        phase_entries = phase_schedule[current_phase - 1]
        phase_key = str(current_phase)
        n_games = len(phase_entries)
        is_final_phase = current_phase >= total_phases

        # Init phase record if needed
        phase_records = curriculum_state.setdefault("phase_records", {})
        if phase_key not in phase_records:
            phase_records[phase_key] = {
                "phase": current_phase,
                "n_games": n_games,
                "best_solve_rate": None,
                "best_mean_fitness": None,
                "non_improving_streak": 0,
                "iterations": 0,
                "advanced": False,
                "completed": False,
                "stop_reason": None,
                "iteration_results": [],
            }
        phase_record = phase_records[phase_key]
        phase_iteration = phase_record["iterations"]

        print(f"\n{'='*70}")
        print(f"Phase {current_phase}/{total_phases}: {n_games} games, "
              f"iteration {phase_iteration + 1}/{max_phase_iterations}")
        print(f"{'='*70}")

        # Load game texts for this phase
        game_texts = _load_game_texts(phase_entries, SCRIPT_DOCTOR_PATH)
        if not game_texts:
            print("  No games loaded — stopping.")
            stop_reason = "no_games"
            break

        # Run evaluation
        scores = gepa_evaluate_candidates(
            evaluator, game_texts,
            heuristic_filter_ratio=0.0,  # no filtering in curriculum
            max_workers=8,
        )

        # Compute metrics
        n_solved = sum(1 for s in scores if s.solvability and s.solvability.solved)
        solve_rate = n_solved / len(scores) if scores else 0.0
        mean_fitness = (sum(s.fitness for s in scores) / len(scores)) if scores else 0.0

        print(f"\n  Results: solve_rate={solve_rate:.3f}, "
              f"mean_fitness={mean_fitness:.3f}, "
              f"solved={n_solved}/{len(scores)}")

        # Print per-game results
        for s in sorted(scores, key=lambda x: x.fitness, reverse=True):
            solved = "Y" if (s.solvability and s.solvability.solved) else "N"
            sol_len = s.solvability.solution_length if s.solvability else -1
            print(f"    {s.game_name:<35} fitness={s.fitness:.3f} "
                  f"solved={solved} sol_len={sol_len}")

        # Track improvement
        improved = False
        best_fitness = phase_record["best_mean_fitness"]
        if best_fitness is None or mean_fitness > best_fitness:
            improved = True
            phase_record["best_mean_fitness"] = mean_fitness
            phase_record["non_improving_streak"] = 0
        else:
            phase_record["non_improving_streak"] += 1

        if phase_record["best_solve_rate"] is None or solve_rate > phase_record["best_solve_rate"]:
            phase_record["best_solve_rate"] = solve_rate

        phase_record["iterations"] += 1
        phase_iteration = phase_record["iterations"]
        global_iteration += 1
        curriculum_state["global_iteration"] = global_iteration

        # Store iteration result
        phase_record["iteration_results"].append({
            "iteration": phase_iteration,
            "solve_rate": solve_rate,
            "mean_fitness": mean_fitness,
            "n_solved": n_solved,
            "improved": improved,
        })

        # Phase advancement logic
        phase_advanced = False
        if not is_final_phase and solve_rate >= PHASE_SOLVE_RATE_THRESHOLD:
            phase_advanced = True
            phase_record["advanced"] = True
            phase_record["completed"] = True
            phase_record["stop_reason"] = "advanced_to_next_phase"
            if current_phase not in curriculum_state["completed_phases"]:
                curriculum_state["completed_phases"].append(current_phase)
            current_phase += 1
            curriculum_state["current_phase"] = current_phase
            print(f"\n  >>> Phase advanced! solve_rate={solve_rate:.3f} >= "
                  f"{PHASE_SOLVE_RATE_THRESHOLD}. Moving to phase {current_phase}.")
        elif is_final_phase:
            if phase_iteration >= max_phase_iterations:
                phase_record["completed"] = True
                phase_record["stop_reason"] = "phase_iteration_cap"
                stop_reason = "phase_iteration_cap"
        else:
            if phase_record["non_improving_streak"] >= PHASE_EARLY_STOP_PATIENCE:
                phase_record["completed"] = True
                phase_record["stop_reason"] = "threshold_failure_early_stop"
                stop_reason = "threshold_failure_early_stop"
                print(f"\n  >>> Early stop: {PHASE_EARLY_STOP_PATIENCE} iterations "
                      f"without improvement.")
            elif phase_iteration >= max_phase_iterations:
                phase_record["completed"] = True
                phase_record["stop_reason"] = "phase_iteration_cap"
                stop_reason = "phase_iteration_cap"

        curriculum_state["stop_reason"] = stop_reason

        # Checkpoint
        with open(state_path, "w") as f:
            json.dump(curriculum_state, f, indent=2)
        print(f"  Checkpointed to {state_path}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("Curriculum Complete")
    print(f"  Stop reason: {stop_reason or 'all phases completed'}")
    print(f"  Global iterations: {curriculum_state['global_iteration']}")
    print(f"  Completed phases: {curriculum_state['completed_phases']}")
    for pk, pr in curriculum_state.get("phase_records", {}).items():
        print(f"  Phase {pk}: best_solve_rate={pr['best_solve_rate']}, "
              f"best_fitness={pr['best_mean_fitness']}, "
              f"iters={pr['iterations']}, reason={pr['stop_reason']}")
    print(f"{'='*70}")

    # --- Holdout evaluation on val set ---
    print(f"\n--- Holdout Evaluation (val set, {len(val_entries)} games) ---")
    val_texts = _load_game_texts(val_entries, SCRIPT_DOCTOR_PATH)
    if val_texts:
        val_scores = gepa_evaluate_candidates(evaluator, val_texts,
                                               heuristic_filter_ratio=0.0)
        n_val_solved = sum(1 for s in val_scores if s.solvability and s.solvability.solved)
        val_solve_rate = n_val_solved / len(val_scores)
        val_mean_fitness = sum(s.fitness for s in val_scores) / len(val_scores)
        print(f"  Val solve_rate={val_solve_rate:.3f}, mean_fitness={val_mean_fitness:.3f}")
        for s in sorted(val_scores, key=lambda x: x.fitness, reverse=True):
            solved = "Y" if (s.solvability and s.solvability.solved) else "N"
            print(f"    {s.game_name:<35} fitness={s.fitness:.3f} solved={solved}")


def run_single(evaluator: PuzzleScriptEvaluator, game_name: str):
    """Evaluate a single game."""
    score = evaluator.score_candidate(
        evaluator.compile_game_file(game_name).__class__,  # placeholder
        game_name=game_name,
    )
    # Actually use the file-based approach
    path = SCRIPT_DOCTOR_PATH / "data" / "scraped_games" / f"{game_name}.txt"
    if not path.exists():
        path = SCRIPT_DOCTOR_PATH / "custom_games" / f"{game_name}.txt"
    game_text = path.read_text()
    score = evaluator.score_candidate(game_text, game_name=game_name)

    print(f"\nGame: {score.game_name}")
    print(f"  Compiled: {score.compiled}")
    print(f"  Levels: {score.n_levels}, Objects: {score.n_objects}")
    print(f"  Grid: {score.grid_w}x{score.grid_h}")
    print(f"  Heuristic: {score.heuristic_score:.3f}")
    if score.solvability:
        s = score.solvability
        print(f"  Solver: solved={s.solved}, iters={s.iterations}, "
              f"time={s.time_s:.3f}s, sol_len={s.solution_length}")
    print(f"  Fitness: {score.fitness:.3f}")


def run_batch(evaluator: PuzzleScriptEvaluator, game_names: list[str]):
    """Evaluate a batch of games."""
    game_texts = {}
    for name in game_names:
        path = SCRIPT_DOCTOR_PATH / "data" / "scraped_games" / f"{name}.txt"
        if not path.exists():
            path = SCRIPT_DOCTOR_PATH / "custom_games" / f"{name}.txt"
        game_texts[name] = path.read_text()

    scores = gepa_evaluate_candidates(evaluator, game_texts)
    print(f"\n{'Game':<40} {'Fitness':>7} {'Solved':>7}")
    print("-" * 60)
    for s in sorted(scores, key=lambda x: x.fitness, reverse=True):
        solved = "Y" if (s.solvability and s.solvability.solved) else "N"
        print(f"{s.game_name:<40} {s.fitness:>7.3f} {solved:>7}")


def main():
    parser = argparse.ArgumentParser(description="GEPA PuzzleScript evaluator")
    parser.add_argument("--mode",
                        choices=["curriculum", "benchmark", "single", "batch"],
                        default="curriculum")
    parser.add_argument("--game", type=str, help="Game name for single mode")
    parser.add_argument("--games", nargs="+", help="Game names for batch mode")
    parser.add_argument("--n-games", type=int, default=0,
                        help="Number of train games for benchmark mode (0=all)")
    parser.add_argument("--split", choices=["train", "val"], default="train",
                        help="Which split to use when --n-games > 0")
    parser.add_argument("--max-phase-iterations", type=int,
                        default=DEFAULT_MAX_PHASE_ITERATIONS,
                        help="Max iterations per curriculum phase")
    parser.add_argument("--script-doctor", type=str,
                        default=str(SCRIPT_DOCTOR_PATH),
                        help="Path to script-doctor repo")
    args = parser.parse_args()

    evaluator = PuzzleScriptEvaluator(args.script_doctor)

    if args.mode == "curriculum":
        run_curriculum(evaluator,
                       max_phase_iterations=args.max_phase_iterations)
    elif args.mode == "benchmark":
        run_benchmark(evaluator, n_games=args.n_games, split=args.split)
    elif args.mode == "single":
        if not args.game:
            parser.error("--game required for single mode")
        run_single(evaluator, args.game)
    elif args.mode == "batch":
        if not args.games:
            parser.error("--games required for batch mode")
        run_batch(evaluator, args.games)


if __name__ == "__main__":
    main()
