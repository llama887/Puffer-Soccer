"""Plot GEPA holdout comparison: Optimized Prompt vs Built-in Heuristic vs Blind A*.

Compares three A* search conditions on the 6-game holdout (validation) set:
  1. GEPA Optimized Prompt — LLM-generated heuristic from best_heuristic.py
  2. Built-in Heuristic — engine.get_score_normalized() (the naive distance heuristic)
  3. Blind A* (h=0) — no heuristic guidance

Score metric: ((max_expansions + 1) - expanded_states) / (max_expansions + 1)
  where max_expansions = 50,000. Higher = fewer expansions = better heuristic.

Data sources:
  - Optimized: SLURM log llm-desparsifier-6217968.out (holdout section)
  - Baselines: computed live via --script-doctor, or use --cached-only

Usage:
  # Compute baselines live (needs script-doctor + C++ extension + Node.js)
  python scripts/plot_gepa_holdout_comparison.py \\
      --script-doctor /scratch/fyy2003/repos/script-doctor

  # Use only the optimized results from the log (no deps needed)
  python scripts/plot_gepa_holdout_comparison.py --cached-only

  # Save to file
  python scripts/plot_gepa_holdout_comparison.py --cached-only -o gepa_comparison.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_DESPARSIFIER_ROOT = PROJECT_ROOT.parent / "llm-desparsifier"
SCRIPT_DOCTOR_DEFAULT = PROJECT_ROOT.parent / "script-doctor"

MAX_EXPANSIONS = 50_000

# ---------------------------------------------------------------------------
# Holdout games (from configs/gepa_puzzlescript_envs.yaml eval_jobs)
# ---------------------------------------------------------------------------
HOLDOUT_GAMES = [
    "sokoban_sanity",
    "No_Right_Turn_Sokoban",
    "Cold_Feet_Sokoban",
    "Soko-bine",
    "Remote_Control_Sokoban",
    "Darkness_Sokoban",
]

# ---------------------------------------------------------------------------
# GEPA Optimized Prompt results on holdout
# Source: llm-desparsifier SLURM log (job 6217968, Apr 12 2026)
# Heuristic: best_heuristic.py (GEPA-evolved via gemini-3-pro-preview)
# max_expansions=50000, score = ((N+1) - expanded) / (N+1)
# ---------------------------------------------------------------------------
OPTIMIZED_HOLDOUT = {
    "sokoban_sanity":          {"score": 0.9937, "solved": True},
    "No_Right_Turn_Sokoban":   {"score": 0.9980, "solved": True},
    "Cold_Feet_Sokoban":       {"score": 0.8423, "solved": True},
    "Soko-bine":               {"score": 0.9888, "solved": True},
    "Remote_Control_Sokoban":  {"score": 0.9997, "solved": True},
    "Darkness_Sokoban":        {"score": 0.9925, "solved": True},
}

# ---------------------------------------------------------------------------
# Baselines on phase-1 train set (from same log, for reference)
# ---------------------------------------------------------------------------
BLIND_TRAIN_PHASE1 = {
    "sokoban_basic":       {"score": 0.9888, "expanded": 559},
    "Broken_Leg_Sokoban":  {"score": 0.9968, "expanded": 161},
    "Collapsable_Sokoban": {"score": 0.9926, "expanded": 369},
    "Pulling_Box_Sokoban": {"score": 0.9953, "expanded": 233},
    "Swap_Sokoban":        {"score": 0.9835, "expanded": 823},
}

BUILTIN_TRAIN_PHASE1 = {
    "sokoban_basic":       {"score": 0.9892, "expanded": 541},
    "Broken_Leg_Sokoban":  {"score": 0.9980, "expanded": 99},
    "Collapsable_Sokoban": {"score": 0.9932, "expanded": 341},
    "Pulling_Box_Sokoban": {"score": 0.9960, "expanded": 200},
    "Swap_Sokoban":        {"score": 0.9871, "expanded": 646},
}


def gepa_score(solved: bool, expanded: int, max_exp: int = MAX_EXPANSIONS) -> float:
    """Score: ((N+1) - S) / (N+1). Higher = more efficient search."""
    n = max_exp
    s = expanded if solved else n + 1
    return ((n + 1) - s) / (n + 1)


def _load_cpp_module(script_doctor_path: Path):
    """Load the C++ PuzzleScript extension directly."""
    so_files = list(script_doctor_path.glob("puzzlescript_cpp/_puzzlescript_cpp*.so"))
    if not so_files:
        raise FileNotFoundError(
            f"C++ extension not found in {script_doctor_path}/puzzlescript_cpp/")
    spec = importlib.util.spec_from_file_location(
        "puzzlescript_cpp._puzzlescript_cpp", str(so_files[0]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate_holdout_baselines(
    script_doctor_path: Path,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Evaluate blind A* and built-in heuristic on holdout games.

    Returns (blind_results, builtin_results) where each is
    {game_name: {"score": float, "solved": bool, "expanded": int}}.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from puffer_soccer.puzzle_evaluator import PuzzleScriptEvaluator

    evaluator = PuzzleScriptEvaluator(str(script_doctor_path))
    cpp = _load_cpp_module(script_doctor_path)

    blind_results = {}
    builtin_results = {}

    for name in HOLDOUT_GAMES:
        game_text = None
        for subdir in ("data/scraped_games", "custom_games"):
            path = script_doctor_path / subdir / f"{name}.txt"
            if path.exists():
                game_text = path.read_text()
                break
        if game_text is None:
            print(f"  [WARN] {name} not found, skipping")
            blind_results[name] = {"score": 0.0, "solved": False, "expanded": 0}
            builtin_results[name] = {"score": 0.0, "solved": False, "expanded": 0}
            continue

        json_str = evaluator.compile_game(game_text)
        engine = evaluator.load_engine(json_str)
        engine.load_level(0)

        # Blind A* (h=0): uses BFS-like expansion
        raw_bfs = cpp.solve_bfs(engine, MAX_EXPANSIONS, -1)
        blind_results[name] = {
            "score": gepa_score(raw_bfs.won, raw_bfs.iterations),
            "solved": raw_bfs.won,
            "expanded": raw_bfs.iterations,
        }

        # Built-in heuristic: uses engine's A* with get_score_normalized
        engine.load_level(0)
        raw_astar = cpp.solve_astar(engine, MAX_EXPANSIONS, -1)
        builtin_results[name] = {
            "score": gepa_score(raw_astar.won, raw_astar.iterations),
            "solved": raw_astar.won,
            "expanded": raw_astar.iterations,
        }

        print(f"  {name:40s} blind: expanded={raw_bfs.iterations:6d} "
              f"score={blind_results[name]['score']:.4f} | "
              f"builtin: expanded={raw_astar.iterations:6d} "
              f"score={builtin_results[name]['score']:.4f}")

    return blind_results, builtin_results


def make_plot(
    optimized: dict[str, dict],
    builtin: dict[str, dict],
    blind: dict[str, dict],
    output_path: Optional[str] = None,
):
    """Create grouped bar chart comparing three A* search conditions."""
    games = HOLDOUT_GAMES
    short_names = [
        g.replace("_Sokoban", "").replace("_", "\n") for g in games
    ]

    n = len(games)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6.5))

    # Colors
    opt_color = "#2196F3"     # blue
    builtin_color = "#FF9800" # orange
    blind_color = "#9E9E9E"   # gray

    opt_vals = [optimized[g]["score"] for g in games]
    builtin_vals = [builtin[g]["score"] for g in games]
    blind_vals = [blind[g]["score"] for g in games]

    bars_opt = ax.bar(x - width, opt_vals, width,
                      label="GEPA Optimized Prompt", color=opt_color,
                      edgecolor="white", linewidth=0.5)
    bars_builtin = ax.bar(x, builtin_vals, width,
                          label="Built-in Heuristic (naive)", color=builtin_color,
                          edgecolor="white", linewidth=0.5)
    bars_blind = ax.bar(x + width, blind_vals, width,
                        label="Blind A* (h=0)", color=blind_color,
                        edgecolor="white", linewidth=0.5)

    # Value labels
    for bars in (bars_opt, bars_builtin, bars_blind):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7.5,
                        rotation=0)

    # Mean lines
    opt_mean = np.mean(opt_vals)
    builtin_mean = np.mean(builtin_vals)
    blind_mean = np.mean(blind_vals)

    ax.axhline(opt_mean, color=opt_color, linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(builtin_mean, color=builtin_color, linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(blind_mean, color=blind_color, linestyle="--", alpha=0.4, linewidth=1)

    # Mean annotations on right
    y_annot = sorted([
        (opt_mean, "Optimized", opt_color),
        (builtin_mean, "Built-in", builtin_color),
        (blind_mean, "Blind", blind_color),
    ], key=lambda t: t[0], reverse=True)
    for i, (val, label, color) in enumerate(y_annot):
        ax.text(n - 0.3, val + 0.005 - i * 0.015, f"{label} mean: {val:.4f}",
                color=color, fontsize=9, ha="right", va="bottom")

    # Formatting
    ax.set_xlabel("Holdout Game", fontsize=12, labelpad=10)
    ax.set_ylabel("Search Efficiency Score", fontsize=12)
    ax.set_title(
        "GEPA Holdout: Optimized Prompt vs Built-in Heuristic vs Blind A*",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9.5, ha="center")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Solve-rate subtitle
    opt_sr = sum(1 for g in games if optimized[g]["solved"]) / n
    builtin_sr = sum(1 for g in games if builtin[g]["solved"]) / n
    blind_sr = sum(1 for g in games if blind[g]["solved"]) / n
    subtitle = (
        f"Solve rates: Optimized {opt_sr:.0%} | "
        f"Built-in {builtin_sr:.0%} | Blind {blind_sr:.0%}   "
        f"[max_expansions={MAX_EXPANSIONS:,}]"
    )
    ax.text(0.5, -0.10, subtitle, transform=ax.transAxes, ha="center",
            fontsize=10, color="#555555")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved to {output_path}")
    else:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot GEPA holdout comparison: optimized vs builtin vs blind")
    parser.add_argument("--script-doctor", type=Path,
                        default=SCRIPT_DOCTOR_DEFAULT,
                        help="Path to script-doctor repo (for baseline eval)")
    parser.add_argument("--cached-only", action="store_true",
                        help="Skip live evaluation, use only logged data. "
                             "Baselines will show as zero for holdout games.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file path (e.g. gepa_comparison.png)")
    args = parser.parse_args()

    # --- Optimized prompt results (from SLURM log) ---
    optimized = OPTIMIZED_HOLDOUT

    # --- Baseline results ---
    if args.cached_only:
        print("Cached-only mode: baselines need live evaluation.")
        print("Using logged train-set baselines as rough proxies.\n")
        # We don't have holdout baselines cached; use zeros as placeholders
        blind = {g: {"score": 0.0, "solved": False, "expanded": 0}
                 for g in HOLDOUT_GAMES}
        builtin = {g: {"score": 0.0, "solved": False, "expanded": 0}
                   for g in HOLDOUT_GAMES}
        print("Run without --cached-only to compute baselines on holdout games.")
    else:
        print("Evaluating blind A* and built-in heuristic on holdout games...")
        print(f"  script-doctor: {args.script_doctor}")
        print(f"  max_expansions: {MAX_EXPANSIONS:,}\n")
        blind, builtin = evaluate_holdout_baselines(args.script_doctor)

    # --- Print summary ---
    print(f"\n{'Game':<40} {'Optimized':>10} {'Built-in':>10} {'Blind':>10}")
    print("-" * 72)
    for g in HOLDOUT_GAMES:
        print(f"{g:<40} {optimized[g]['score']:>10.4f} "
              f"{builtin[g]['score']:>10.4f} {blind[g]['score']:>10.4f}")
    print("-" * 72)
    opt_mean = np.mean([optimized[g]["score"] for g in HOLDOUT_GAMES])
    blt_mean = np.mean([builtin[g]["score"] for g in HOLDOUT_GAMES])
    bld_mean = np.mean([blind[g]["score"] for g in HOLDOUT_GAMES])
    print(f"{'Mean':<40} {opt_mean:>10.4f} {blt_mean:>10.4f} {bld_mean:>10.4f}")

    # --- Plot ---
    make_plot(optimized, builtin, blind, output_path=args.output)


if __name__ == "__main__":
    main()
