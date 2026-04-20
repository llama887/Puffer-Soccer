"""Integration tests for the PuzzleScript evaluator and GEPA pipeline.

Requires script-doctor repo with C++ extension built and Node.js available.
Run with: python -m pytest tests/test_puzzle_evaluator.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Add project source to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from puffer_soccer.puzzle_evaluator import (
    PuzzleScriptEvaluator,
    QualityScore,
    SolvabilityResult,
    RolloutResult,
)

SCRIPT_DOCTOR_PATH = PROJECT_ROOT.parent / "script-doctor"
NODE_PATH = Path("/scratch/fyy2003/node-v20.18.0-linux-x64/bin")

# Ensure Node.js is on PATH for all tests
if NODE_PATH.exists():
    os.environ["PATH"] = str(NODE_PATH) + ":" + os.environ.get("PATH", "")


def _skip_if_no_deps():
    """Skip tests if script-doctor or Node.js aren't available."""
    if not SCRIPT_DOCTOR_PATH.exists():
        pytest.skip("script-doctor repo not found")
    so_files = list(SCRIPT_DOCTOR_PATH.glob("puzzlescript_cpp/_puzzlescript_cpp*.so"))
    if not so_files:
        pytest.skip("C++ extension not built")
    if not NODE_PATH.exists() and not any(
        (Path(p) / "node").exists() for p in os.environ.get("PATH", "").split(":")
    ):
        pytest.skip("Node.js not available")


@pytest.fixture(scope="module")
def evaluator():
    _skip_if_no_deps()
    return PuzzleScriptEvaluator(SCRIPT_DOCTOR_PATH)


@pytest.fixture(scope="module")
def sokoban_basic_json(evaluator):
    return evaluator.compile_game_file("sokoban_basic")


@pytest.fixture(scope="module")
def sokoban_basic_text():
    path = SCRIPT_DOCTOR_PATH / "data" / "scraped_games" / "sokoban_basic.txt"
    return path.read_text()


# --- Test: Game Compilation ---

class TestCompilation:
    def test_compile_valid_game(self, evaluator, sokoban_basic_json):
        """Valid game compiles to non-empty JSON."""
        assert len(sokoban_basic_json) > 100
        parsed = json.loads(sokoban_basic_json)
        assert "objectCount" in parsed
        assert "levels" in parsed

    def test_compile_empty_game(self, evaluator):
        """Empty game text still returns JSON (JS compiler is lenient).
        The JS compiler does not raise on invalid input — it silently
        produces a (possibly stale) compiled state. This is a known
        limitation of the JS backend."""
        # Just verify it doesn't crash; the JS compiler is lenient
        json_str = evaluator.compile_game("")
        assert isinstance(json_str, str)

    def test_compile_game_file(self, evaluator):
        """compile_game_file finds and compiles a named game."""
        json_str = evaluator.compile_game_file("sokoban_sanity")
        assert len(json_str) > 100

    def test_compile_game_file_not_found(self, evaluator):
        """compile_game_file raises for nonexistent game."""
        with pytest.raises(FileNotFoundError):
            evaluator.compile_game_file("definitely_not_a_real_game_xyz")


# --- Test: Game Info ---

class TestGameInfo:
    def test_get_game_info(self, evaluator, sokoban_basic_json):
        info = evaluator.get_game_info(sokoban_basic_json)
        assert info["n_levels"] == 2
        assert info["n_objects"] == 5
        assert info["grid_w"] > 0
        assert info["grid_h"] > 0


# --- Test: A* Solver ---

class TestSolvability:
    def test_solve_easy_game(self, evaluator, sokoban_basic_json):
        """A* should solve sokoban_basic level 0."""
        result = evaluator.evaluate_solvability(sokoban_basic_json, level_i=0)
        assert isinstance(result, SolvabilityResult)
        assert result.solved is True
        assert result.solution_length > 0
        assert result.iterations > 0
        assert result.time_s > 0
        assert result.fps > 0

    def test_solve_with_bfs(self, evaluator, sokoban_basic_json):
        """BFS should also solve it."""
        result = evaluator.evaluate_solvability(
            sokoban_basic_json, level_i=0, algo="bfs")
        assert result.solved is True

    def test_solve_with_gbfs(self, evaluator, sokoban_basic_json):
        """GBFS should also solve it."""
        result = evaluator.evaluate_solvability(
            sokoban_basic_json, level_i=0, algo="gbfs")
        assert result.solved is True

    def test_solve_with_timeout(self, evaluator, sokoban_basic_json):
        """Very small iteration budget may fail to solve."""
        result = evaluator.evaluate_solvability(
            sokoban_basic_json, level_i=0, max_iters=5, timeout_ms=1)
        # May or may not solve with only 5 iterations, but should not crash
        assert isinstance(result, SolvabilityResult)


# --- Test: Batch Evaluation ---

class TestBatchEval:
    def test_evaluate_batch_sequential(self, evaluator):
        """Batch eval with max_workers=1 runs sequentially."""
        games = ["sokoban_basic", "sokoban_sanity"]
        json_strs = [evaluator.compile_game_file(g) for g in games]
        results = evaluator.evaluate_batch(json_strs, max_workers=1)
        assert len(results) == 2
        assert all(isinstance(r, SolvabilityResult) for r in results)
        assert all(r.solved for r in results)

    def test_evaluate_batch_parallel(self, evaluator):
        """Batch eval with max_workers>1 uses threads."""
        games = ["sokoban_basic", "sokoban_sanity", "Broken_Leg_Sokoban"]
        json_strs = [evaluator.compile_game_file(g) for g in games]
        results = evaluator.evaluate_batch(json_strs, max_workers=4)
        assert len(results) == 3
        assert all(r.solved for r in results)


# --- Test: BatchedEngine Random Rollouts ---

class TestBatchedRollouts:
    def test_random_rollout(self, evaluator, sokoban_basic_json):
        """BatchedEngine random rollout runs without error."""
        result = evaluator.evaluate_batch_random_rollout(
            sokoban_basic_json, n_rollouts=8, n_steps=50)
        assert isinstance(result, RolloutResult)
        assert result.n_rollouts == 8
        assert result.n_steps == 50
        assert result.total_time_s > 0
        assert result.steps_per_sec > 0


# --- Test: Heuristic ---

class TestHeuristic:
    def test_heuristic_score(self, evaluator, sokoban_basic_json):
        """Heuristic returns a value in [0, 1]."""
        score = evaluator.compute_heuristic(sokoban_basic_json, level_i=0)
        assert 0.0 <= score <= 1.0


# --- Test: Full GEPA Scoring ---

class TestScoreCandidate:
    def test_score_valid_game(self, evaluator, sokoban_basic_text):
        """Full scoring pipeline on a valid, solvable game."""
        score = evaluator.score_candidate(
            sokoban_basic_text, game_name="sokoban_basic")
        assert isinstance(score, QualityScore)
        assert score.compiled is True
        assert score.n_levels == 2
        assert score.fitness > 0
        assert score.solvability is not None
        assert score.solvability.solved is True

    def test_score_empty_game(self, evaluator):
        """Empty game text: JS compiler is lenient so it may 'compile'
        but the resulting game should still get scored (possibly with
        stale state). This tests that the pipeline doesn't crash."""
        score = evaluator.score_candidate("", game_name="empty")
        assert isinstance(score, QualityScore)
        # The JS compiler may reuse stale state, so compiled could be True
        # The key invariant is that the pipeline completes without error


# --- Test: Train/Val Config ---

class TestConfig:
    def test_config_loads(self):
        config_path = PROJECT_ROOT / "configs" / "sokoban_envs.json"
        with open(config_path) as f:
            config = json.load(f)
        assert "train" in config
        assert "val" in config
        assert len(config["train"]) == 10
        assert len(config["val"]) == 6

    def test_no_overlap(self):
        config_path = PROJECT_ROOT / "configs" / "sokoban_envs.json"
        with open(config_path) as f:
            config = json.load(f)
        train_names = {e["name"] for e in config["train"]}
        val_names = {e["name"] for e in config["val"]}
        assert train_names.isdisjoint(val_names), "Train and val sets must not overlap"

    def test_all_games_exist(self):
        """All configured games exist in script-doctor."""
        if not SCRIPT_DOCTOR_PATH.exists():
            pytest.skip("script-doctor not available")
        config_path = PROJECT_ROOT / "configs" / "sokoban_envs.json"
        with open(config_path) as f:
            config = json.load(f)
        for split in ("train", "val"):
            for entry in config[split]:
                name = entry["name"]
                scraped = SCRIPT_DOCTOR_PATH / "data" / "scraped_games" / f"{name}.txt"
                custom = SCRIPT_DOCTOR_PATH / "custom_games" / f"{name}.txt"
                assert scraped.exists() or custom.exists(), \
                    f"Game '{name}' not found in script-doctor"
