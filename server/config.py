"""Server configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
POLICIES_DIR = BASE_DIR / "policies"
POLICIES_DIR.mkdir(exist_ok=True)
DB_PATH = Path(os.environ.get("DB_PATH", str(BASE_DIR / "leaderboard.db")))

# Admin
ADMIN_KEY = os.environ.get("ADMIN_KEY", "admin-change-me")

# Evaluation settings
PLAYERS_PER_TEAM = int(os.environ.get("PLAYERS_PER_TEAM", "2"))
GAME_LENGTH = int(os.environ.get("GAME_LENGTH", "400"))
GAMES_PER_MATCHUP = int(os.environ.get("GAMES_PER_MATCHUP", "20"))
EVAL_NUM_ENVS = int(os.environ.get("EVAL_NUM_ENVS", "4"))

# ELO settings
INITIAL_ELO = 1000.0
K_FACTOR = float(os.environ.get("K_FACTOR", "32"))
