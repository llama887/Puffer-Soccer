"""SQLite database operations for the leaderboard server."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _tx(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        with self._tx() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    api_key_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS submissions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id),
                    filename TEXT NOT NULL,
                    policy_path TEXT NOT NULL,
                    elo_rating REAL NOT NULL DEFAULT 1000.0,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error_message TEXT,
                    submitted_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS matches (
                    id TEXT PRIMARY KEY,
                    submission_a_id TEXT NOT NULL REFERENCES submissions(id),
                    submission_b_id TEXT NOT NULL REFERENCES submissions(id),
                    games_played INTEGER NOT NULL,
                    wins_a INTEGER NOT NULL,
                    wins_b INTEGER NOT NULL,
                    draws INTEGER NOT NULL,
                    win_rate_a REAL NOT NULL,
                    score_diff REAL NOT NULL,
                    played_at TEXT NOT NULL
                );
            """)

    # ── Users ──

    def create_user(self, name: str, email: str) -> dict[str, str]:
        """Create a user and return {id, name, email, api_key, created_at}."""
        user_id = str(uuid.uuid4())
        api_key = f"sk_{secrets.token_urlsafe(32)}"
        now = _now()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO users (id, name, email, api_key_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, name, email, _hash_key(api_key), now),
            )
        return {"id": user_id, "name": name, "email": email, "api_key": api_key, "created_at": now}

    def get_user_by_api_key(self, api_key: str) -> dict[str, Any] | None:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT id, name, email, created_at FROM users WHERE api_key_hash = ?",
                (_hash_key(api_key),),
            ).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list[dict[str, Any]]:
        with self._tx() as conn:
            rows = conn.execute("SELECT id, name, email, created_at FROM users").fetchall()
        return [dict(r) for r in rows]

    def delete_user(self, user_id: str) -> bool:
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return cur.rowcount > 0

    def regenerate_api_key(self, user_id: str) -> str | None:
        api_key = f"sk_{secrets.token_urlsafe(32)}"
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE users SET api_key_hash = ? WHERE id = ?",
                (_hash_key(api_key), user_id),
            )
        return api_key if cur.rowcount > 0 else None

    # ── Submissions ──

    def create_submission(self, user_id: str, filename: str, policy_path: str) -> dict[str, Any]:
        sub_id = str(uuid.uuid4())
        now = _now()
        with self._tx() as conn:
            # Deactivate previous submissions from this user
            conn.execute(
                "UPDATE submissions SET is_active = 0 WHERE user_id = ? AND is_active = 1",
                (user_id,),
            )
            conn.execute(
                "INSERT INTO submissions (id, user_id, filename, policy_path, submitted_at) VALUES (?, ?, ?, ?, ?)",
                (sub_id, user_id, filename, policy_path, now),
            )
        return {
            "id": sub_id,
            "user_id": user_id,
            "filename": filename,
            "policy_path": policy_path,
            "elo_rating": 1000.0,
            "is_active": True,
            "status": "pending",
            "submitted_at": now,
        }

    def update_submission_status(self, sub_id: str, status: str, error_message: str | None = None):
        with self._tx() as conn:
            conn.execute(
                "UPDATE submissions SET status = ?, error_message = ? WHERE id = ?",
                (status, error_message, sub_id),
            )

    def update_submission_elo(self, sub_id: str, elo: float):
        with self._tx() as conn:
            conn.execute(
                "UPDATE submissions SET elo_rating = ? WHERE id = ?",
                (elo, sub_id),
            )

    def get_submission(self, sub_id: str) -> dict[str, Any] | None:
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM submissions WHERE id = ?", (sub_id,)).fetchone()
        return dict(row) if row else None

    def get_active_submissions(self) -> list[dict[str, Any]]:
        """Get all active submissions, one per user, ordered by ELO descending."""
        with self._tx() as conn:
            rows = conn.execute(
                """SELECT s.*, u.name as user_name, u.email as user_email
                   FROM submissions s JOIN users u ON s.user_id = u.id
                   WHERE s.is_active = 1 AND s.status = 'ready'
                   ORDER BY s.elo_rating DESC""",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_active_submissions_except(self, exclude_id: str) -> list[dict[str, Any]]:
        """Get all active+ready submissions except the given one."""
        with self._tx() as conn:
            rows = conn.execute(
                """SELECT * FROM submissions
                   WHERE is_active = 1 AND status = 'ready' AND id != ?""",
                (exclude_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_user_submissions(self, user_id: str) -> list[dict[str, Any]]:
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT * FROM submissions WHERE user_id = ? ORDER BY submitted_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Matches ──

    def record_match(
        self,
        sub_a_id: str,
        sub_b_id: str,
        games_played: int,
        wins_a: int,
        wins_b: int,
        draws: int,
        win_rate_a: float,
        score_diff: float,
    ) -> dict[str, Any]:
        match_id = str(uuid.uuid4())
        now = _now()
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO matches
                   (id, submission_a_id, submission_b_id, games_played, wins_a, wins_b, draws, win_rate_a, score_diff, played_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (match_id, sub_a_id, sub_b_id, games_played, wins_a, wins_b, draws, win_rate_a, score_diff, now),
            )
        return {
            "id": match_id,
            "submission_a_id": sub_a_id,
            "submission_b_id": sub_b_id,
            "games_played": games_played,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "win_rate_a": win_rate_a,
            "score_diff": score_diff,
            "played_at": now,
        }

    def get_matches_for_submission(self, sub_id: str) -> list[dict[str, Any]]:
        with self._tx() as conn:
            rows = conn.execute(
                """SELECT m.*,
                    ua.name as user_a_name, ub.name as user_b_name
                   FROM matches m
                   JOIN submissions sa ON m.submission_a_id = sa.id
                   JOIN submissions sb ON m.submission_b_id = sb.id
                   JOIN users ua ON sa.user_id = ua.id
                   JOIN users ub ON sb.user_id = ub.id
                   WHERE m.submission_a_id = ? OR m.submission_b_id = ?
                   ORDER BY m.played_at DESC""",
                (sub_id, sub_id),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Get leaderboard: active+ready submissions ranked by ELO."""
        with self._tx() as conn:
            rows = conn.execute(
                """SELECT s.id, s.elo_rating, s.submitted_at, s.filename,
                          u.name as user_name, u.email as user_email,
                          (SELECT COUNT(*) FROM matches m
                           WHERE (m.submission_a_id = s.id AND m.win_rate_a > 0.5)
                              OR (m.submission_b_id = s.id AND m.win_rate_a < 0.5)) as wins,
                          (SELECT COUNT(*) FROM matches m
                           WHERE (m.submission_a_id = s.id AND m.win_rate_a < 0.5)
                              OR (m.submission_b_id = s.id AND m.win_rate_a > 0.5)) as losses,
                          (SELECT COUNT(*) FROM matches m
                           WHERE (m.submission_a_id = s.id OR m.submission_b_id = s.id)
                             AND m.win_rate_a = 0.5) as draws
                   FROM submissions s
                   JOIN users u ON s.user_id = u.id
                   WHERE s.is_active = 1 AND s.status = 'ready'
                   ORDER BY s.elo_rating DESC""",
            ).fetchall()
        return [dict(r) for r in rows]

    def recalculate_all_elo(self, initial_elo: float, k_factor: float):
        """Recalculate all ELO ratings from match history (for consistency)."""
        from server.elo import update_elo

        with self._tx() as conn:
            # Reset all active submissions to initial ELO
            conn.execute(
                "UPDATE submissions SET elo_rating = ? WHERE is_active = 1 AND status = 'ready'",
                (initial_elo,),
            )
            # Replay all matches in chronological order
            matches = conn.execute(
                "SELECT * FROM matches ORDER BY played_at ASC"
            ).fetchall()

            ratings: dict[str, float] = {}
            for m in matches:
                a_id = m["submission_a_id"]
                b_id = m["submission_b_id"]
                if a_id not in ratings:
                    ratings[a_id] = initial_elo
                if b_id not in ratings:
                    ratings[b_id] = initial_elo

                new_a, new_b = update_elo(
                    ratings[a_id], ratings[b_id], m["win_rate_a"], k=k_factor
                )
                ratings[a_id] = new_a
                ratings[b_id] = new_b

            for sub_id, elo in ratings.items():
                conn.execute(
                    "UPDATE submissions SET elo_rating = ? WHERE id = ?",
                    (elo, sub_id),
                )
