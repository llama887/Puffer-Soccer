"""Puffer Soccer Leaderboard Server.

A FastAPI server that accepts policy submissions, runs round-robin tournaments
using head-to-head soccer matches, and maintains an ELO-based leaderboard.
"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from server.config import (
    ADMIN_KEY,
    DB_PATH,
    GAMES_PER_MATCHUP,
    INITIAL_ELO,
    K_FACTOR,
    PLAYERS_PER_TEAM,
    POLICIES_DIR,
)
from server.database import Database
from server.evaluator import run_round_robin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Puffer Soccer Leaderboard",
    description="Submit soccer policies and compete on the ELO leaderboard",
    version="1.0.0",
)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

db = Database(DB_PATH)

# Lock to serialize evaluations (one match at a time)
_eval_lock = threading.Lock()


# ── Pydantic models ──


class UserCreate(BaseModel):
    name: str
    email: str


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    api_key: str
    created_at: str


class SubmissionResponse(BaseModel):
    id: str
    user_id: str
    filename: str
    elo_rating: float
    status: str
    submitted_at: str


class LeaderboardEntry(BaseModel):
    rank: int
    user_name: str
    elo_rating: float
    wins: int
    losses: int
    draws: int
    submitted_at: str


class HealthResponse(BaseModel):
    status: str
    players_per_team: int


# ── Auth helpers ──


def require_admin(x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")


def get_user_from_key(api_key: str):
    user = db.get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return user


# ── Background evaluation ──


def _evaluate_submission(submission_id: str, policy_path: str):
    """Run round-robin evaluation in a background thread."""
    with _eval_lock:
        try:
            db.update_submission_status(submission_id, "evaluating")
            opponents = db.get_active_submissions_except(submission_id)

            if opponents:
                final_elo = run_round_robin(
                    new_submission_id=submission_id,
                    new_policy_path=policy_path,
                    opponents=opponents,
                    db=db,
                    k_factor=K_FACTOR,
                    games_per_matchup=GAMES_PER_MATCHUP,
                )
                logger.info("Submission %s evaluated. Final ELO: %.1f", submission_id, final_elo)
            else:
                logger.info("Submission %s: no opponents yet, placed at initial ELO", submission_id)

            db.update_submission_status(submission_id, "ready")
        except Exception:
            logger.exception("Evaluation failed for submission %s", submission_id)
            db.update_submission_status(submission_id, "error", error_message="Evaluation failed")


# ── Health ──


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "players_per_team": PLAYERS_PER_TEAM}


# ── Web UI routes ──


@app.get("/", response_class=HTMLResponse)
def web_leaderboard(request: Request):
    leaderboard = db.get_leaderboard()
    for i, entry in enumerate(leaderboard):
        entry["rank"] = i + 1
    return templates.TemplateResponse("leaderboard.html", {
        "request": request,
        "leaderboard": leaderboard,
        "players_per_team": PLAYERS_PER_TEAM,
    })


@app.get("/submit", response_class=HTMLResponse)
def web_submit_form(request: Request):
    return templates.TemplateResponse("submit.html", {
        "request": request,
        "players_per_team": PLAYERS_PER_TEAM,
    })


@app.post("/submit")
async def web_submit(
    request: Request,
    background_tasks: BackgroundTasks,
    api_key: str = Form(...),
    policy_file: UploadFile = File(...),
):
    """Handle web form policy submission."""
    user = get_user_from_key(api_key)
    return await _handle_submission(user, policy_file, background_tasks)


@app.get("/my-submissions", response_class=HTMLResponse)
def web_my_submissions(request: Request, api_key: str = ""):
    if not api_key:
        return templates.TemplateResponse("my_submissions.html", {
            "request": request,
            "submissions": None,
            "user": None,
        })
    user = db.get_user_by_api_key(api_key)
    if not user:
        return templates.TemplateResponse("my_submissions.html", {
            "request": request,
            "submissions": None,
            "user": None,
            "error": "Invalid API key",
        })
    submissions = db.get_user_submissions(user["id"])
    return templates.TemplateResponse("my_submissions.html", {
        "request": request,
        "submissions": submissions,
        "user": user,
    })


# ── API routes ──


@app.post("/api/submit", response_model=SubmissionResponse)
async def api_submit(
    background_tasks: BackgroundTasks,
    policy_file: UploadFile = File(...),
    x_api_key: str = Header(...),
):
    """API endpoint for policy submission."""
    user = get_user_from_key(x_api_key)
    return await _handle_submission(user, policy_file, background_tasks)


@app.get("/api/my-submissions")
def api_my_submissions(x_api_key: str = Header(...)):
    user = get_user_from_key(x_api_key)
    return db.get_user_submissions(user["id"])


@app.get("/api/leaderboard")
def api_leaderboard():
    leaderboard = db.get_leaderboard()
    for i, entry in enumerate(leaderboard):
        entry["rank"] = i + 1
    return leaderboard


@app.get("/api/matches/{submission_id}")
def api_matches(submission_id: str):
    sub = db.get_submission(submission_id)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    return db.get_matches_for_submission(submission_id)


# ── Admin routes ──


@app.post("/admin/users", response_model=UserResponse, dependencies=[Depends(require_admin)])
def admin_create_user(body: UserCreate):
    try:
        return db.create_user(body.name, body.email)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/admin/users", dependencies=[Depends(require_admin)])
def admin_list_users():
    return db.list_users()


@app.delete("/admin/users/{user_id}", dependencies=[Depends(require_admin)])
def admin_delete_user(user_id: str):
    if not db.delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "deleted"}


@app.post("/admin/users/{user_id}/regenerate-key", dependencies=[Depends(require_admin)])
def admin_regenerate_key(user_id: str):
    new_key = db.regenerate_api_key(user_id)
    if not new_key:
        raise HTTPException(status_code=404, detail="User not found")
    return {"api_key": new_key}


@app.post("/admin/recalculate-elo", dependencies=[Depends(require_admin)])
def admin_recalculate_elo():
    """Recalculate all ELO ratings from match history."""
    db.recalculate_all_elo(INITIAL_ELO, K_FACTOR)
    return {"status": "recalculated"}


# ── Shared submission handler ──


async def _handle_submission(
    user: dict,
    policy_file: UploadFile,
    background_tasks: BackgroundTasks,
) -> dict:
    """Save the uploaded policy and queue round-robin evaluation."""
    if not policy_file.filename or not policy_file.filename.endswith(".pt"):
        raise HTTPException(
            status_code=400,
            detail="Policy file must be a .pt TorchScript module",
        )

    # Save policy file
    user_dir = POLICIES_DIR / user["id"]
    user_dir.mkdir(parents=True, exist_ok=True)
    policy_path = user_dir / policy_file.filename
    with open(policy_path, "wb") as f:
        shutil.copyfileobj(policy_file.file, f)

    # Validate that it loads as a TorchScript module
    try:
        import torch
        module = torch.jit.load(str(policy_path), map_location="cpu")
        module.eval()
        del module
    except Exception as e:
        policy_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load policy as TorchScript module: {e}",
        )

    # Create submission record
    sub = db.create_submission(
        user_id=user["id"],
        filename=policy_file.filename,
        policy_path=str(policy_path),
    )

    # Queue background evaluation
    background_tasks.add_task(_evaluate_submission, sub["id"], str(policy_path))

    return {
        "id": sub["id"],
        "user_id": sub["user_id"],
        "filename": sub["filename"],
        "elo_rating": sub["elo_rating"],
        "status": sub["status"],
        "submitted_at": sub["submitted_at"],
    }
