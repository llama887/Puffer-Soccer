"""Sanity-check: Python-built observation vs env's real observation on reset."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from value_heatmap import build_observation_batch
from puffer_soccer.envs.marl2d import make_puffer_env
from puffer_soccer.envs.marl2d.core import MARL2DPufferEnv  # noqa: F401

env = make_puffer_env(players_per_team=5, action_mode="discrete", game_length=400, seed=42)
obs_env, _ = env.reset(seed=42)
state = env.get_state(0)
positions = state["positions"]  # (10, 2)
rotations = state["rotations"]  # (10,)
ball = state["ball"]  # (bx, by, bvx, bvy)
blue_left = state["blue_left"]

team = np.zeros(10, dtype=np.int32)
team[5:] = 1

bxy_one = np.array([[ball[0], ball[1]]], dtype=np.float32)
bv = np.array([ball[2], ball[3]], dtype=np.float32)

py_obs_blue = build_observation_batch(
    agents_xy=positions,
    agents_rot=rotations,
    agents_team=team,
    ball_xy_grid=bxy_one,
    ball_v=bv,
    blue_left=blue_left,
    focus_team=0,
)
py_obs_red = build_observation_batch(
    agents_xy=positions,
    agents_rot=rotations,
    agents_team=team,
    ball_xy_grid=bxy_one,
    ball_v=bv,
    blue_left=blue_left,
    focus_team=1,
)

py_all = np.concatenate([py_obs_blue[0], py_obs_red[0]], axis=0)
print(f"env obs shape: {obs_env.shape}")
print(f"py  obs shape: {py_all.shape}")
print(f"env blue first 5 agents first 20 floats:\n{obs_env[:5, :20]}")
print(f"py  blue first 5 agents first 20 floats:\n{py_all[:5, :20]}")
print(f"max abs diff across entire obs: {np.abs(obs_env - py_all).max()}")
print(f"mean abs diff: {np.abs(obs_env - py_all).mean()}")
print(f"positions:\n{positions}")
print(f"rotations:\n{rotations}")
print(f"ball: {ball}")
print(f"blue_left: {blue_left}")
