# Puffer Soccer

Native C-backed MARL 2D soccer environment for PufferLib.

## Current Best Agent

The current best checked-in policy is the no-warmstart LSTM curriculum agent:

- Policy checkpoint: [`experiments/t5f7yhut.pt`](experiments/t5f7yhut.pt)
- Run id: `t5f7yhut`
- Training length: `999,997,440` agent steps
- Final built-in evaluation against the previous best: `0.750` win rate,
  `+0.750` mean score difference over 64 games
- Fresh side-balanced comparison against the previous README-plot best:
  `0.767` win rate, `+0.755` mean score difference over 1,024 games

The previous best used for this comparison is preserved as a policy bundle at
[`experiments/baselines/current_best`](experiments/baselines/current_best).
It was exported from run `61xajhha`, epoch `049520`, the checkpoint that
generated the old README plots.

Current policy videos:

- [`experiments/2026-05-21_12-50-10_t5f7yhut/video/self_play.mp4`](experiments/2026-05-21_12-50-10_t5f7yhut/video/self_play.mp4)
- [`experiments/2026-05-21_12-50-10_t5f7yhut/video/best_checkpoint.mp4`](experiments/2026-05-21_12-50-10_t5f7yhut/video/best_checkpoint.mp4)
- [`experiments/2026-05-21_12-50-10_t5f7yhut/video/evals/epoch_024414/vs_best_checkpoint.mp4`](experiments/2026-05-21_12-50-10_t5f7yhut/video/evals/epoch_024414/vs_best_checkpoint.mp4)

## Archived README Plot Agent

The old README plots and videos are archived under
[`archive/readme_plots_2026_04_23`](archive/readme_plots_2026_04_23). This
archive keeps the older checkpoint sequence, generated plots, cached traces,
curated clips, and settings figures separate from the active training outputs.

It contains:

- The slide deck (`docs/thesis_defense_2026-04-23.pdf`)
- Every plot shown in the slides, with the script and data cache that
  produced it
- The trained policy
  (`archive/readme_plots_2026_04_23/61xajhha/model_049520.pt`) and all 248
  stride-200 intermediate checkpoints
- The final-checkpoint trace (input to the occupancy / clip-extraction
  scripts) and the 50 emergence-stats JSONs (input to every emergence plot)
- The two policy videos shown in slide 12 and the warmstart video shown in
  slide 10
- The value-trajectory videos and the curated dribble / pass / goalie /
  defender / forward clips shown in slides 16–20 and 24

The archived run id is `61xajhha` (Slurm job `6856525`): 50,000 self-play epochs
with the gallant regularized-RL update (PPO + KL-to-past + KL-to-uniform),
loaded from the no-opponent curriculum warmstart `y3id1i7o` (job `6849509`).

## Plot index — checklist against the slides

Each entry below shows the slide, the rendered plot embedded inline, and
the script + data inputs needed to remake it. If a plot from the deck is
missing from this list, it is missing from the repo — please flag it.

### Slide 8 — Regularized RL Algorithm (architecture diagram)

[`docs/gallant_regularized_rl_arch.svg`](docs/gallant_regularized_rl_arch.svg).

### Slide 10 — Pure MMD doesn't work; warmstart curriculum

Warmstart-policy self-play (no opponent):
[`archive/readme_plots_2026_04_23/y3id1i7o/video/self_play_no_opponent.mp4`](archive/readme_plots_2026_04_23/y3id1i7o/video/self_play_no_opponent.mp4).

### Slide 12 — Policy Video

Final-policy self-play and best-checkpoint videos:

- [`archive/readme_plots_2026_04_23/61xajhha/video/self_play.mp4`](archive/readme_plots_2026_04_23/61xajhha/video/self_play.mp4)
- [`archive/readme_plots_2026_04_23/61xajhha/video/best_checkpoint.mp4`](archive/readme_plots_2026_04_23/61xajhha/video/best_checkpoint.mp4)

### Slide 14 — Ball Touches

![ball touches per game](archive/readme_plots_2026_04_23/autoloop/plots/emergence/num_touches.png)

Script: `scripts/plot_emergence_individual.py` reading
`archive/readme_plots_2026_04_23/teamplay_trace/61xajhha/stats/`.

### Slide 15 — Field Coverage (ball x-position entropy)

![ball x entropy](archive/readme_plots_2026_04_23/autoloop/plots/emergence/ball_x_entropy.png)

Script: `scripts/plot_emergence_individual.py`.

### Slide 16 — Dribbling

![dribbles per game](archive/readme_plots_2026_04_23/autoloop/plots/emergence/n_dribbles.png)

Script: `scripts/plot_emergence_individual.py`.

Curated dribble clips (`scripts/extract_dribble_pass_clips.py`,
input: final-checkpoint trace at
`archive/readme_plots_2026_04_23/teamplay_trace/61xajhha/traces/trace_epoch_049200.npz`):
[`archive/readme_plots_2026_04_23/autoloop/plots/clips/dribble/`](archive/readme_plots_2026_04_23/autoloop/plots/clips/dribble/) — 4 clips with `.txt` captions.

### Slide 17 — Passing

![passes per game](archive/readme_plots_2026_04_23/autoloop/plots/emergence/n_passes.png)

Script: `scripts/plot_emergence_individual.py`.

Curated pass clips (`scripts/extract_dribble_pass_clips.py`):
[`archive/readme_plots_2026_04_23/autoloop/plots/clips/pass/`](archive/readme_plots_2026_04_23/autoloop/plots/clips/pass/) — 3 clips with `.txt` captions.

### Slide 18 — Goalie

![goalie rotations per game](archive/readme_plots_2026_04_23/autoloop/plots/emergence/goalie_rotations.png)

![% time a goalie on the baseline](archive/readme_plots_2026_04_23/autoloop/plots/emergence/goalie_frac.png)

Script: `scripts/plot_emergence_individual.py`.

Goalie-rotation clips (`scripts/extract_goalie_transition_clips.py`):
[`archive/readme_plots_2026_04_23/autoloop/plots/clips/goalie_transition/`](archive/readme_plots_2026_04_23/autoloop/plots/clips/goalie_transition/) — 4 clips, with team / handoff
agents / possession / forward-extent encoded in the filenames.

### Slide 19 — Defensive Players

![mean # defenders back while attacking](archive/readme_plots_2026_04_23/autoloop/plots/emergence/cond_n_defenders_while_attacking.png)

![% time team keeps a defender back while attacking](archive/readme_plots_2026_04_23/autoloop/plots/emergence/def_while_off_frac.png)

Script: `scripts/plot_emergence_individual.py`.

Defender-vs-offense behavior clips (`scripts/extract_behavior_clips.py`):
[`archive/readme_plots_2026_04_23/autoloop/plots/clips/def_vs_off/`](archive/readme_plots_2026_04_23/autoloop/plots/clips/def_vs_off/) — 4 clips with `.txt` captions.

### Slide 20 — Offensive Players

![% time team leaves a striker forward while defending](archive/readme_plots_2026_04_23/autoloop/plots/emergence/off_while_def_frac.png)

![mean # forwards forward while defending](archive/readme_plots_2026_04_23/autoloop/plots/emergence/cond_n_forwards_while_defending.png)

Script: `scripts/plot_emergence_individual.py`.

Forward-vs-defense behavior clips (`scripts/extract_behavior_clips.py`):
[`archive/readme_plots_2026_04_23/autoloop/plots/clips/fwd_vs_def/`](archive/readme_plots_2026_04_23/autoloop/plots/clips/fwd_vs_def/) — 4 clips with `.txt` captions.

### Slide 21 — Bias Towards Defense (per-team occupancy heatmaps)

| blue team (attacks +x) | red team (attacks −x) |
|---|---|
| ![blue occupancy log](archive/readme_plots_2026_04_23/autoloop/plots/occupancy/occupancy_blue_ep049200_log.png) | ![red occupancy log](archive/readme_plots_2026_04_23/autoloop/plots/occupancy/occupancy_red_ep049200_log.png) |

Script: `scripts/plot_occupancy_heatmaps.py` reading
`archive/readme_plots_2026_04_23/teamplay_trace/61xajhha/traces/trace_epoch_049200.npz`.

### Slide 22 — Agents tend to play along the sides (ball occupancy)

![ball occupancy linear](archive/readme_plots_2026_04_23/autoloop/plots/occupancy/occupancy_ball_ep049200_linear.png)

Script: `scripts/plot_occupancy_heatmaps.py`.

### Slide 24 — Critic's value rises as the scoring team approaches the goal

![value along trajectory](archive/readme_plots_2026_04_23/autoloop/value_trajectory/value_along_trajectory.png)

Script: `scripts/value_along_trajectory.py`. Cached data:
[`archive/readme_plots_2026_04_23/autoloop/value_trajectory/trajectories.npz`](archive/readme_plots_2026_04_23/autoloop/value_trajectory/trajectories.npz),
[`archive/readme_plots_2026_04_23/autoloop/value_trajectory/trajectory_summary.json`](archive/readme_plots_2026_04_23/autoloop/value_trajectory/trajectory_summary.json).
The 10 example single-goal trajectories (5 blue, 5 red) are rendered to
mp4 in [`archive/readme_plots_2026_04_23/autoloop/value_trajectory/videos/`](archive/readme_plots_2026_04_23/autoloop/value_trajectory/videos/).

### Slide 25 — V(carrier) under varied formations

![formation value heatmap](archive/readme_plots_2026_04_23/autoloop/formation/formation_value_heatmap.png)

Script: `scripts/formation_value_heatmap.py`. Cached data:
[`archive/readme_plots_2026_04_23/autoloop/formation/formation_v.npy`](archive/readme_plots_2026_04_23/autoloop/formation/formation_v.npy).

### Slide 26 — Goalie ΔV across blue/red line positions

![formation value goalie delta](archive/readme_plots_2026_04_23/autoloop/formation/formation_value_goalie_delta.png)

Script: `scripts/formation_value_goalie_delta.py`. Cached data:
[`archive/readme_plots_2026_04_23/autoloop/formation/formation_v.npy`](archive/readme_plots_2026_04_23/autoloop/formation/formation_v.npy),
[`archive/readme_plots_2026_04_23/autoloop/formation/formation_v_no_goalie.npy`](archive/readme_plots_2026_04_23/autoloop/formation/formation_v_no_goalie.npy).

### Slide 27 — Goalie value in a realistic mid-attack state

![goalie save region](archive/readme_plots_2026_04_23/autoloop/formation/goalie_save_region.png)

Script: `scripts/goalie_save_region.py`. Cached data:
[`archive/readme_plots_2026_04_23/autoloop/formation/goalie_save_region_v_no.npy`](archive/readme_plots_2026_04_23/autoloop/formation/goalie_save_region_v_no.npy),
[`archive/readme_plots_2026_04_23/autoloop/formation/goalie_save_region_v_with.npy`](archive/readme_plots_2026_04_23/autoloop/formation/goalie_save_region_v_with.npy).

---

## Settings-section figures (thesis Chapter 3.1)

Static figures that visualize the env's geometry and dynamics — the
prose in the Settings section of the written thesis describes these
behaviors mathematically; these plots make them visible. None require
the trained policy except the FOV figure, which uses one frame from the
final-checkpoint trace.

### Field schematic

Annotated 100×70 field with goal regions (`|y| ≤ 20`), midline, and one
example 5v5 reset showing player positions and headings.

![field schematic](archive/readme_plots_2026_04_23/setting/field_schematic.png)

Script: `scripts/plot_setting_field.py`. Snapshot of the rendered state
saved at `archive/readme_plots_2026_04_23/setting/field_schematic_state.npz` so the figure
can be regenerated without re-running the env.

### Kick-strength → ball trajectory

Ball-only simulation of each of the 8 discrete kick strengths under the
env's impulse + 0.85-decay + 5.0-clip rule. Constants (`leg_speed=4.0`,
`BALL_VELOCITY_DECAY=0.85`, `MAX_BALL_SPEED=5.0`, the 8 kick scales)
match `binding.c` at this commit.

![kick strengths](archive/readme_plots_2026_04_23/setting/kick_strengths.png)

Script: `scripts/plot_setting_kick_strengths.py`. Cached data at
`archive/readme_plots_2026_04_23/setting/kick_strengths_data.npz`.

### Field-of-view overlay (partial observability)

One frame from the final-checkpoint self-play trace, with one player
highlighted as the observer. The gold wedge is its 180° forward FOV.
Solid agents are visible to the observer; hollow/dashed agents lie
outside the FOV and have their feature blocks zero-masked in the
observation.

![FOV overlay](archive/readme_plots_2026_04_23/setting/fov_overlay.png)

Script: `scripts/plot_setting_fov.py`. Input:
`archive/readme_plots_2026_04_23/teamplay_trace/61xajhha/traces/trace_epoch_049200.npz`.

## Reproducing the plots

`uv sync --extra dev` and the env at this commit. The trained policy is
`archive/readme_plots_2026_04_23/61xajhha/model_049520.pt`; the warmstart is
`experiments/cached_warm_start.pt`.

```bash
# Emergence stats — slow, walks every checkpoint and writes stats + traces
uv run python scripts/teamplay_trace.py \
  --checkpoint-dir archive/readme_plots_2026_04_23/61xajhha \
  --output-dir archive/readme_plots_2026_04_23/teamplay_trace/61xajhha \
  --stride 5 \
  --traces-dir archive/readme_plots_2026_04_23/teamplay_trace/61xajhha/traces \
  --traces-keep-last-only

# Per-metric emergence panels (slides 14, 15, 16, 17, 18, 19, 20)
uv run python scripts/plot_emergence_individual.py \
  --input-dir archive/readme_plots_2026_04_23/teamplay_trace/61xajhha/stats \
  --output-dir archive/readme_plots_2026_04_23/autoloop/plots/emergence

# Occupancy heatmaps (slides 21, 22)
uv run python scripts/plot_occupancy_heatmaps.py \
  --traces archive/readme_plots_2026_04_23/teamplay_trace/61xajhha/traces/trace_epoch_049200.npz \
  --output-dir archive/readme_plots_2026_04_23/autoloop/plots/occupancy

# Value along trajectory (slide 24)
uv run python scripts/value_along_trajectory.py \
  --checkpoint archive/readme_plots_2026_04_23/61xajhha/model_049520.pt \
  --output-dir archive/readme_plots_2026_04_23/autoloop/value_trajectory

# Formation value heatmap (slide 25) — also writes formation_v.npy that
# the goalie-delta script consumes as --v-with-goalie
uv run python scripts/formation_value_heatmap.py \
  --checkpoint archive/readme_plots_2026_04_23/61xajhha/model_049520.pt \
  --output-dir archive/readme_plots_2026_04_23/autoloop/formation

# Formation goalie delta (slide 26) — needs the formation_v.npy from the
# previous step
uv run python scripts/formation_value_goalie_delta.py \
  --checkpoint archive/readme_plots_2026_04_23/61xajhha/model_049520.pt \
  --v-with-goalie archive/readme_plots_2026_04_23/autoloop/formation/formation_v.npy \
  --output-dir archive/readme_plots_2026_04_23/autoloop/formation

# Goalie save region (slide 27)
uv run python scripts/goalie_save_region.py \
  --checkpoint archive/readme_plots_2026_04_23/61xajhha/model_049520.pt \
  --output-dir archive/readme_plots_2026_04_23/autoloop/formation

# Settings figures (no policy required, except FOV uses one trace frame)
uv run python scripts/plot_setting_field.py \
  --output-dir archive/readme_plots_2026_04_23/setting
uv run python scripts/plot_setting_kick_strengths.py \
  --output-dir archive/readme_plots_2026_04_23/setting
uv run python scripts/plot_setting_fov.py \
  --trace archive/readme_plots_2026_04_23/teamplay_trace/61xajhha/traces/trace_epoch_049200.npz \
  --output-dir archive/readme_plots_2026_04_23/setting --frame 200 --observer 2

# Behavior clips (slides 16, 17, 18, 19, 20) — each runs its own short
# self-play rollout from the final checkpoint and writes mp4 + .txt clip
# files into a behavior-typed subdir under --output-dir
uv run python scripts/extract_dribble_pass_clips.py \
  --checkpoint archive/readme_plots_2026_04_23/61xajhha/model_049520.pt \
  --output-dir archive/readme_plots_2026_04_23/autoloop/plots/clips
uv run python scripts/extract_goalie_transition_clips.py \
  --checkpoint archive/readme_plots_2026_04_23/61xajhha/model_049520.pt \
  --output-dir archive/readme_plots_2026_04_23/autoloop/plots/clips
uv run python scripts/extract_behavior_clips.py \
  --checkpoint archive/readme_plots_2026_04_23/61xajhha/model_049520.pt \
  --output-dir archive/readme_plots_2026_04_23/autoloop/plots/clips
```

---

## Install

```bash
uv sync --extra dev
```

## Quick demo

```bash
uv run python main.py
```

## Train (PuffeRL PPO baseline)

```bash
uv run scripts/train_pufferl.py --players-per-team 5 --vec-backend auto --ppo-iterations 1000
```

This writes a self-play video at `experiments/self_play.mp4` after training.
W&B logging is enabled by default with the `robot-soccer` project and logs the generated self-play video to the same run.

The training path runs directly on `MARL2DPufferEnv` without a PettingZoo wrapper or Python serial vectorizer.

To autotune the vector layout on the current machine before training, use the auto backend.
This runs a short pre-training sweep over vector layouts, prefers near-100% CPU usage,
and then trains with the fastest selected configuration:

```bash
uv run python scripts/train_pufferl.py \
  --players-per-team 5 \
  --ppo-iterations 1000 \
  --vec-backend auto
```

For higher CPU throughput, use Puffer's multiprocessing vecenv with small native shards per worker:

```bash
uv run python scripts/train_pufferl.py \
  --players-per-team 5 \
  --ppo-iterations 1000 \
  --vec-backend multiprocessing \
  --num-envs 3072 \
  --vec-num-shards 16 \
  --vec-batch-size 1
```

## Benchmark

```bash
uv run python scripts/benchmark_sps.py --num-envs 64 --seconds 10 --action-mode discrete
```

Autotune across native and multiprocessing layouts until the CPU saturates, then pick the highest-SPS configuration:

```bash
uv run python scripts/benchmark_sps.py --backend auto --players-per-team 5 --autotune --seconds 3 --action-mode discrete

# `--seconds` is optional; autotune uses a built-in short sample and stops once
# it reaches near-100% CPU usage and SPS plateaus.
```

Benchmark the Puffer multiprocessing layout directly:

```bash
uv run python scripts/benchmark_sps.py \
  --backend multiprocessing \
  --players-per-team 5 \
  --shard-num-envs-list 160,192,224 \
  --num-shards-list 16 \
  --batch-size-list 1,2,4 \
  --seconds 3
```

## Tests

```bash
uv run pytest -q
```

`tests/test_parity.py` compares against `third-party/MARL2DFootball` when its dependencies are available.
