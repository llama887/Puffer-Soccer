# Archived README Plot Artifacts

This directory contains the older thesis-defense / README-plot artifact set.
It is intentionally separate from active training outputs in `experiments/`.

Contents:

- `61xajhha/`: previous best checkpoint run, including
  `model_049520.pt`, stride-200 intermediate checkpoints, trainer state, and
  policy videos.
- `y3id1i7o/`: warmstart-policy video used by the old plot narrative.
- `teamplay_trace/`: cached stats and final trace used to create the emergence
  and occupancy plots.
- `autoloop/`: generated emergence plots, occupancy plots, value-trajectory
  plots, formation-value plots, cached arrays, and curated behavior clips.
- `setting/`: static environment-setting figures and their small data caches.

The plotting and clip-extraction code remains in the top-level `scripts/`
directory so it continues to share the current package imports and `uv run`
workflow. The top-level README lists the exact commands and points them at this
archive path.
