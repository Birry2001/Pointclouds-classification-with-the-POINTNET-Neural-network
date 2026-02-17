# PointNetFull Experimental Protocol (Robust)

This file summarizes the new benchmark protocol implemented in:

- `run_pointnetfull_augmentation_suite.py`

## 1) Normalization (`--normalize-unit-sphere`)

When enabled, each point cloud is:
1. centered by its centroid,
2. scaled by the max point norm to fit into the unit sphere.

This normalization is applied consistently to train and test data.

## 2) Multi-seeds benchmark

Use `--seeds` (default `0 1 2`) to run each experiment multiple times.

Each run is isolated:

- `runs/<experiment>/seed_<seed>/...`

No overwrite occurs between seeds or experiments.

## 3) Early stopping

Configurable with:

- `--early-stop-monitor val_acc|val_loss`
- `--early-stop-patience <int>`
- `--early-stop-min-delta <float>`

Training stops when there is no improvement on the chosen monitor for `patience` epochs.

## 4) Sweep mode

Use:

- `--mode sweep --sweep jitter`
- `--mode sweep --sweep occlusion`
- `--mode sweep --sweep dropout`

Generated experiment names are explicit:

- jitter: `jitter_sX_cY`
- occlusion: `occ_pX_rY`
- dropout: `drop_rX`

Sweep parameters are logged in run JSON and aggregated summaries.

## 5) Output folder structure

Inside timestamped run dir (`pointnetfull_aug_suite_YYYYMMDD_HHMMSS`):

- `runs/...` per-seed run artifacts (`history`, `checkpoint`, `curves`, `confusion`, `predictions`)
- `summaries/run_summary.csv|json`
- `summaries/summary_experiments.csv|json`
- `summaries/summary_experiments_by_eval.csv`
- `results/plots/` comparative plots
- `results/top_confusions.csv` + hard examples plots
- `notebooks/` summary notebooks
- `run_metadata.json` full run config + artifact paths

## 6) Reproducibility

The script sets:

- Python `random`
- NumPy
- PyTorch (CPU/CUDA)

And can enable deterministic behavior with:

- `--deterministic true` (default)

