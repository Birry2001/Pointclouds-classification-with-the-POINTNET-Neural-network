# Colab Quickstart - PointNetFull Robust Benchmark

This guide runs the new robust protocol:
- multi-seeds,
- early stopping,
- optional normalization,
- optional sweep mode,
- auto device selection (`TPU > GPU > CPU`).

## 1) Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2) Go to project

```python
%cd /content/drive/MyDrive/M2_PAR/Apprentissage/Pointclouds-classification-with-the-POINTNET-Neural-network/PointNetLab/Code
```

## 3) Probe devices

```python
!python run_pointnetfull_augmentation_suite.py --probe-devices
```

## 4) Check dataset path

```python
DATA_ROOT="/content/drive/MyDrive/M2_PAR/Apprentissage/Pointclouds-classification-with-the-POINTNET-Neural-network/PointNetLab/data/ModelNet10_PLY"
!find "$DATA_ROOT" -type f -iname "*.ply" | head -n 20
!find "$DATA_ROOT" -type f -iname "*.ply" | wc -l
```

## 5) Robust benchmark (recommended)

Example requested setup:
- experiments: baseline, plus_jitter, plus_occlusion, combo_full
- seeds: 0,1,2
- epochs max: 150
- early stopping patience: 15

```python
!python -u run_pointnetfull_augmentation_suite.py \
  --mode benchmark \
  --experiments baseline plus_jitter plus_occlusion combo_full \
  --seeds 0 1 2 \
  --epochs-max 150 \
  --early-stop-monitor val_acc \
  --early-stop-patience 15 \
  --batch-size 16 \
  --num-workers 2 \
  --normalize-unit-sphere true \
  --use-scheduler true \
  --device auto \
  --make-plots true \
  --make-error-analysis true \
  --make-gallery false \
  --data-root "$DATA_ROOT"
```

## 6) Sweep mode examples

```python
!python -u run_pointnetfull_augmentation_suite.py \
  --mode sweep --sweep jitter \
  --seeds 0 1 2 \
  --epochs-max 80 \
  --batch-size 16 \
  --device auto \
  --data-root "$DATA_ROOT"
```

```python
!python -u run_pointnetfull_augmentation_suite.py \
  --mode sweep --sweep occlusion \
  --seeds 0 1 2 \
  --epochs-max 80 \
  --batch-size 16 \
  --device auto \
  --data-root "$DATA_ROOT"
```

```python
!python -u run_pointnetfull_augmentation_suite.py \
  --mode sweep --sweep dropout \
  --seeds 0 1 2 \
  --epochs-max 80 \
  --batch-size 16 \
  --device auto \
  --data-root "$DATA_ROOT"
```

## 7) Outputs

Each run is written to a new timestamped directory:

- `pointnetfull_aug_suite_YYYYMMDD_HHMMSS/`
  - `runs/<experiment>/seed_<seed>/...`
  - `summaries/run_summary.csv`
  - `summaries/summary_experiments.csv`
  - `summaries/summary_experiments_by_eval.csv`
  - `results/plots/...`
  - `results/top_confusions.csv`
  - `notebooks/01_benchmark_summary.ipynb`
  - `notebooks/02_detailed_runs.ipynb`
  - `run_metadata.json`

