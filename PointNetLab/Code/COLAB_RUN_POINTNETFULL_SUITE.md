# Colab Quickstart - PointNetFull Augmentation Suite

Ce guide te permet de lancer la campagne automatisée sur Google Colab (GPU) en gardant tous les résultats sur Google Drive.

## 1) Monter Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2) Aller dans ton projet

Adapte ce chemin si ton dossier n'a pas le même nom dans Drive.

```python
%cd /content/drive/MyDrive/Apprenyissage_Pointcloud/Pointclouds-classification-with-the-POINTNET-Neural-network/PointNetLab/Code
```

## 3) Vérifier GPU

```python
import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## 4) Lancer la campagne complète (25 epochs/config)

Le script crée automatiquement un dossier `reports/pointnetfull_aug_suite_YYYYMMDD_HHMMSS`,
donc les campagnes précédentes ne sont jamais écrasées.

```python
!python -u run_pointnetfull_augmentation_suite.py \
  --data-root /content/drive/MyDrive/Apprenyissage_Pointcloud/Pointclouds-classification-with-the-POINTNET-Neural-network/PointNetLab/data/ModelNet10_PLY \
  --epochs 25 \
  --batch-size 32 \
  --num-workers 2
```

## 5) (Optionnel) Lancer en arrière-plan + log fichier

Utile si tu veux laisser tourner la cellule sans tout afficher.

```python
%%bash
set -e
mkdir -p reports/logs
LOG="reports/logs/pointnetfull_aug_suite_colab_$(date +%Y%m%d_%H%M%S).log"
echo "LOG=$LOG"
nohup python -u run_pointnetfull_augmentation_suite.py \
  --data-root /content/drive/MyDrive/Apprenyissage_Pointcloud/Pointclouds-classification-with-the-POINTNET-Neural-network/PointNetLab/data/ModelNet10_PLY \
  --epochs 25 \
  --batch-size 32 \
  --num-workers 2 \
  > "$LOG" 2>&1 &
echo "PID=$!"
```

Puis suivre le log:

```python
!tail -n 60 reports/logs/<TON_LOG>.log
```

## 6) Où récupérer les résultats

Dans le dossier de run généré:

- `experiment_summary.csv`
- `experiment_summary.json`
- `notebooks/01_pointnetfull_augmentation_summary.ipynb`
- `notebooks/02_pointnetfull_augmentation_detailed_logs.ipynb`
- `figures/`
- `histories/`
- `checkpoints/`

