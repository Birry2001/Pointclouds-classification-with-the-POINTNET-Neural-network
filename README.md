# Classification de nuages de points 3D avec PointNet

Ce dépôt contient le travail réalisé pour le projet **PointNetLab** (Master PAR – 3D Deep Learning) autour de la classification de nuages de points 3D.

## Objectif

Implémenter et comparer plusieurs variantes de réseaux pour la classification de nuages de points :
- `PointMLP` (baseline MLP),
- `PointNetBasic` (sans T-Net),
- `PointNetFull` (avec T-Net 3x3 + régularisation).

Le projet inclut aussi une étude d’augmentations de données (rotation, bruit, jitter, occlusion, etc.) et un protocole d’évaluation rigoureux avec séparation `train/val/test`.

## Structure du dépôt

- `PointNetLab/Code/pointnet.py` : script principal (version finale propre, commentée).
- `PointNetLab/Code/pointnet_notebook.ipynb` : notebook d’expérimentation et de visualisation.
- `PointNetLab/Code/ply.py` : utilitaires de lecture PLY.
- `PointNetLab/Collab_Code/` : version collaborative/template de départ.
- `PointNetLab/Bonus/` : contenus bonus (MinkowskiNet).
- `PointNetLab/data/ModelNet10_PLY` : jeu de données ModelNet10 (format PLY).
- `PointNetLab/data/ModelNet40_PLY` : jeu de données ModelNet40 (format PLY).
- `PointNetLab/master_PAR_enonce_Lab_PointNet_2026.pdf` : énoncé officiel du projet.
- `Rapport_NOCHI.pdf` : rapport final.

## Prérequis

- Python 3.9+
- PyTorch
- torchvision
- numpy
- matplotlib (pour le notebook)
- jupyter (pour le notebook)
- scipy (utilisé dans `Collab_Code`)

Exemple d’installation rapide :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision numpy matplotlib jupyter scipy
```

## Exécution rapide

Lancer depuis la racine du dépôt :

```bash
python3 PointNetLab/Code/pointnet.py
```

Dans `PointNetLab/Code/pointnet.py`, les paramètres principaux sont :
- `DATA_ROOT` (jeu de données à utiliser, ModelNet10/40),
- `MODEL_NAME` (`PointMLP`, `PointNetBasic`, `PointNetFull`),
- `AUGMENTATIONS_TRAIN`, `AUGMENTATIONS_VAL`, `AUGMENTATIONS_TEST`,
- `VAL_RATIO`, `SPLIT_SEED`.

## Notebook

Ouvrir le notebook :

```bash
jupyter notebook PointNetLab/Code/pointnet_notebook.ipynb
```

Le notebook inclut :
- la configuration du seed global,
- le split `train/val` pour l’early stopping,
- une évaluation sur `test` uniquement en fin d’entraînement,
- des courbes d’apprentissage (loss/accuracy),
- une cellule de visualisation des effets des augmentations.

## Protocole expérimental 

- **Phase exploratoire** : comparaison de politiques d’augmentation sur un budget court.
- **Phase finale** : sélection des configurations candidates et comparaison plus robuste.
- **Rigueur d’évaluation** :
  - `train` pour apprendre,
  - `val` pour les choix (early stopping / sélection),
  - `test` évalué une seule fois sur modèle figé.

## Auteurs

- Étudiant : **NOCHI MAGOUO**
- Encadrant : **PAUL CHECCHIN**

## Licence

Ce projet est distribué sous la licence MIT.
