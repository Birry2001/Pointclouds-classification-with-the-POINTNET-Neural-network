# Pointclouds-classification-with-the-POINTNET-Neural-network
In this project, we implement the POINTNET neural network for pointcloud classification

## Latest notebook updates (work branch)

- Added an EarlyStopping workflow in `PointNetLab/Code/pointnet_notebook.ipynb` to keep a fixed `MAX_EPOCHS` while stopping when validation stalls.
- Added a dedicated visualization cell to display one random point cloud and the effect of each data augmentation.

## Reproducibility protocol update

- Training now uses a **train/validation split** for early stopping.
- Test set is kept for **final evaluation only**.
- Notebook experiments expose a global seed (`GLOBAL_SEED`) for deterministic runs.
