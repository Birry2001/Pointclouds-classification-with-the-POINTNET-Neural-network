#!/usr/bin/env python
# PointNet for point cloud classification (clean TP version)

import math
import os
import random
import sys
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# Force local module resolution to avoid conflict with external "ply" package
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
from ply import read_ply


# -----------------------------
# Data augmentations
# -----------------------------

class RandomRotationZ(object):
    def __call__(self, pointcloud: np.ndarray):
        theta = random.random() * 2.0 * math.pi
        rot = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return pointcloud @ rot.T


class RandomNoise(object):
    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, pointcloud: np.ndarray):
        noise = np.random.normal(0.0, self.std, pointcloud.shape).astype(np.float32)
        return pointcloud + noise


class ShufflePoints(object):
    def __call__(self, pointcloud: np.ndarray):
        out = pointcloud.copy()
        np.random.shuffle(out)
        return out


class RandomScale(object):
    def __init__(self, low: float = 0.85, high: float = 1.20):
        self.low = low
        self.high = high

    def __call__(self, pointcloud: np.ndarray):
        s = np.random.uniform(self.low, self.high)
        return pointcloud * np.float32(s)


class RandomTranslate(object):
    def __init__(self, shift: float = 0.10):
        self.shift = shift

    def __call__(self, pointcloud: np.ndarray):
        t = np.random.uniform(-self.shift, self.shift, size=(1, 3)).astype(np.float32)
        return pointcloud + t


class RandomJitterClip(object):
    def __init__(self, sigma: float = 0.01, clip: float = 0.04):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, pointcloud: np.ndarray):
        jit = self.sigma * np.random.randn(*pointcloud.shape).astype(np.float32)
        jit = np.clip(jit, -self.clip, self.clip)
        return pointcloud + jit


class RandomMirrorXY(object):
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, pointcloud: np.ndarray):
        if random.random() >= self.p:
            return pointcloud
        out = pointcloud.copy()
        axis = random.choice([0, 1])
        out[:, axis] = -out[:, axis]
        return out


class RandomPointDropout(object):
    def __init__(self, max_ratio: float = 0.45):
        self.max_ratio = max_ratio

    def __call__(self, pointcloud: np.ndarray):
        out = pointcloud.copy()
        n = out.shape[0]
        ratio = np.random.uniform(0.0, self.max_ratio)
        drop_idx = np.where(np.random.random(n) <= ratio)[0]
        if len(drop_idx) > 0:
            out[drop_idx] = out[0]
        return out


class RandomSphericalOcclusion(object):
    def __init__(self, radius: float = 0.18, p: float = 0.25):
        self.radius = radius
        self.p = p

    def __call__(self, pointcloud: np.ndarray):
        if random.random() >= self.p:
            return pointcloud

        out = pointcloud.copy()
        center = out[np.random.randint(0, out.shape[0])]
        d = np.linalg.norm(out - center[None, :], axis=1)
        keep_mask = d > self.radius

        if np.sum(keep_mask) < 8:
            return out

        kept = out[keep_mask]
        if kept.shape[0] >= out.shape[0]:
            return kept[: out.shape[0]]

        extra_idx = np.random.choice(kept.shape[0], out.shape[0] - kept.shape[0], replace=True)
        return np.concatenate([kept, kept[extra_idx]], axis=0)


class UnitSphereNormalize(object):
    def __call__(self, pointcloud: np.ndarray):
        centered = pointcloud - np.mean(pointcloud, axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        m = float(np.max(norms)) if norms.size else 1.0
        if m < 1e-12:
            return centered.astype(np.float32)
        return (centered / m).astype(np.float32)


class ToTensor(object):
    def __call__(self, pointcloud: np.ndarray):
        return torch.from_numpy(pointcloud.astype(np.float32))



AUGMENTATION_REGISTRY = {
    "rot_z": lambda: RandomRotationZ(),
    "noise": lambda: RandomNoise(std=0.02),
    "shuffle": lambda: ShufflePoints(),
    "scale": lambda: RandomScale(low=0.85, high=1.20),
    "translate": lambda: RandomTranslate(shift=0.10),
    "jitter": lambda: RandomJitterClip(sigma=0.01, clip=0.04),
    "mirror": lambda: RandomMirrorXY(p=0.3),
    "dropout": lambda: RandomPointDropout(max_ratio=0.45),
    "occlusion": lambda: RandomSphericalOcclusion(radius=0.18, p=0.25),
    "normalize": lambda: UnitSphereNormalize(),
}


def build_transforms(selected: Optional[List[str]] = None):
    selected = selected or []
    ops = [AUGMENTATION_REGISTRY[name]() for name in selected]
    ops.append(ToTensor())
    return transforms.Compose(ops)


# -----------------------------
# Dataset
# -----------------------------

class PointCloudData(Dataset):
    def __init__(self, root_dir, folder="train", transform=None):
        self.root_dir = root_dir
        self.transforms = transform if transform is not None else build_transforms(["rot_z", "noise", "shuffle"])

        folders = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes = {name: i for i, name in enumerate(folders)}

        self.files = []
        for category in self.classes.keys():
            split_dir = os.path.join(root_dir, category, folder)
            for file in os.listdir(split_dir):
                if file.endswith('.ply'):
                    self.files.append({
                        'ply_path': os.path.join(split_dir, file),
                        'category': category,
                    })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = self.files[idx]
        data = read_ply(sample['ply_path'])
        pointcloud = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)
        pointcloud = self.transforms(pointcloud)
        label = self.classes[sample['category']]
        return {'pointcloud': pointcloud, 'category': label}


# -----------------------------
# Models
# -----------------------------

class PointMLP(nn.Module):
    # Ex.1: 3072 -> 512 -> 256 -> N, BN + ReLU, dropout(0.3), LogSoftmax
    def __init__(self, classes=40):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.drop(torch.relu(self.bn2(self.fc2(x))))
        x = self.logsoftmax(self.fc3(x))
        return x


class PointNetBasic(nn.Module):
    # Ex.2.1: PointNet sans T-Net
    def __init__(self, classes=40):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)

        self.maxpool = nn.MaxPool1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x).squeeze(-1)
        x = torch.relu(self.bn6(self.fc1(x)))
        x = self.drop(torch.relu(self.bn7(self.fc2(x))))
        x = self.logsoftmax(self.fc3(x))
        return x


class Tnet(nn.Module):
    # Ex.2.2: mini-PointNet qui régressse une matrice k x k
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.maxpool = nn.MaxPool1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x).squeeze(-1)
        x = torch.relu(self.bn4(self.fc1(x)))
        x = torch.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).reshape(x.size(0), self.k, self.k)

        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(x.size(0), 1, 1)
        return x + identity


class PointNetFull(nn.Module):
    # Ex.2.2 demandé: PointNet avec le 1er T-Net (3x3)
    def __init__(self, classes=40):
        super().__init__()
        self.tnet1 = Tnet(k=3)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)

        self.maxpool = nn.MaxPool1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        m3x3 = self.tnet1(x)
        x = torch.bmm(m3x3, x)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))

        x = self.maxpool(x).squeeze(-1)
        x = torch.relu(self.bn6(self.fc1(x)))
        x = self.drop(torch.relu(self.bn7(self.fc2(x))))
        x = self.logsoftmax(self.fc3(x))
        return x, m3x3


# -----------------------------
# Losses
# -----------------------------

def basic_loss(outputs, labels):
    criterion = nn.NLLLoss()
    return criterion(outputs, labels)


def pointnet_full_loss(outputs, labels, m3x3, alpha=0.001):
    criterion = nn.NLLLoss()
    bsize = outputs.size(0)
    id3x3 = torch.eye(3, device=outputs.device, dtype=outputs.dtype).unsqueeze(0).repeat(bsize, 1, 1)
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    return criterion(outputs, labels) + alpha * torch.norm(diff3x3) / float(bsize)


# -----------------------------
# Training helpers
# -----------------------------

def forward_with_model(model, x):
    if isinstance(model, PointMLP):
        out = model(x.transpose(1, 2))
        return out, None
    if isinstance(model, PointNetBasic):
        out = model(x.transpose(1, 2))
        return out, None
    out, m3x3 = model(x.transpose(1, 2))
    return out, m3x3


def evaluate(model, device, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch['pointcloud'].to(device).float()
            y = batch['category'].to(device)
            out, _ = forward_with_model(model, x)
            pred = out.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return 100.0 * correct / max(1, total)


def train(model,
          device,
          train_loader,
          val_loader=None,
          epochs=25,
          lr=1e-3,
          use_early_stopping=False,
          early_stop_patience=8,
          early_stop_min_delta=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_val = None
    no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch['pointcloud'].to(device).float()  # (B, N, 3)
            y = batch['category'].to(device)

            optimizer.zero_grad()
            out, m3x3 = forward_with_model(model, x)

            if isinstance(model, PointNetFull):
                loss = pointnet_full_loss(out, y, m3x3)
            else:
                loss = basic_loss(out, y)

            loss.backward()
            optimizer.step()

        scheduler.step()

        if val_loader is not None:
            val_acc = evaluate(model, device, val_loader)
            print(f"Epoch {epoch+1:03d} | loss={loss.item():.4f} | val_acc={val_acc:.2f}%")

            if use_early_stopping:
                improved = (best_val is None) or (val_acc > best_val + early_stop_min_delta)
                if improved:
                    best_val = val_acc
                    no_improve = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= early_stop_patience:
                        print(f"Early stopping déclenché à l'epoch {epoch+1}")
                        break

    if best_state is not None:
        model.load_state_dict(best_state)


if __name__ == '__main__':
    DATA_ROOT = "PointNetLab/data/ModelNet10_PLY"

    # Choisis ici les augmentations que tu veux composer
    AUGMENTATIONS_TRAIN = ["normalize", "rot_z", "noise", "shuffle", "occlusion"]
    AUGMENTATIONS_VAL = ["normalize"]
    AUGMENTATIONS_TEST = ["normalize"]


    VAL_RATIO = 0.2
    SPLIT_SEED = 42

    full_train_ds = PointCloudData(DATA_ROOT, folder='train', transform=build_transforms(AUGMENTATIONS_TRAIN))
    full_train_eval_ds = PointCloudData(DATA_ROOT, folder='train', transform=build_transforms(AUGMENTATIONS_VAL))
    test_ds = PointCloudData(DATA_ROOT, folder='test', transform=build_transforms(AUGMENTATIONS_TEST))

    # Split train -> train_sub + val_sub (early stopping sur val, pas sur test)
    rng = np.random.RandomState(SPLIT_SEED)
    indices = np.arange(len(full_train_ds))
    rng.shuffle(indices)
    val_size = int(len(indices) * VAL_RATIO)
    val_idx = indices[:val_size].tolist()
    train_idx = indices[val_size:].tolist()

    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(full_train_eval_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Choix du modèle: PointMLP | PointNetBasic | PointNetFull
    MODEL_NAME = "PointNetFull"
    classes = len(full_train_ds.classes)

    if MODEL_NAME == "PointMLP":
        model = PointMLP(classes=classes)
    elif MODEL_NAME == "PointNetBasic":
        model = PointNetBasic(classes=classes)
    elif MODEL_NAME == "PointNetFull":
        model = PointNetFull(classes=classes)
    else:
        raise ValueError(f"Unknown model: {MODEL_NAME}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Classes:", classes)
    print("Train size:", len(train_ds), "Val size:", len(val_ds), "Test size:", len(test_ds))
    print("Model:", MODEL_NAME, "Device:", device)
    print("Augs train:", AUGMENTATIONS_TRAIN)
    print("Augs val:", AUGMENTATIONS_VAL)
    print("Augs test:", AUGMENTATIONS_TEST)

    t0 = time.time()
    train(model,
          device,
          train_loader,
          val_loader=val_loader,
          epochs=50,
          lr=1e-3,
          use_early_stopping=True,
          early_stop_patience=8,
          early_stop_min_delta=0.0)
    test_acc = evaluate(model, device, test_loader)
    print(f"Final TEST accuracy (évalué une seule fois): {test_acc:.2f}%")
    print(f"Total time: {time.time() - t0:.1f}s")
