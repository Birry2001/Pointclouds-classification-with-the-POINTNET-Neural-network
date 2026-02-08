#!/usr/bin/env python3
"""
Automated PointNetFull augmentation benchmark suite.

This script runs a full experiment campaign on ModelNet10_PLY with:
- multiple single augmentations and combinations,
- per-epoch metrics,
- saved checkpoints and confusion matrices,
- generated figures and notebook reports.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Local import for PLY reader
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from ply import read_ply  # noqa: E402


def set_seed(seed: int) -> None:
    """Fixe toutes les sources de hasard pour rendre les runs comparables."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Initialise les workers DataLoader avec des seeds dérivés de PyTorch."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Compose:
    """Version locale de Compose pour éviter une dépendance torchvision."""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, pointcloud: np.ndarray):
        # Applique chaque transformation séquentiellement.
        for transform in self.transforms:
            pointcloud = transform(pointcloud)
        return pointcloud


class RandomRotationZ:
    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
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


class RandomNoise:
    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0.0, self.std, size=pointcloud.shape).astype(np.float32)
        return pointcloud + noise


class ShufflePoints:
    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        pointcloud = pointcloud.copy()
        np.random.shuffle(pointcloud)
        return pointcloud


class ToTensor:
    def __call__(self, pointcloud: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(pointcloud.astype(np.float32))


class RandomScale:
    def __init__(self, low: float = 0.8, high: float = 1.25):
        self.low = low
        self.high = high

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.low, self.high)
        return pointcloud * np.float32(scale)


class RandomTranslate:
    def __init__(self, shift: float = 0.1):
        self.shift = shift

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        t = np.random.uniform(-self.shift, self.shift, size=(1, 3)).astype(np.float32)
        return pointcloud + t


class RandomJitterClip:
    def __init__(self, sigma: float = 0.01, clip: float = 0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        jitter = self.sigma * np.random.randn(*pointcloud.shape).astype(np.float32)
        jitter = np.clip(jitter, -self.clip, self.clip)
        return pointcloud + jitter


class RandomMirrorXY:
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return pointcloud
        mirrored = pointcloud.copy()
        axis = random.choice([0, 1])
        mirrored[:, axis] = -mirrored[:, axis]
        return mirrored


class RandomPointDropout:
    def __init__(self, max_ratio: float = 0.5):
        self.max_ratio = max_ratio

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        dropout_ratio = np.random.random() * self.max_ratio
        keep = np.random.random(pointcloud.shape[0]) >= dropout_ratio
        out = pointcloud.copy()
        if np.any(~keep):
            out[~keep] = out[0]
        return out


class RandomSphericalOcclusion:
    def __init__(self, radius: float = 0.18, p: float = 0.25):
        self.radius = radius
        self.p = p

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        # Supprime une zone sphérique puis ré-échantillonne pour garder N points.
        if random.random() >= self.p:
            return pointcloud
        n = pointcloud.shape[0]
        center = pointcloud[np.random.randint(0, n)]
        d2 = np.sum((pointcloud - center) ** 2, axis=1)
        keep_idx = np.where(d2 > self.radius**2)[0]
        if keep_idx.size < max(32, n // 10):
            return pointcloud
        kept = pointcloud[keep_idx]
        if kept.shape[0] >= n:
            return kept[:n]
        rep_idx = np.random.choice(kept.shape[0], n - kept.shape[0], replace=True)
        return np.concatenate([kept, kept[rep_idx]], axis=0)


class PointCloudData(Dataset):
    # Cache partagé entre expériences pour ne pas relire les .ply à chaque run.
    _shared_cache: Dict[Tuple[str, str], Dict] = {}

    def __init__(self, root_dir: str, folder: str, transform):
        self.root_dir = root_dir
        self.transform = transform
        cache_key = (os.path.abspath(root_dir), folder)

        # Reuse direct si ce split a déjà été chargé en mémoire.
        if cache_key in PointCloudData._shared_cache:
            cached = PointCloudData._shared_cache[cache_key]
            self.classes = cached["classes"]
            self.points = cached["points"]
            self.labels = cached["labels"]
            self.category_names = cached["category_names"]
            return

        # Chargement initial: on parse tous les PLY du split (train ou test).
        folders = [d for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes = {cat: i for i, cat in enumerate(folders)}

        points = []
        labels = []
        category_names = []
        for category in folders:
            current = os.path.join(root_dir, category, folder)
            for fname in os.listdir(current):
                if not fname.endswith(".ply"):
                    continue
                ply_path = os.path.join(current, fname)
                data = read_ply(ply_path)
                pts = np.vstack((data["x"], data["y"], data["z"])).T.astype(np.float32)
                points.append(pts)
                labels.append(self.classes[category])
                category_names.append(category)

        self.points = points
        self.labels = labels
        self.category_names = category_names
        PointCloudData._shared_cache[cache_key] = {
            "classes": self.classes,
            "points": self.points,
            "labels": self.labels,
            "category_names": self.category_names,
        }

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int):
        points = self.points[idx].copy()
        label = self.labels[idx]
        category = self.category_names[idx]
        points = self.transform(points)
        return {"pointcloud": points, "category": label, "category_name": category}


class Tnet(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.maxpool = nn.MaxPool1d(1024)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).reshape(x.size(0), self.k, self.k)
        ident = torch.eye(self.k, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(x.size(0), 1, 1)
        return x + ident


class PointNetFull(nn.Module):
    # Enonce-compatible full version: only first T-Net (3x3 input transform)
    def __init__(self, classes: int = 10):
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
        self.flatten = nn.Flatten(start_dim=1)
        self.fc6 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc7 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.drop7 = nn.Dropout(0.3)
        self.fc8 = nn.Linear(256, classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        m3x3 = self.tnet1(x)
        x = torch.bmm(m3x3, x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.relu(self.bn6(self.fc6(x)))
        x = self.drop7(self.relu(self.bn7(self.fc7(x))))
        x = self.logsoftmax(self.fc8(x))
        return x, m3x3


def pointnet_full_loss(outputs, labels, m3x3, alpha: float = 0.001):
    """
    Loss PointNet = NLLLoss + régularisation d'orthogonalité du T-Net 3x3.
    """
    criterion = nn.NLLLoss()
    bsize = outputs.size(0)
    id3x3 = torch.eye(3, device=outputs.device, dtype=outputs.dtype).unsqueeze(0).repeat(bsize, 1, 1)
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    return criterion(outputs, labels) + alpha * torch.norm(diff3x3) / float(bsize)


def evaluate(model, loader, device, amp_enabled: bool):
    """Évalue un modèle et retourne loss, accuracy, y_true, y_pred."""
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            x = batch["pointcloud"].to(device, non_blocking=True).float().transpose(1, 2)
            y = batch["category"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out, m3x3 = model(x)
                loss = pointnet_full_loss(out, y, m3x3)

            pred = out.argmax(dim=1)
            bs = y.size(0)
            total += bs
            correct += (pred == y).sum().item()
            loss_sum += loss.item() * bs
            all_true.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    acc = 100.0 * correct / max(1, total)
    avg_loss = loss_sum / max(1, total)
    return avg_loss, acc, y_true, y_pred


def train_one_experiment(
    exp_name: str,
    exp_desc: str,
    train_transform,
    data_root: str,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> Dict:
    """
    Exécute une config d'augmentation de bout en bout:
    train -> sélection du meilleur epoch -> métriques + figures + artefacts.
    """
    set_seed(seed)

    # Important: pas d'augmentation au test (évaluation propre).
    test_transform = Compose([ToTensor()])
    train_ds = PointCloudData(data_root, folder="train", transform=train_transform)
    test_ds = PointCloudData(data_root, folder="test", transform=test_transform)
    class_names = [None] * len(train_ds.classes)
    for name, idx in train_ds.classes.items():
        class_names[idx] = name

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    model = PointNetFull(classes=len(train_ds.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    exp_ckpt_dir = out_dir / "checkpoints"
    exp_hist_dir = out_dir / "histories"
    exp_fig_dir = out_dir / "figures"
    exp_ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_hist_dir.mkdir(parents=True, exist_ok=True)
    exp_fig_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_acc = -1.0
    best_epoch = -1
    best_ckpt_path = exp_ckpt_dir / f"{exp_name}_best.pt"

    training_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        total = 0
        correct = 0
        train_loss_sum = 0.0

        for batch in train_loader:
            x = batch["pointcloud"].to(device, non_blocking=True).float().transpose(1, 2)
            y = batch["category"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out, m3x3 = model(x)
                loss = pointnet_full_loss(out, y, m3x3)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred = out.argmax(dim=1)
            bs = y.size(0)
            total += bs
            correct += (pred == y).sum().item()
            train_loss_sum += loss.item() * bs

        scheduler.step()
        epoch_time = time.time() - epoch_start
        train_acc = 100.0 * correct / max(1, total)
        train_loss = train_loss_sum / max(1, total)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, device, amp_enabled)

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_sec": epoch_time,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(history_row)

        # On sauvegarde le meilleur checkpoint selon la validation.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "classes": class_names,
                    "experiment": exp_name,
                },
                best_ckpt_path,
            )

        print(
            f"[{exp_name}] Epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
            f"time={epoch_time:.2f}s"
        )

    total_time = time.time() - training_start

    # Évaluation finale à partir du meilleur checkpoint.
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    best_val_loss, best_eval_acc, y_true, y_pred = evaluate(model, test_loader, device, amp_enabled)

    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    per_class_acc = []
    for i in range(len(class_names)):
        denom = cm[i].sum()
        per_class_acc.append(float(cm[i, i] / denom) if denom > 0 else 0.0)

    history_csv = exp_hist_dir / f"{exp_name}_history.csv"
    with history_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    history_json = exp_hist_dir / f"{exp_name}_history.json"
    with history_json.open("w") as f:
        json.dump(history, f, indent=2)

    cm_npy = exp_hist_dir / f"{exp_name}_confusion_matrix.npy"
    np.save(cm_npy, cm)

    # Courbes train/val pour cette expérience.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    epochs_axis = [row["epoch"] for row in history]
    axs[0].plot(epochs_axis, [row["train_acc"] for row in history], label="train_acc")
    axs[0].plot(epochs_axis, [row["val_acc"] for row in history], label="val_acc")
    axs[0].set_title(f"{exp_name} - Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(epochs_axis, [row["train_loss"] for row in history], label="train_loss")
    axs[1].plot(epochs_axis, [row["val_loss"] for row in history], label="val_loss")
    axs[1].set_title(f"{exp_name} - Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    fig.tight_layout()
    curve_png = exp_fig_dir / f"{exp_name}_curves.png"
    fig.savefig(curve_png, dpi=160)
    plt.close(fig)

    # Matrice de confusion du meilleur checkpoint.
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{exp_name} - Confusion Matrix (best epoch {best_epoch})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    fig.tight_layout()
    cm_png = exp_fig_dir / f"{exp_name}_confusion_matrix.png"
    fig.savefig(cm_png, dpi=180)
    plt.close(fig)

    return {
        "experiment": exp_name,
        "description": exp_desc,
        "epochs": epochs,
        "seed": seed,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_classes": len(class_names),
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "best_eval_acc": float(best_eval_acc),
        "final_train_acc": float(history[-1]["train_acc"]),
        "final_val_acc": float(history[-1]["val_acc"]),
        "final_train_loss": float(history[-1]["train_loss"]),
        "final_val_loss": float(history[-1]["val_loss"]),
        "avg_epoch_time_sec": float(np.mean([row["epoch_time_sec"] for row in history])),
        "total_train_time_sec": float(total_time),
        "history_csv": str(history_csv.relative_to(out_dir)),
        "history_json": str(history_json.relative_to(out_dir)),
        "curve_png": str(curve_png.relative_to(out_dir)),
        "confusion_png": str(cm_png.relative_to(out_dir)),
        "checkpoint": str(best_ckpt_path.relative_to(out_dir)),
        "class_names": class_names,
        "per_class_acc": per_class_acc,
    }


@dataclass
class ExperimentConfig:
    """Décrit une expérience (nom, description, liste d'augmentations)."""
    name: str
    description: str
    ops: List


def get_experiment_configs() -> List[ExperimentConfig]:
    """
    Définit la grille d'expériences:
    - baseline,
    - ajouts unitaires,
    - combinaisons.
    """
    base = [RandomRotationZ(), RandomNoise(std=0.02)]
    return [
        ExperimentConfig("baseline", "RotationZ + Gaussian noise", base),
        ExperimentConfig("plus_scale", "Baseline + random isotropic scale", base + [RandomScale(0.85, 1.2)]),
        ExperimentConfig("plus_translate", "Baseline + random translation", base + [RandomTranslate(0.1)]),
        ExperimentConfig("plus_jitter", "Baseline + clipped jitter", base + [RandomJitterClip(0.01, 0.04)]),
        ExperimentConfig("plus_mirror", "Baseline + random mirror (x/y)", base + [RandomMirrorXY(p=0.3)]),
        ExperimentConfig("plus_dropout", "Baseline + random point dropout", base + [RandomPointDropout(0.45)]),
        ExperimentConfig(
            "plus_occlusion",
            "Baseline + random spherical occlusion",
            base + [RandomSphericalOcclusion(radius=0.18, p=0.25)],
        ),
        ExperimentConfig(
            "combo_geo",
            "Baseline + scale + translation + jitter + mirror",
            base + [RandomScale(0.85, 1.2), RandomTranslate(0.1), RandomJitterClip(0.01, 0.04), RandomMirrorXY(0.3)],
        ),
        ExperimentConfig(
            "combo_robust",
            "Baseline + scale + translation + point dropout + occlusion",
            base
            + [
                RandomScale(0.85, 1.2),
                RandomTranslate(0.1),
                RandomPointDropout(0.45),
                RandomSphericalOcclusion(0.18, 0.25),
            ],
        ),
        ExperimentConfig(
            "combo_full",
            "Baseline + scale + translation + jitter + mirror + point dropout + occlusion",
            base
            + [
                RandomScale(0.85, 1.2),
                RandomTranslate(0.1),
                RandomJitterClip(0.01, 0.04),
                RandomMirrorXY(0.3),
                RandomPointDropout(0.45),
                RandomSphericalOcclusion(0.18, 0.25),
            ],
        ),
    ]


def save_augmentation_gallery(data_root: str, configs: List[ExperimentConfig], out_path: Path, seed: int) -> None:
    """Crée une planche visuelle montrant l'effet des augmentations."""
    set_seed(seed)
    raw_ds = PointCloudData(data_root, folder="train", transform=lambda x: x)
    idx = random.randint(0, len(raw_ds) - 1)
    raw = raw_ds[idx]["pointcloud"]
    category = raw_ds[idx]["category_name"]

    cols = 4
    rows = int(math.ceil((len(configs) + 1) / cols))
    fig = plt.figure(figsize=(4.2 * cols, 3.8 * rows))

    names = ["original"] + [cfg.name for cfg in configs]
    transforms = [lambda x: x] + [Compose(cfg.ops + [ShufflePoints()]) for cfg in configs]

    for i, (name, tf) in enumerate(zip(names, transforms)):
        pts = tf(raw.copy())
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.5, c=pts[:, 2], cmap="viridis")
        ax.set_title(name, fontsize=9)
        ax.set_axis_off()

    fig.suptitle(f"Augmentation previews (sample idx={idx}, class={category})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_summary_files(results: List[Dict], out_dir: Path) -> Tuple[Path, Path]:
    """Sauvegarde un résumé global des expériences en CSV + JSON."""
    summary_csv = out_dir / "experiment_summary.csv"
    keys = [
        "experiment",
        "description",
        "epochs",
        "best_epoch",
        "best_val_acc",
        "best_val_loss",
        "best_eval_acc",
        "final_train_acc",
        "final_val_acc",
        "final_train_loss",
        "final_val_loss",
        "avg_epoch_time_sec",
        "total_train_time_sec",
        "history_csv",
        "curve_png",
        "confusion_png",
        "checkpoint",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in keys})

    summary_json = out_dir / "experiment_summary.json"
    with summary_json.open("w") as f:
        json.dump(results, f, indent=2)
    return summary_csv, summary_json


def generate_global_plots(results: List[Dict], out_dir: Path) -> Dict[str, str]:
    """Construit les figures de comparaison inter-expériences."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    names = [r["experiment"] for r in results]
    best_acc = [r["best_val_acc"] for r in results]
    avg_epoch_time = [r["avg_epoch_time_sec"] for r in results]

    # Best accuracy bar
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111)
    order = np.argsort(best_acc)[::-1]
    names_sorted = [names[i] for i in order]
    acc_sorted = [best_acc[i] for i in order]
    ax.bar(names_sorted, acc_sorted, color="#2a9d8f")
    ax.set_title("Best validation accuracy by augmentation setup")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Experiment")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    acc_bar_path = fig_dir / "global_best_val_accuracy_bar.png"
    fig.savefig(acc_bar_path, dpi=170)
    plt.close(fig)

    # Accuracy vs speed
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.scatter(avg_epoch_time, best_acc, c="#e76f51", s=65)
    for i, n in enumerate(names):
        ax.annotate(n, (avg_epoch_time[i], best_acc[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title("Accuracy vs average epoch time")
    ax.set_xlabel("Avg epoch time (s)")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    scatter_path = fig_dir / "global_accuracy_vs_time.png"
    fig.savefig(scatter_path, dpi=170)
    plt.close(fig)

    # Overlay validation curves
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111)
    for r in results:
        hist = json.loads((out_dir / r["history_json"]).read_text())
        ax.plot([h["epoch"] for h in hist], [h["val_acc"] for h in hist], label=r["experiment"])
    ax.set_title("Validation accuracy curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    overlay_path = fig_dir / "global_val_accuracy_curves.png"
    fig.savefig(overlay_path, dpi=170)
    plt.close(fig)

    return {
        "global_best_val_accuracy_bar": str(acc_bar_path.relative_to(out_dir)),
        "global_accuracy_vs_time": str(scatter_path.relative_to(out_dir)),
        "global_val_accuracy_curves": str(overlay_path.relative_to(out_dir)),
    }


def generate_report_notebooks(
    out_dir: Path,
    data_root: str,
    device: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    results: List[Dict],
    global_figs: Dict[str, str],
    aug_gallery_relpath: str,
) -> Tuple[Path, Path]:
    """
    Génère 2 notebooks statiques:
    - un résumé global (classement + figures),
    - un log détaillé par expérience.
    """
    notebooks_dir = out_dir / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(results, key=lambda r: r["best_val_acc"], reverse=True)
    best = sorted_results[0]

    table_lines = [
        "| Rank | Experiment | Best Val Acc (%) | Best Epoch | Avg Epoch Time (s) |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, r in enumerate(sorted_results, start=1):
        table_lines.append(
            f"| {rank} | {r['experiment']} | {r['best_val_acc']:.2f} | {r['best_epoch']} | {r['avg_epoch_time_sec']:.2f} |"
        )

    summary_nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# PointNetFull Augmentation Benchmark Report\n",
                    "\n",
                    "Automatic campaign report ready for analysis.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Experimental Setup\n",
                    f"- Dataset: `{data_root}`\n",
                    "- Model: PointNetFull (first 3x3 T-Net only, enonce-compatible)\n",
                    f"- Device: `{device}`\n",
                    f"- Epochs per experiment: `{epochs}`\n",
                    f"- Batch size: `{batch_size}`\n",
                    f"- Num workers: `{num_workers}`\n",
                    f"- Seed: `{seed}`\n",
                    f"- Number of tested augmentation configs: `{len(results)}`\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Ranking by Best Validation Accuracy\n", "\n".join(table_lines) + "\n"],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Augmentation Visual Preview\n",
                    f"![augmentation_gallery](../{aug_gallery_relpath})\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Global Comparison Figures\n",
                    f"![best_val_bar](../{global_figs['global_best_val_accuracy_bar']})\n",
                    f"![acc_vs_time](../{global_figs['global_accuracy_vs_time']})\n",
                    f"![val_curves](../{global_figs['global_val_accuracy_curves']})\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Best Experiment Diagnostics\n",
                    f"- Best setup: `{best['experiment']}`\n",
                    f"- Best val accuracy: `{best['best_val_acc']:.2f}%`\n",
                    f"- Best epoch: `{best['best_epoch']}`\n",
                    f"![best_curves](../{best['curve_png']})\n",
                    f"![best_cm](../{best['confusion_png']})\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Raw Artifacts\n",
                    "- Summary CSV: `../experiment_summary.csv`\n",
                    "- Summary JSON: `../experiment_summary.json`\n",
                    "- Per-experiment history CSV/JSON are in `../histories/`.\n",
                    "- Best checkpoints are in `../checkpoints/`.\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    detail_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# PointNetFull Detailed Experiment Logs\n"],
        }
    ]
    for r in sorted_results:
        detail_cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"## {r['experiment']}\n",
                    f"- Description: {r['description']}\n",
                    f"- Best val acc: {r['best_val_acc']:.2f}% (epoch {r['best_epoch']})\n",
                    f"- Final val acc: {r['final_val_acc']:.2f}%\n",
                    f"- Avg epoch time: {r['avg_epoch_time_sec']:.2f}s\n",
                    f"- History CSV: `../{r['history_csv']}`\n",
                    f"- Checkpoint: `../{r['checkpoint']}`\n",
                    f"![curves](../{r['curve_png']})\n",
                    f"![confusion](../{r['confusion_png']})\n",
                ],
            }
        )

    details_nb = {
        "cells": detail_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    summary_path = notebooks_dir / "01_pointnetfull_augmentation_summary.ipynb"
    detail_path = notebooks_dir / "02_pointnetfull_augmentation_detailed_logs.ipynb"
    summary_path.write_text(json.dumps(summary_nb, indent=1))
    detail_path.write_text(json.dumps(details_nb, indent=1))
    return summary_path, detail_path


def parse_args():
    """Arguments CLI pour lancer la campagne depuis terminal/Colab."""
    parser = argparse.ArgumentParser(description="Run PointNetFull augmentation benchmark suite.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/nochi/NOCHI/M2_PAR/Apprenyissage_Pointcloud/Pointclouds-classification-with-the-POINTNET-Neural-network/PointNetLab/data/ModelNet10_PLY",
        help="Path to ModelNet10_PLY dataset root.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs per experiment.")
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Global seed.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(THIS_DIR / "reports"),
        help="Root folder where report run folder will be created.",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=0,
        help="If >0, run only first N configs (debug mode).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Un dossier horodaté par run pour éviter toute écriture par-dessus.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / f"pointnetfull_aug_suite_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output directory: {out_dir}")

    configs = get_experiment_configs()
    if args.max_experiments > 0:
        configs = configs[: args.max_experiments]
    print(f"Running {len(configs)} experiments")

    gallery_path = out_dir / "figures" / "augmentation_gallery.png"
    save_augmentation_gallery(args.data_root, configs, gallery_path, args.seed)

    results = []
    for idx, cfg in enumerate(configs, start=1):
        print(f"\n=== [{idx}/{len(configs)}] {cfg.name} ===")
        # Pipeline train: augmentations choisies + shuffle des points + conversion tensor.
        train_tf = Compose(cfg.ops + [ShufflePoints(), ToTensor()])
        result = train_one_experiment(
            exp_name=cfg.name,
            exp_desc=cfg.description,
            train_transform=train_tf,
            data_root=args.data_root,
            out_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            seed=args.seed,
        )
        results.append(result)

    results_sorted = sorted(results, key=lambda r: r["best_val_acc"], reverse=True)
    summary_csv, summary_json = write_summary_files(results_sorted, out_dir)
    global_figs = generate_global_plots(results_sorted, out_dir)
    summary_nb, details_nb = generate_report_notebooks(
        out_dir=out_dir,
        data_root=args.data_root,
        device=str(device),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        results=results_sorted,
        global_figs=global_figs,
        aug_gallery_relpath=str(gallery_path.relative_to(out_dir)),
    )

    # Métadonnées minimales pour retrouver rapidement les artefacts du run.
    run_meta = {
        "timestamp": ts,
        "device": str(device),
        "data_root": args.data_root,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "num_experiments": len(results_sorted),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "summary_notebook": str(summary_nb),
        "details_notebook": str(details_nb),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    print("\n=== Completed ===")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary notebook: {summary_nb}")
    print(f"Detailed notebook: {details_nb}")
    print(f"Best setup: {results_sorted[0]['experiment']} ({results_sorted[0]['best_val_acc']:.2f}%)")


if __name__ == "__main__":
    main()
