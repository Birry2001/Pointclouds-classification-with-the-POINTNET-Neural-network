#!/usr/bin/env python3
"""
PointNetFull augmentation benchmark suite (robust experimental protocol).

Main features:
- Auto device selection: TPU > CUDA > CPU (or forced via --device)
- Multi-seed benchmark with per-run isolation (no overwrite)
- Long training with early stopping and optional LR scheduler
- Optional unit-sphere normalization (train + test)
- Sweep mode for augmentation-strength studies (jitter / occlusion / dropout)
- Aggregation mean+-std across seeds
- Automatic plots, notebooks, error analysis, augmentation galleries

Output structure (inside timestamped run directory):
- runs/<experiment>/seed_<seed>/...               : per-run artifacts
- summaries/run_summary.csv|json                  : one row per (experiment, seed)
- summaries/summary_experiments.csv|json          : aggregated mean+-std per experiment
- summaries/summary_experiments_by_eval.csv       : same data sorted by mean(best_eval_acc)
- notebooks/*.ipynb                               : summary notebooks
- results/plots/...                               : comparative plots
- results/top_confusions.csv                      : confusion pairs analysis
- run_metadata.json                               : global run metadata + config
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Helps deterministic CuBLAS behavior when deterministic mode is enabled on CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Local import for PLY reader
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
from ply import read_ply  # noqa: E402

# Optional TPU support (PyTorch/XLA)
try:
    import torch_xla.core.xla_model as xm

    _XLA_AVAILABLE = True
except Exception:
    xm = None  # type: ignore[assignment]
    _XLA_AVAILABLE = False


# -----------------------------
# Runtime / Config Dataclasses
# -----------------------------


@dataclass
class RuntimeContext:
    device: torch.device
    device_kind: str  # cpu | cuda | tpu
    device_label: str
    amp_enabled: bool


@dataclass
class ExperimentConfig:
    name: str
    description: str
    params: Dict[str, float]
    build_ops: Callable[[], List]


@dataclass
class EarlyStoppingConfig:
    monitor: str  # val_acc | val_loss
    patience: int
    min_delta: float


# -----------------------------
# Utility functions
# -----------------------------


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)


def fmt_float(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Determinism can reduce speed but is important for reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------
# Device selection
# -----------------------------


def _tpu_is_available() -> bool:
    if not _XLA_AVAILABLE:
        return False
    try:
        devices = xm.get_xla_supported_devices("TPU")
        return len(devices) > 0
    except Exception:
        return False


def probe_devices() -> Dict[str, Dict[str, str]]:
    cuda_available = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else "n/a"
    tpu_available = _tpu_is_available()
    return {
        "cpu": {"available": "yes", "name": "CPU"},
        "cuda": {"available": "yes" if cuda_available else "no", "name": cuda_name},
        "tpu": {"available": "yes" if tpu_available else "no", "name": "XLA TPU" if tpu_available else "n/a"},
    }


def print_device_probe() -> None:
    info = probe_devices()
    print("Device probe:")
    print(f"- CPU  : available={info['cpu']['available']} ({info['cpu']['name']})")
    print(f"- CUDA : available={info['cuda']['available']} ({info['cuda']['name']})")
    print(f"- TPU  : available={info['tpu']['available']} ({info['tpu']['name']})")


def select_runtime(device_request: str) -> RuntimeContext:
    info = probe_devices()

    req = device_request
    if req == "auto":
        if info["tpu"]["available"] == "yes":
            req = "tpu"
        elif info["cuda"]["available"] == "yes":
            req = "cuda"
        else:
            req = "cpu"

    if req == "tpu":
        if info["tpu"]["available"] != "yes":
            raise RuntimeError("TPU requested but unavailable.")
        device = xm.xla_device()
        return RuntimeContext(device=device, device_kind="tpu", device_label=str(device), amp_enabled=False)

    if req == "cuda":
        if info["cuda"]["available"] != "yes":
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device("cuda:0")
        return RuntimeContext(
            device=device,
            device_kind="cuda",
            device_label=f"cuda:0 ({info['cuda']['name']})",
            amp_enabled=True,
        )

    return RuntimeContext(device=torch.device("cpu"), device_kind="cpu", device_label="cpu", amp_enabled=False)


# -----------------------------
# Transforms
# -----------------------------


class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, pointcloud: np.ndarray):
        for t in self.transforms:
            pointcloud = t(pointcloud)
        return pointcloud


class UnitSphereNormalize:
    """Center to centroid then scale to unit sphere."""

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        centered = pointcloud - np.mean(pointcloud, axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        m = float(np.max(norms)) if norms.size else 1.0
        if m < 1e-12:
            return centered.astype(np.float32)
        return (centered / m).astype(np.float32)


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
        out = pointcloud.copy()
        np.random.shuffle(out)
        return out


class ToTensor:
    def __call__(self, pointcloud: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(pointcloud.astype(np.float32))


class RandomScale:
    def __init__(self, low: float = 0.8, high: float = 1.25):
        self.low = low
        self.high = high

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        s = np.random.uniform(self.low, self.high)
        return pointcloud * np.float32(s)


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
        jit = self.sigma * np.random.randn(*pointcloud.shape).astype(np.float32)
        jit = np.clip(jit, -self.clip, self.clip)
        return pointcloud + jit


class RandomMirrorXY:
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return pointcloud
        out = pointcloud.copy()
        axis = random.choice([0, 1])
        out[:, axis] = -out[:, axis]
        return out


class RandomPointDropout:
    def __init__(self, max_ratio: float = 0.45):
        self.max_ratio = max_ratio

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        ratio = np.random.random() * self.max_ratio
        keep = np.random.random(pointcloud.shape[0]) >= ratio
        out = pointcloud.copy()
        if np.any(~keep):
            out[~keep] = out[0]
        return out


class RandomSphericalOcclusion:
    def __init__(self, radius: float = 0.18, p: float = 0.25):
        self.radius = radius
        self.p = p

    def __call__(self, pointcloud: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return pointcloud
        n = pointcloud.shape[0]
        if n == 0:
            return pointcloud
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


# -----------------------------
# Dataset
# -----------------------------


class PointCloudData(Dataset):
    """
    In-memory cached dataset to avoid repeated disk IO between runs/seeds.
    """

    _shared_cache: Dict[Tuple[str, str], Dict] = {}

    def __init__(self, root_dir: str, folder: str, transform):
        self.root_dir = root_dir
        self.folder = folder
        self.transform = transform
        cache_key = (os.path.abspath(root_dir), folder)

        if cache_key in PointCloudData._shared_cache:
            c = PointCloudData._shared_cache[cache_key]
            self.classes = c["classes"]
            self.points = c["points"]
            self.labels = c["labels"]
            self.category_names = c["category_names"]
            self.paths = c["paths"]
            return

        root = Path(root_dir)
        class_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
        self.classes = {d.name: i for i, d in enumerate(class_dirs)}

        points: List[np.ndarray] = []
        labels: List[int] = []
        category_names: List[str] = []
        paths: List[str] = []

        for cdir in class_dirs:
            split_dir = cdir / folder
            if not split_dir.exists():
                continue
            ply_paths = sorted([p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".ply"])
            for ply_path in ply_paths:
                data = read_ply(str(ply_path))
                pts = np.vstack((data["x"], data["y"], data["z"])).T.astype(np.float32)
                points.append(pts)
                labels.append(self.classes[cdir.name])
                category_names.append(cdir.name)
                paths.append(str(ply_path))

        self.points = points
        self.labels = labels
        self.category_names = category_names
        self.paths = paths

        PointCloudData._shared_cache[cache_key] = {
            "classes": self.classes,
            "points": self.points,
            "labels": self.labels,
            "category_names": self.category_names,
            "paths": self.paths,
        }

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int):
        pts = self.points[idx].copy()
        label = self.labels[idx]
        cat = self.category_names[idx]
        path = self.paths[idx]
        pts = self.transform(pts)
        return {
            "pointcloud": pts,
            "category": label,
            "category_name": cat,
            "index": idx,
            "ply_path": path,
        }


def inspect_dataset_root(data_root: str) -> Dict[str, int]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {data_root}\n"
            "Check --data-root and mounted storage path."
        )

    class_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(
            f"No class directories in: {data_root}\n"
            "Expected: .../ModelNet*_PLY/<class_name>/train and /test"
        )

    train_ply = 0
    test_ply = 0
    for cdir in class_dirs:
        train_dir = cdir / "train"
        test_dir = cdir / "test"
        if train_dir.exists():
            train_ply += sum(1 for p in train_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".ply")
        if test_dir.exists():
            test_ply += sum(1 for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".ply")

    if train_ply == 0 or test_ply == 0:
        raise RuntimeError(
            "Dataset seems empty or malformed for ModelNet PLY.\n"
            f"Path: {data_root}\n"
            f"Detected classes: {len(class_dirs)} | train .ply: {train_ply} | test .ply: {test_ply}\n"
            "Expected: .../ModelNet10_PLY/<class_name>/train/*.ply and .../test/*.ply"
        )

    return {"num_classes": len(class_dirs), "train_ply": train_ply, "test_ply": test_ply}


# -----------------------------
# Model / Loss
# -----------------------------


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
    """Enonce-compatible full version: first T-Net (3x3) only."""

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
    criterion = nn.NLLLoss()
    bsize = outputs.size(0)
    id3x3 = torch.eye(3, device=outputs.device, dtype=outputs.dtype).unsqueeze(0).repeat(bsize, 1, 1)
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    return criterion(outputs, labels) + alpha * torch.norm(diff3x3) / float(bsize)


# -----------------------------
# Experiments definitions
# -----------------------------


DEFAULT_TOP_EXPERIMENTS = ["baseline", "plus_jitter", "plus_occlusion", "combo_full"]


def build_base_experiment_registry() -> Dict[str, ExperimentConfig]:
    registry: Dict[str, ExperimentConfig] = {}

    def reg(cfg: ExperimentConfig):
        registry[cfg.name] = cfg

    reg(
        ExperimentConfig(
            name="baseline",
            description="RotationZ + Gaussian noise",
            params={"rot_z": 1.0, "noise_std": 0.02},
            build_ops=lambda: [RandomRotationZ(), RandomNoise(std=0.02)],
        )
    )

    reg(
        ExperimentConfig(
            name="plus_scale",
            description="Baseline + random isotropic scale",
            params={"scale_low": 0.85, "scale_high": 1.2},
            build_ops=lambda: [RandomRotationZ(), RandomNoise(0.02), RandomScale(0.85, 1.2)],
        )
    )

    reg(
        ExperimentConfig(
            name="plus_translate",
            description="Baseline + random translation",
            params={"translate_shift": 0.1},
            build_ops=lambda: [RandomRotationZ(), RandomNoise(0.02), RandomTranslate(0.1)],
        )
    )

    reg(
        ExperimentConfig(
            name="plus_jitter",
            description="Baseline + clipped jitter",
            params={"jitter_sigma": 0.01, "jitter_clip": 0.04},
            build_ops=lambda: [RandomRotationZ(), RandomNoise(0.02), RandomJitterClip(0.01, 0.04)],
        )
    )

    reg(
        ExperimentConfig(
            name="plus_mirror",
            description="Baseline + random mirror XY",
            params={"mirror_p": 0.3},
            build_ops=lambda: [RandomRotationZ(), RandomNoise(0.02), RandomMirrorXY(0.3)],
        )
    )

    reg(
        ExperimentConfig(
            name="plus_dropout",
            description="Baseline + random point dropout",
            params={"dropout_max_ratio": 0.45},
            build_ops=lambda: [RandomRotationZ(), RandomNoise(0.02), RandomPointDropout(0.45)],
        )
    )

    reg(
        ExperimentConfig(
            name="plus_occlusion",
            description="Baseline + random spherical occlusion",
            params={"occlusion_radius": 0.18, "occlusion_p": 0.25},
            build_ops=lambda: [RandomRotationZ(), RandomNoise(0.02), RandomSphericalOcclusion(0.18, 0.25)],
        )
    )

    reg(
        ExperimentConfig(
            name="combo_geo",
            description="Baseline + scale + translation + jitter + mirror",
            params={
                "scale_low": 0.85,
                "scale_high": 1.2,
                "translate_shift": 0.1,
                "jitter_sigma": 0.01,
                "jitter_clip": 0.04,
                "mirror_p": 0.3,
            },
            build_ops=lambda: [
                RandomRotationZ(),
                RandomNoise(0.02),
                RandomScale(0.85, 1.2),
                RandomTranslate(0.1),
                RandomJitterClip(0.01, 0.04),
                RandomMirrorXY(0.3),
            ],
        )
    )

    reg(
        ExperimentConfig(
            name="combo_robust",
            description="Baseline + scale + translate + dropout + occlusion",
            params={
                "scale_low": 0.85,
                "scale_high": 1.2,
                "translate_shift": 0.1,
                "dropout_max_ratio": 0.45,
                "occlusion_radius": 0.18,
                "occlusion_p": 0.25,
            },
            build_ops=lambda: [
                RandomRotationZ(),
                RandomNoise(0.02),
                RandomScale(0.85, 1.2),
                RandomTranslate(0.1),
                RandomPointDropout(0.45),
                RandomSphericalOcclusion(0.18, 0.25),
            ],
        )
    )

    reg(
        ExperimentConfig(
            name="combo_full",
            description="Baseline + scale + translate + jitter + mirror + dropout + occlusion",
            params={
                "scale_low": 0.85,
                "scale_high": 1.2,
                "translate_shift": 0.1,
                "jitter_sigma": 0.01,
                "jitter_clip": 0.04,
                "mirror_p": 0.3,
                "dropout_max_ratio": 0.45,
                "occlusion_radius": 0.18,
                "occlusion_p": 0.25,
            },
            build_ops=lambda: [
                RandomRotationZ(),
                RandomNoise(0.02),
                RandomScale(0.85, 1.2),
                RandomTranslate(0.1),
                RandomJitterClip(0.01, 0.04),
                RandomMirrorXY(0.3),
                RandomPointDropout(0.45),
                RandomSphericalOcclusion(0.18, 0.25),
            ],
        )
    )

    return registry


def build_sweep_configs(sweep_name: str) -> List[ExperimentConfig]:
    out: List[ExperimentConfig] = []

    if sweep_name == "jitter":
        for sigma, clip in [(0.005, 0.02), (0.01, 0.04), (0.02, 0.05)]:
            name = f"jitter_s{fmt_float(sigma)}_c{fmt_float(clip)}"
            params = {"sweep": "jitter", "sigma": sigma, "clip": clip}
            out.append(
                ExperimentConfig(
                    name=name,
                    description=f"Baseline + jitter sweep sigma={sigma}, clip={clip}",
                    params=params,
                    build_ops=lambda sigma=sigma, clip=clip: [
                        RandomRotationZ(),
                        RandomNoise(0.02),
                        RandomJitterClip(sigma=sigma, clip=clip),
                    ],
                )
            )

    elif sweep_name == "occlusion":
        for p in [0.15, 0.25, 0.35]:
            for radius in [0.12, 0.18, 0.25]:
                name = f"occ_p{fmt_float(p)}_r{fmt_float(radius)}"
                params = {"sweep": "occlusion", "p": p, "radius": radius}
                out.append(
                    ExperimentConfig(
                        name=name,
                        description=f"Baseline + occlusion sweep p={p}, radius={radius}",
                        params=params,
                        build_ops=lambda p=p, radius=radius: [
                            RandomRotationZ(),
                            RandomNoise(0.02),
                            RandomSphericalOcclusion(radius=radius, p=p),
                        ],
                    )
                )

    elif sweep_name == "dropout":
        for ratio in [0.2, 0.35, 0.45]:
            name = f"drop_r{fmt_float(ratio)}"
            params = {"sweep": "dropout", "max_ratio": ratio}
            out.append(
                ExperimentConfig(
                    name=name,
                    description=f"Baseline + dropout sweep max_ratio={ratio}",
                    params=params,
                    build_ops=lambda ratio=ratio: [
                        RandomRotationZ(),
                        RandomNoise(0.02),
                        RandomPointDropout(max_ratio=ratio),
                    ],
                )
            )

    else:
        raise ValueError(f"Unknown sweep: {sweep_name}")

    return out


# -----------------------------
# Data pipeline helpers
# -----------------------------


def build_transform_pipeline(
    train_ops: Optional[List],
    normalize_unit_sphere: bool,
    for_train: bool,
) -> Compose:
    ops: List = []
    if normalize_unit_sphere:
        ops.append(UnitSphereNormalize())
    if train_ops is not None:
        ops.extend(train_ops)
    if for_train:
        ops.append(ShufflePoints())
    ops.append(ToTensor())
    return Compose(ops)


def create_dataloaders(
    data_root: str,
    normalize_unit_sphere: bool,
    train_ops: List,
    batch_size: int,
    num_workers: int,
    runtime: RuntimeContext,
    seed: int,
):
    train_tf = build_transform_pipeline(train_ops=train_ops, normalize_unit_sphere=normalize_unit_sphere, for_train=True)
    test_tf = build_transform_pipeline(train_ops=None, normalize_unit_sphere=normalize_unit_sphere, for_train=False)

    train_ds = PointCloudData(data_root, folder="train", transform=train_tf)
    test_ds = PointCloudData(data_root, folder="test", transform=test_tf)

    class_names = [None] * len(train_ds.classes)
    for cname, cidx in train_ds.classes.items():
        class_names[cidx] = cname

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(runtime.device_kind == "cuda"),
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(runtime.device_kind == "cuda"),
        persistent_workers=False,
    )

    return train_ds, test_ds, train_loader, test_loader, class_names


# -----------------------------
# Train / Eval core
# -----------------------------


def metric_improved(monitor: str, best_value: Optional[float], current: float, min_delta: float) -> bool:
    if best_value is None:
        return True
    if monitor == "val_acc":
        return current > (best_value + min_delta)
    return current < (best_value - min_delta)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_per_class_acc(cm: np.ndarray) -> List[float]:
    out: List[float] = []
    for i in range(cm.shape[0]):
        denom = cm[i].sum()
        out.append(float(cm[i, i] / denom) if denom > 0 else 0.0)
    return out


def evaluate_model(
    model,
    loader,
    runtime: RuntimeContext,
    class_names: List[str],
    return_predictions: bool = False,
):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    y_true_all = []
    y_pred_all = []
    pred_rows: List[Dict] = []

    with torch.no_grad():
        for batch in loader:
            x = (
                batch["pointcloud"]
                .to(runtime.device, non_blocking=(runtime.device_kind == "cuda"))
                .float()
                .transpose(1, 2)
            )
            y = batch["category"].to(runtime.device, non_blocking=(runtime.device_kind == "cuda"))

            autocast_ctx = torch.cuda.amp.autocast(enabled=True) if runtime.amp_enabled else nullcontext()
            with autocast_ctx:
                out, m3x3 = model(x)
                loss = pointnet_full_loss(out, y, m3x3)

            probs = torch.exp(out)
            conf, pred = torch.max(probs, dim=1)

            bs = y.size(0)
            total += bs
            correct += (pred == y).sum().item()
            loss_sum += loss.item() * bs

            y_cpu = y.detach().cpu().numpy()
            p_cpu = pred.detach().cpu().numpy()
            c_cpu = conf.detach().cpu().numpy()

            y_true_all.append(y_cpu)
            y_pred_all.append(p_cpu)

            if return_predictions:
                idx_cpu = batch["index"].detach().cpu().numpy() if torch.is_tensor(batch["index"]) else np.array(batch["index"])
                paths = list(batch["ply_path"])
                for i in range(bs):
                    t = int(y_cpu[i])
                    p = int(p_cpu[i])
                    pred_rows.append(
                        {
                            "index": int(idx_cpu[i]),
                            "ply_path": paths[i],
                            "true": t,
                            "pred": p,
                            "true_name": class_names[t],
                            "pred_name": class_names[p],
                            "confidence": float(c_cpu[i]),
                        }
                    )

            if runtime.device_kind == "tpu":
                xm.mark_step()

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)

    acc = 100.0 * correct / max(1, total)
    avg_loss = loss_sum / max(1, total)

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "y_true": y_true,
        "y_pred": y_pred,
        "pred_rows": pred_rows,
    }


def save_history(history: List[Dict], csv_path: Path, json_path: Path) -> None:
    if not history:
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    json_path.write_text(json.dumps(history, indent=2))


def plot_run_curves(history: List[Dict], out_png: Path, title_prefix: str) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(epochs, [h["train_acc"] for h in history], label="train_acc")
    axs[0].plot(epochs, [h["val_acc"] for h in history], label="val_acc")
    axs[0].set_title(f"{title_prefix} - Accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(epochs, [h["train_loss"] for h in history], label="train_loss")
    axs[1].plot(epochs, [h["val_loss"] for h in history], label="val_loss")
    axs[1].set_title(f"{title_prefix} - Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, class_names: List[str], out_png: Path, title: str) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_predictions_csv(pred_rows: List[Dict], out_csv: Path, experiment: str, seed: int) -> None:
    if not pred_rows:
        out_csv.write_text("")
        return

    rows = []
    for r in pred_rows:
        row = dict(r)
        row["experiment"] = experiment
        row["seed"] = seed
        rows.append(row)

    fields = [
        "experiment",
        "seed",
        "index",
        "ply_path",
        "true",
        "pred",
        "true_name",
        "pred_name",
        "confidence",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_one_experiment_seed(
    exp_cfg: ExperimentConfig,
    seed: int,
    args,
    runtime: RuntimeContext,
    out_dir: Path,
    early_cfg: EarlyStoppingConfig,
) -> Dict:
    set_seed(seed, deterministic=args.deterministic)

    run_dir = out_dir / "runs" / exp_cfg.name / f"seed_{seed}"
    ensure_dir(run_dir)

    train_ds, test_ds, train_loader, test_loader, class_names = create_dataloaders(
        data_root=args.data_root,
        normalize_unit_sphere=args.normalize_unit_sphere,
        train_ops=exp_cfg.build_ops(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        runtime=runtime,
        seed=seed,
    )

    model = PointNetFull(classes=len(class_names)).to(runtime.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma,
        )

    scaler = torch.cuda.amp.GradScaler(enabled=runtime.amp_enabled)

    best_value: Optional[float] = None
    best_epoch = -1
    best_checkpoint = run_dir / "checkpoint_best.pt"
    history: List[Dict] = []
    no_improve_count = 0

    start_time = time.time()

    for epoch in range(1, args.epochs_max + 1):
        model.train()
        epoch_start = time.time()
        total = 0
        correct = 0
        train_loss_sum = 0.0

        for batch in train_loader:
            x = (
                batch["pointcloud"]
                .to(runtime.device, non_blocking=(runtime.device_kind == "cuda"))
                .float()
                .transpose(1, 2)
            )
            y = batch["category"].to(runtime.device, non_blocking=(runtime.device_kind == "cuda"))

            optimizer.zero_grad()
            autocast_ctx = torch.cuda.amp.autocast(enabled=True) if runtime.amp_enabled else nullcontext()
            with autocast_ctx:
                out, m3x3 = model(x)
                loss = pointnet_full_loss(out, y, m3x3)

            if runtime.amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if runtime.device_kind == "tpu":
                    xm.optimizer_step(optimizer, barrier=False)
                    xm.mark_step()
                else:
                    optimizer.step()

            pred = out.argmax(dim=1)
            bs = y.size(0)
            total += bs
            correct += (pred == y).sum().item()
            train_loss_sum += loss.item() * bs

        if scheduler is not None:
            scheduler.step()

        train_loss = train_loss_sum / max(1, total)
        train_acc = 100.0 * correct / max(1, total)

        ev = evaluate_model(model, test_loader, runtime, class_names=class_names, return_predictions=False)
        val_loss = ev["loss"]
        val_acc = ev["acc"]

        monitor_value = val_acc if early_cfg.monitor == "val_acc" else val_loss
        improved = metric_improved(early_cfg.monitor, best_value, monitor_value, early_cfg.min_delta)

        if improved:
            best_value = monitor_value
            best_epoch = epoch
            no_improve_count = 0
            payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_monitor_value": best_value,
                "classes": class_names,
                "experiment": exp_cfg.name,
                "seed": seed,
            }
            if runtime.device_kind == "tpu":
                xm.save(payload, str(best_checkpoint))
            else:
                torch.save(payload, best_checkpoint)
        else:
            no_improve_count += 1

        epoch_row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": float(time.time() - epoch_start),
            "is_best": int(improved),
            "no_improve_count": int(no_improve_count),
        }
        history.append(epoch_row)

        print(
            f"[{exp_cfg.name}][seed={seed}] "
            f"Epoch {epoch:03d}/{args.epochs_max} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
            f"best_epoch={best_epoch} no_improve={no_improve_count}/{early_cfg.patience}"
        )

        if no_improve_count >= early_cfg.patience:
            print(
                f"[{exp_cfg.name}][seed={seed}] Early stopping at epoch {epoch} "
                f"(monitor={early_cfg.monitor})."
            )
            break

    total_time = time.time() - start_time

    # Evaluate using best checkpoint
    ckpt = torch.load(best_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    final_eval = evaluate_model(model, test_loader, runtime, class_names=class_names, return_predictions=True)
    y_true = final_eval["y_true"]
    y_pred = final_eval["y_pred"]

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    per_class_acc = compute_per_class_acc(cm)

    history_csv = run_dir / "history.csv"
    history_json = run_dir / "history.json"
    curves_png = run_dir / "curves.png"
    cm_png = run_dir / "confusion_matrix.png"
    cm_npy = run_dir / "confusion_matrix.npy"
    predictions_csv = run_dir / "predictions.csv"

    save_history(history, history_csv, history_json)
    plot_run_curves(history, curves_png, title_prefix=f"{exp_cfg.name} | seed {seed}")
    plot_confusion(cm, class_names, cm_png, title=f"{exp_cfg.name} | seed {seed} | best epoch {best_epoch}")
    np.save(cm_npy, cm)
    save_predictions_csv(final_eval["pred_rows"], predictions_csv, exp_cfg.name, seed)

    run_result = {
        "experiment": exp_cfg.name,
        "description": exp_cfg.description,
        "params": exp_cfg.params,
        "seed": seed,
        "epochs_max": args.epochs_max,
        "epochs_ran": len(history),
        "early_stop_monitor": early_cfg.monitor,
        "early_stop_patience": early_cfg.patience,
        "early_stop_min_delta": early_cfg.min_delta,
        "stopped_early": int(len(history) < args.epochs_max),
        "best_epoch": int(best_epoch),
        "best_val_acc": float(max(h["val_acc"] for h in history)),
        "best_val_loss": float(min(h["val_loss"] for h in history)),
        "best_eval_acc": float(final_eval["acc"]),
        "best_eval_loss": float(final_eval["loss"]),
        "final_val_acc": float(history[-1]["val_acc"]),
        "final_train_acc": float(history[-1]["train_acc"]),
        "final_val_loss": float(history[-1]["val_loss"]),
        "final_train_loss": float(history[-1]["train_loss"]),
        "avg_epoch_time_sec": float(np.mean([h["epoch_time_sec"] for h in history])),
        "total_train_time_sec": float(total_time),
        "num_classes": len(class_names),
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "class_names": class_names,
        "per_class_acc": per_class_acc,
        "mean_best_per_class_acc": float(np.mean(per_class_acc) if per_class_acc else 0.0),
        "run_dir": str(run_dir.relative_to(out_dir)),
        "history_csv": str(history_csv.relative_to(out_dir)),
        "history_json": str(history_json.relative_to(out_dir)),
        "curves_png": str(curves_png.relative_to(out_dir)),
        "confusion_png": str(cm_png.relative_to(out_dir)),
        "confusion_npy": str(cm_npy.relative_to(out_dir)),
        "predictions_csv": str(predictions_csv.relative_to(out_dir)),
        "checkpoint": str(best_checkpoint.relative_to(out_dir)),
    }

    (run_dir / "run_result.json").write_text(json.dumps(run_result, indent=2))
    return run_result


# -----------------------------
# Aggregation
# -----------------------------


AGG_METRICS = [
    "best_val_acc",
    "best_eval_acc",
    "best_val_loss",
    "best_epoch",
    "final_val_acc",
    "final_train_acc",
    "final_val_loss",
    "final_train_loss",
    "mean_best_per_class_acc",
]


def aggregate_results(run_results: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for r in run_results:
        grouped[r["experiment"]].append(r)

    agg_records: List[Dict] = []

    for exp, rows in grouped.items():
        out: Dict = {
            "experiment": exp,
            "description": rows[0]["description"],
            "params": rows[0].get("params", {}),
            "num_runs": len(rows),
            "seeds": sorted([int(r["seed"]) for r in rows]),
            "class_names": rows[0].get("class_names", []),
        }

        for m in AGG_METRICS:
            vals = [float(r[m]) for r in rows]
            out[f"mean_{m}"] = float(np.mean(vals))
            out[f"std_{m}"] = float(np.std(vals, ddof=0))

        # Per-class mean/std across seeds
        per_class = [np.array(r.get("per_class_acc", []), dtype=np.float32) for r in rows]
        if per_class and all(pc.shape == per_class[0].shape for pc in per_class):
            arr = np.stack(per_class, axis=0)
            out["per_class_acc_mean"] = arr.mean(axis=0).astype(float).tolist()
            out["per_class_acc_std"] = arr.std(axis=0).astype(float).tolist()
            out["mean_best_per_class_acc"] = float(arr.mean(axis=0).mean())
        else:
            out["per_class_acc_mean"] = []
            out["per_class_acc_std"] = []

        agg_records.append(out)

    return agg_records


def write_run_summary(run_results: List[Dict], summaries_dir: Path) -> Tuple[Path, Path]:
    ensure_dir(summaries_dir)
    csv_path = summaries_dir / "run_summary.csv"
    json_path = summaries_dir / "run_summary.json"

    if not run_results:
        csv_path.write_text("")
        json_path.write_text("[]")
        return csv_path, json_path

    fields = [
        "experiment",
        "seed",
        "epochs_max",
        "epochs_ran",
        "stopped_early",
        "best_epoch",
        "best_val_acc",
        "best_eval_acc",
        "best_val_loss",
        "final_val_acc",
        "final_train_acc",
        "final_val_loss",
        "final_train_loss",
        "mean_best_per_class_acc",
        "avg_epoch_time_sec",
        "total_train_time_sec",
        "run_dir",
        "history_csv",
        "curves_png",
        "confusion_png",
        "predictions_csv",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in run_results:
            writer.writerow({k: r.get(k, "") for k in fields})

    json_path.write_text(json.dumps(run_results, indent=2))
    return csv_path, json_path


def write_experiment_summary(agg_records: List[Dict], summaries_dir: Path) -> Dict[str, Path]:
    ensure_dir(summaries_dir)

    sort_by_val = sorted(agg_records, key=lambda x: x["mean_best_val_acc"], reverse=True)
    sort_by_eval = sorted(agg_records, key=lambda x: x["mean_best_eval_acc"], reverse=True)

    csv_val = summaries_dir / "summary_experiments.csv"
    json_val = summaries_dir / "summary_experiments.json"
    csv_eval = summaries_dir / "summary_experiments_by_eval.csv"

    fields = [
        "experiment",
        "num_runs",
        "seeds",
        "mean_best_val_acc",
        "std_best_val_acc",
        "mean_best_eval_acc",
        "std_best_eval_acc",
        "mean_best_val_loss",
        "std_best_val_loss",
        "mean_best_epoch",
        "std_best_epoch",
        "mean_final_val_acc",
        "std_final_val_acc",
        "mean_final_train_acc",
        "std_final_train_acc",
        "mean_mean_best_per_class_acc",
        "std_mean_best_per_class_acc",
        "params",
    ]

    def _write_csv(path: Path, rows: List[Dict]):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                row = {k: r.get(k, "") for k in fields}
                row["seeds"] = " ".join(str(s) for s in r.get("seeds", []))
                row["params"] = json.dumps(r.get("params", {}), sort_keys=True)
                writer.writerow(row)

    _write_csv(csv_val, sort_by_val)
    _write_csv(csv_eval, sort_by_eval)
    json_val.write_text(json.dumps(sort_by_val, indent=2))

    return {
        "summary_experiments_csv": csv_val,
        "summary_experiments_json": json_val,
        "summary_experiments_by_eval_csv": csv_eval,
    }


# -----------------------------
# Plots (comparative)
# -----------------------------


def _bar_mean_std(agg_records: List[Dict], mean_key: str, std_key: str, title: str, out_png: Path) -> None:
    labels = [r["experiment"] for r in agg_records]
    means = [r[mean_key] for r in agg_records]
    stds = [r[std_key] for r in agg_records]

    fig = plt.figure(figsize=(max(8, len(labels) * 1.3), 5))
    ax = fig.add_subplot(111)
    ax.bar(labels, means, yerr=stds, capsize=4, color="#2a9d8f")
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _load_history_metric(history_json_path: Path, metric: str) -> List[float]:
    hist = json.loads(history_json_path.read_text())
    return [float(h[metric]) for h in hist]


def _plot_mean_std_over_epochs(run_results: List[Dict], metric: str, title: str, out_png: Path) -> None:
    grouped: Dict[str, List[List[float]]] = defaultdict(list)
    base_dir = out_png.parent.parent.parent  # out_dir

    for r in run_results:
        h_path = base_dir / r["history_json"]
        grouped[r["experiment"]].append(_load_history_metric(h_path, metric))

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)

    for exp, series_list in grouped.items():
        max_len = max(len(s) for s in series_list)
        arr = np.full((len(series_list), max_len), np.nan, dtype=np.float32)
        for i, s in enumerate(series_list):
            arr[i, : len(s)] = s
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        epochs = np.arange(1, max_len + 1)

        valid = ~np.isnan(mean)
        ax.plot(epochs[valid], mean[valid], label=exp)
        ax.fill_between(epochs[valid], mean[valid] - std[valid], mean[valid] + std[valid], alpha=0.18)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_best_val_boxplot(run_results: List[Dict], out_png: Path) -> None:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for r in run_results:
        grouped[r["experiment"]].append(float(r["best_val_acc"]))

    labels = sorted(grouped.keys())
    data = [grouped[k] for k in labels]

    fig = plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    ax = fig.add_subplot(111)
    # Matplotlib 3.9+ renamed `labels` -> `tick_labels`.
    try:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title("Best validation accuracy distribution by experiment")
    ax.set_ylabel("best_val_acc (%)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def generate_multi_seed_plots(
    run_results: List[Dict],
    agg_records: List[Dict],
    plots_dir: Path,
) -> Dict[str, str]:
    ensure_dir(plots_dir)

    sorted_val = sorted(agg_records, key=lambda r: r["mean_best_val_acc"], reverse=True)

    p1 = plots_dir / "best_val_acc_mean_std.png"
    _bar_mean_std(sorted_val, "mean_best_val_acc", "std_best_val_acc", "best_val_acc mean+-std", p1)

    p2 = plots_dir / "best_eval_acc_mean_std.png"
    _bar_mean_std(sorted_val, "mean_best_eval_acc", "std_best_eval_acc", "best_eval_acc mean+-std", p2)

    p3 = plots_dir / "final_val_acc_mean_std.png"
    _bar_mean_std(sorted_val, "mean_final_val_acc", "std_final_val_acc", "final_val_acc mean+-std", p3)

    p4 = plots_dir / "val_acc_mean_std_over_epochs.png"
    _plot_mean_std_over_epochs(run_results, "val_acc", "val_acc mean+-std over epochs", p4)

    p5 = plots_dir / "val_loss_mean_std_over_epochs.png"
    _plot_mean_std_over_epochs(run_results, "val_loss", "val_loss mean+-std over epochs", p5)

    p6 = plots_dir / "train_acc_mean_std_over_epochs.png"
    _plot_mean_std_over_epochs(run_results, "train_acc", "train_acc mean+-std over epochs", p6)

    p7 = plots_dir / "best_val_acc_boxplot.png"
    _plot_best_val_boxplot(run_results, p7)

    return {
        "best_val_acc_mean_std": str(p1),
        "best_eval_acc_mean_std": str(p2),
        "final_val_acc_mean_std": str(p3),
        "val_acc_mean_std_over_epochs": str(p4),
        "val_loss_mean_std_over_epochs": str(p5),
        "train_acc_mean_std_over_epochs": str(p6),
        "best_val_acc_boxplot": str(p7),
    }


def generate_sweep_plots(agg_records: List[Dict], sweep_name: str, sweep_plot_dir: Path) -> Dict[str, str]:
    ensure_dir(sweep_plot_dir)
    out: Dict[str, str] = {}

    ranked = sorted(agg_records, key=lambda r: r["mean_best_val_acc"], reverse=True)
    labels = [r["experiment"] for r in ranked]
    means = [r["mean_best_val_acc"] for r in ranked]
    stds = [r["std_best_val_acc"] for r in ranked]

    p_rank = sweep_plot_dir / "sweep_ranked_best_val_acc.png"
    fig = plt.figure(figsize=(max(8, len(labels) * 0.8), 5))
    ax = fig.add_subplot(111)
    ax.bar(labels, means, yerr=stds, capsize=4, color="#264653")
    ax.set_title(f"Sweep {sweep_name} - ranked by mean(best_val_acc)")
    ax.set_ylabel("mean best_val_acc (%)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(p_rank, dpi=170)
    plt.close(fig)
    out["sweep_ranked_best_val_acc"] = str(p_rank)

    if sweep_name == "jitter":
        sigmas = sorted({float(r["params"].get("sigma")) for r in agg_records})
        clips = sorted({float(r["params"].get("clip")) for r in agg_records})
        M = np.full((len(sigmas), len(clips)), np.nan, dtype=np.float32)
        for r in agg_records:
            s = float(r["params"].get("sigma"))
            c = float(r["params"].get("clip"))
            i = sigmas.index(s)
            j = clips.index(c)
            M[i, j] = float(r["mean_best_val_acc"])

        p = sweep_plot_dir / "sweep_jitter_sigma_vs_clip_heatmap.png"
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(M, cmap="viridis", origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(clips)))
        ax.set_yticks(np.arange(len(sigmas)))
        ax.set_xticklabels([fmt_float(x) for x in clips])
        ax.set_yticklabels([fmt_float(x) for x in sigmas])
        ax.set_xlabel("clip")
        ax.set_ylabel("sigma")
        ax.set_title("Jitter sweep: mean(best_val_acc)")
        fig.tight_layout()
        fig.savefig(p, dpi=170)
        plt.close(fig)
        out["sweep_jitter_heatmap"] = str(p)

    elif sweep_name == "occlusion":
        ps = sorted({float(r["params"].get("p")) for r in agg_records})
        rs = sorted({float(r["params"].get("radius")) for r in agg_records})
        M = np.full((len(ps), len(rs)), np.nan, dtype=np.float32)
        for r in agg_records:
            p = float(r["params"].get("p"))
            rad = float(r["params"].get("radius"))
            i = ps.index(p)
            j = rs.index(rad)
            M[i, j] = float(r["mean_best_val_acc"])

        p = sweep_plot_dir / "sweep_occlusion_p_vs_radius_heatmap.png"
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(M, cmap="viridis", origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(rs)))
        ax.set_yticks(np.arange(len(ps)))
        ax.set_xticklabels([fmt_float(x) for x in rs])
        ax.set_yticklabels([fmt_float(x) for x in ps])
        ax.set_xlabel("radius")
        ax.set_ylabel("p")
        ax.set_title("Occlusion sweep: mean(best_val_acc)")
        fig.tight_layout()
        fig.savefig(p, dpi=170)
        plt.close(fig)
        out["sweep_occlusion_heatmap"] = str(p)

    elif sweep_name == "dropout":
        rows = sorted(agg_records, key=lambda r: float(r["params"].get("max_ratio")))
        xs = [float(r["params"].get("max_ratio")) for r in rows]
        ys = [float(r["mean_best_val_acc"]) for r in rows]
        errs = [float(r["std_best_val_acc"]) for r in rows]

        p = sweep_plot_dir / "sweep_dropout_ratio_curve.png"
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=4)
        ax.set_xlabel("dropout max_ratio")
        ax.set_ylabel("mean(best_val_acc) (%)")
        ax.set_title("Dropout sweep")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(p, dpi=170)
        plt.close(fig)
        out["sweep_dropout_curve"] = str(p)

    return out


# -----------------------------
# Augmentation gallery
# -----------------------------


def get_gallery_augmentations() -> Dict[str, Callable[[], List]]:
    return {
        "original": lambda: [],
        "baseline(rot+noise)": lambda: [RandomRotationZ(), RandomNoise(0.02)],
        "jitter(s=0.01,c=0.04)": lambda: [RandomJitterClip(0.01, 0.04)],
        "mirror(p=0.3)": lambda: [RandomMirrorXY(0.3)],
        "scale(0.85-1.2)": lambda: [RandomScale(0.85, 1.2)],
        "translate(0.1)": lambda: [RandomTranslate(0.1)],
        "dropout(0.45)": lambda: [RandomPointDropout(0.45)],
        "occlusion(p=0.25,r=0.18)": lambda: [RandomSphericalOcclusion(0.18, 0.25)],
        "combo_full": lambda: [
            RandomRotationZ(),
            RandomNoise(0.02),
            RandomScale(0.85, 1.2),
            RandomTranslate(0.1),
            RandomJitterClip(0.01, 0.04),
            RandomMirrorXY(0.3),
            RandomPointDropout(0.45),
            RandomSphericalOcclusion(0.18, 0.25),
        ],
    }


def apply_ops_np(points: np.ndarray, ops: List) -> np.ndarray:
    out = points.copy()
    for op in ops:
        out = op(out)
    return out


def _scatter_xy(ax, pts: np.ndarray, title: str, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
    ax.scatter(pts[:, 0], pts[:, 1], s=1.5, c="#1f77b4")
    ax.set_title(title, fontsize=8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def generate_augmentation_galleries(
    data_root: str,
    normalize_unit_sphere: bool,
    gallery_indices: List[int],
    gallery_multishot: int,
    out_dir: Path,
) -> List[str]:
    ensure_dir(out_dir)

    base_tf = Compose([UnitSphereNormalize()]) if normalize_unit_sphere else (lambda x: x)
    train_raw = PointCloudData(data_root, folder="train", transform=base_tf)

    aug_defs = get_gallery_augmentations()
    aug_names = list(aug_defs.keys())
    created: List[str] = []

    for idx in gallery_indices:
        if idx < 0 or idx >= len(train_raw):
            print(f"[gallery] skip idx={idx}, out of range [0, {len(train_raw)-1}]")
            continue

        sample = train_raw[idx]
        base_pts = sample["pointcloud"]
        if torch.is_tensor(base_pts):
            base_pts = base_pts.numpy()

        # One-shot per augmentation
        pts_list = [apply_ops_np(base_pts, aug_defs[n]()) for n in aug_names]
        all_pts = np.concatenate(pts_list, axis=0)
        xlim = (float(np.min(all_pts[:, 0])), float(np.max(all_pts[:, 0])))
        ylim = (float(np.min(all_pts[:, 1])), float(np.max(all_pts[:, 1])))

        cols = 3
        rows = int(math.ceil(len(aug_names) / cols))
        fig = plt.figure(figsize=(4.2 * cols, 3.4 * rows))
        for i, (name, pts) in enumerate(zip(aug_names, pts_list), start=1):
            ax = fig.add_subplot(rows, cols, i)
            _scatter_xy(ax, pts, name, xlim, ylim)
        fig.suptitle(f"Augmentation gallery (XY) idx={idx} class={sample['category_name']}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        one_shot_png = out_dir / f"augment_gallery_idx{idx}.png"
        fig.savefig(one_shot_png, dpi=170)
        plt.close(fig)
        created.append(str(one_shot_png))

        # Multi-shot stochasticity view
        n = max(1, int(gallery_multishot))
        fig = plt.figure(figsize=(2.8 * n, 2.5 * len(aug_names)))
        for r, name in enumerate(aug_names, start=1):
            for c in range(1, n + 1):
                ax = fig.add_subplot(len(aug_names), n, (r - 1) * n + c)
                pts = apply_ops_np(base_pts, aug_defs[name]())
                title = name if c == 1 else ""
                _scatter_xy(ax, pts, title, xlim, ylim)
        fig.suptitle(f"Augmentation stochasticity (XY) idx={idx}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        multi_png = out_dir / f"augment_gallery_idx{idx}_multishot.png"
        fig.savefig(multi_png, dpi=170)
        plt.close(fig)
        created.append(str(multi_png))

    return created


# -----------------------------
# Error analysis
# -----------------------------


def load_predictions_from_runs(run_results: List[Dict], out_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for rr in run_results:
        p_csv = out_dir / rr["predictions_csv"]
        if not p_csv.exists():
            continue
        with p_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(
                    {
                        "experiment": r["experiment"],
                        "seed": int(r["seed"]),
                        "index": int(r["index"]),
                        "ply_path": r["ply_path"],
                        "true": int(r["true"]),
                        "pred": int(r["pred"]),
                        "true_name": r["true_name"],
                        "pred_name": r["pred_name"],
                        "confidence": float(r["confidence"]),
                    }
                )
    return rows


def generate_error_analysis(
    run_results: List[Dict],
    out_dir: Path,
    results_dir: Path,
    plots_dir: Path,
    top_k: int,
    hard_examples_per_pair: int,
    data_root: str,
    normalize_unit_sphere: bool,
) -> Dict[str, str]:
    ensure_dir(results_dir)
    ensure_dir(plots_dir)

    pred_rows = load_predictions_from_runs(run_results, out_dir)
    if not pred_rows:
        return {}

    total = len(pred_rows)
    true_counts: Dict[int, int] = defaultdict(int)
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    pair_names: Dict[Tuple[int, int], Tuple[str, str]] = {}

    for r in pred_rows:
        true_counts[r["true"]] += 1
        if r["true"] != r["pred"]:
            k = (r["true"], r["pred"])
            pair_counts[k] += 1
            pair_names[k] = (r["true_name"], r["pred_name"])

    ranked_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    top_pairs = ranked_pairs[: max(1, top_k)]

    conf_csv = results_dir / "top_confusions.csv"
    with conf_csv.open("w", newline="") as f:
        fields = ["true", "pred", "true_name", "pred_name", "count", "rate_true_class", "rate_overall"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (t, p), cnt in top_pairs:
            denom_true = max(1, true_counts[t])
            writer.writerow(
                {
                    "true": t,
                    "pred": p,
                    "true_name": pair_names[(t, p)][0],
                    "pred_name": pair_names[(t, p)][1],
                    "count": cnt,
                    "rate_true_class": float(cnt / denom_true),
                    "rate_overall": float(cnt / max(1, total)),
                }
            )

    # Barplot top confusions
    bar_png = plots_dir / "top_confusions_bar.png"
    labels = [f"{pair_names[(t,p)][0]} -> {pair_names[(t,p)][1]}" for (t, p), _ in top_pairs]
    counts = [cnt for _, cnt in top_pairs]

    fig = plt.figure(figsize=(max(8, len(labels) * 0.75), 5))
    ax = fig.add_subplot(111)
    ax.bar(labels, counts, color="#d62828")
    ax.set_title("Top confusions (off-diagonal)")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(bar_png, dpi=170)
    plt.close(fig)

    # Hard examples gallery for top-3 confusion pairs (both directions)
    base_tf = Compose([UnitSphereNormalize()]) if normalize_unit_sphere else (lambda x: x)
    test_ds = PointCloudData(data_root, folder="test", transform=base_tf)

    hard_created = []
    for (t, p), _ in top_pairs[:3]:
        t_name, p_name = pair_names[(t, p)]
        forward = [r for r in pred_rows if r["true"] == t and r["pred"] == p][:hard_examples_per_pair]
        backward = [r for r in pred_rows if r["true"] == p and r["pred"] == t][:hard_examples_per_pair]

        cols = max(1, hard_examples_per_pair)
        fig = plt.figure(figsize=(3.0 * cols, 5.5))

        # Get global axis limits from selected samples for fair visual comparison
        pts_for_lim = []
        for row in forward + backward:
            idx = row["index"]
            if 0 <= idx < len(test_ds):
                pts = test_ds[idx]["pointcloud"]
                if torch.is_tensor(pts):
                    pts = pts.numpy()
                pts_for_lim.append(pts)

        if pts_for_lim:
            cat = np.concatenate(pts_for_lim, axis=0)
            xlim = (float(np.min(cat[:, 0])), float(np.max(cat[:, 0])))
            ylim = (float(np.min(cat[:, 1])), float(np.max(cat[:, 1])))
        else:
            xlim = (-1.0, 1.0)
            ylim = (-1.0, 1.0)

        for i in range(cols):
            ax = fig.add_subplot(2, cols, i + 1)
            if i < len(forward):
                row = forward[i]
                idx = row["index"]
                pts = test_ds[idx]["pointcloud"]
                if torch.is_tensor(pts):
                    pts = pts.numpy()
                title = f"{t_name}->{p_name}\nconf={row['confidence']:.2f}\n{row['experiment']}/s{row['seed']}"
                _scatter_xy(ax, pts, title, xlim, ylim)
            else:
                ax.axis("off")

            ax2 = fig.add_subplot(2, cols, cols + i + 1)
            if i < len(backward):
                row = backward[i]
                idx = row["index"]
                pts = test_ds[idx]["pointcloud"]
                if torch.is_tensor(pts):
                    pts = pts.numpy()
                title = f"{p_name}->{t_name}\nconf={row['confidence']:.2f}\n{row['experiment']}/s{row['seed']}"
                _scatter_xy(ax2, pts, title, xlim, ylim)
            else:
                ax2.axis("off")

        fig.suptitle(f"Hard examples: {t_name} vs {p_name}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_png = plots_dir / f"hard_examples_{safe_name(t_name)}_vs_{safe_name(p_name)}.png"
        fig.savefig(out_png, dpi=170)
        plt.close(fig)
        hard_created.append(str(out_png))

    return {
        "top_confusions_csv": str(conf_csv),
        "top_confusions_bar": str(bar_png),
        "hard_examples": hard_created,
    }


# -----------------------------
# Notebook synthesis
# -----------------------------


def format_mean_std(mean_val: float, std_val: float, digits: int = 2) -> str:
    return f"{mean_val:.{digits}f} +- {std_val:.{digits}f}"


def to_markdown_table(rows: List[Dict], headers: List[Tuple[str, str]]) -> str:
    # headers: [(key, title)]
    lines = ["| " + " | ".join(title for _, title in headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        vals = [str(r.get(k, "")) for k, _ in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_report_notebooks(
    out_dir: Path,
    args,
    run_results: List[Dict],
    agg_records: List[Dict],
    plot_paths: Dict[str, str],
    sweep_plot_paths: Dict[str, str],
    error_artifacts: Dict[str, str],
    gallery_files: List[str],
) -> Dict[str, str]:
    notebooks_dir = out_dir / "notebooks"
    ensure_dir(notebooks_dir)

    sorted_by_val = sorted(agg_records, key=lambda r: r["mean_best_val_acc"], reverse=True)
    sorted_by_eval = sorted(agg_records, key=lambda r: r["mean_best_eval_acc"], reverse=True)

    table_rows_val = []
    for rank, r in enumerate(sorted_by_val, start=1):
        table_rows_val.append(
            {
                "rank": rank,
                "experiment": r["experiment"],
                "runs": r["num_runs"],
                "best_val_acc": format_mean_std(r["mean_best_val_acc"], r["std_best_val_acc"]),
                "best_eval_acc": format_mean_std(r["mean_best_eval_acc"], r["std_best_eval_acc"]),
                "best_val_loss": format_mean_std(r["mean_best_val_loss"], r["std_best_val_loss"], digits=4),
                "best_epoch": format_mean_std(r["mean_best_epoch"], r["std_best_epoch"], digits=2),
                "final_val_acc": format_mean_std(r["mean_final_val_acc"], r["std_final_val_acc"]),
                "final_train_acc": format_mean_std(r["mean_final_train_acc"], r["std_final_train_acc"]),
                "mean_best_per_class_acc": format_mean_std(
                    r["mean_mean_best_per_class_acc"], r["std_mean_best_per_class_acc"], digits=4
                ),
            }
        )

    table_rows_eval = []
    for rank, r in enumerate(sorted_by_eval, start=1):
        table_rows_eval.append(
            {
                "rank": rank,
                "experiment": r["experiment"],
                "runs": r["num_runs"],
                "best_eval_acc": format_mean_std(r["mean_best_eval_acc"], r["std_best_eval_acc"]),
                "best_val_acc": format_mean_std(r["mean_best_val_acc"], r["std_best_val_acc"]),
            }
        )

    table1 = to_markdown_table(
        table_rows_val,
        [
            ("rank", "Rank"),
            ("experiment", "Experiment"),
            ("runs", "Runs"),
            ("best_val_acc", "best_val_acc (mean+-std)"),
            ("best_eval_acc", "best_eval_acc (mean+-std)"),
            ("best_val_loss", "best_val_loss (mean+-std)"),
            ("best_epoch", "best_epoch (mean+-std)"),
            ("final_val_acc", "final_val_acc (mean+-std)"),
            ("final_train_acc", "final_train_acc (mean+-std)"),
            ("mean_best_per_class_acc", "mean best per-class acc (mean+-std)"),
        ],
    )

    table2 = to_markdown_table(
        table_rows_eval,
        [
            ("rank", "Rank"),
            ("experiment", "Experiment"),
            ("runs", "Runs"),
            ("best_eval_acc", "best_eval_acc (mean+-std)"),
            ("best_val_acc", "best_val_acc (mean+-std)"),
        ],
    )

    gallery_md = "\n".join([f"![gallery]({Path(g).as_posix()})" for g in gallery_files]) if gallery_files else "(disabled)"

    plot_md = "\n".join([f"![{k}]({Path(v).as_posix()})" for k, v in plot_paths.items()]) if plot_paths else "(disabled)"
    sweep_plot_md = "\n".join([f"![{k}]({Path(v).as_posix()})" for k, v in sweep_plot_paths.items()]) if sweep_plot_paths else "(n/a)"

    err_md_lines = []
    if error_artifacts:
        if "top_confusions_csv" in error_artifacts:
            err_md_lines.append(f"- top_confusions_csv: `{error_artifacts['top_confusions_csv']}`")
        if "top_confusions_bar" in error_artifacts:
            err_md_lines.append(f"![top_confusions]({Path(error_artifacts['top_confusions_bar']).as_posix()})")
        for p in error_artifacts.get("hard_examples", []):
            err_md_lines.append(f"![hard_examples]({Path(p).as_posix()})")
    else:
        err_md_lines.append("(disabled)")

    summary_nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# PointNetFull Multi-seed Benchmark Report\n",
                    "\n",
                    "Auto-generated report with robust protocol (multi-seeds, early stopping, optional sweep).\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Setup\n",
                    f"- mode: `{args.mode}`\n",
                    f"- sweep: `{args.sweep if args.mode == 'sweep' else 'n/a'}`\n",
                    f"- seeds: `{args.seeds}`\n",
                    f"- epochs_max: `{args.epochs_max}`\n",
                    f"- batch_size: `{args.batch_size}`\n",
                    f"- early_stop_monitor: `{args.early_stop_monitor}`\n",
                    f"- early_stop_patience: `{args.early_stop_patience}`\n",
                    f"- normalize_unit_sphere: `{args.normalize_unit_sphere}`\n",
                    f"- use_scheduler: `{args.use_scheduler}`\n",
                ],
            },
            {"cell_type": "markdown", "metadata": {}, "source": ["## Ranking by mean(best_val_acc)\n", table1 + "\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Ranking by mean(best_eval_acc)\n", table2 + "\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Comparative Plots\n", plot_md + "\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Sweep-specific Plots\n", sweep_plot_md + "\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Augmentation Galleries\n", gallery_md + "\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Error Analysis\n", "\n".join(err_md_lines) + "\n"]},
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Raw summaries\n",
                    "- `summaries/run_summary.csv`\n",
                    "- `summaries/summary_experiments.csv`\n",
                    "- `summaries/summary_experiments_by_eval.csv`\n",
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

    # Detailed notebook (one section per experiment with run-level entries)
    grouped_runs: Dict[str, List[Dict]] = defaultdict(list)
    for rr in run_results:
        grouped_runs[rr["experiment"]].append(rr)

    detail_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# PointNetFull Detailed Runs\n"],
        }
    ]

    for exp in sorted(grouped_runs.keys()):
        exp_runs = sorted(grouped_runs[exp], key=lambda r: r["seed"])
        agg = next((a for a in agg_records if a["experiment"] == exp), None)
        detail_cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"## {exp}\n",
                    f"- description: {exp_runs[0]['description']}\n",
                    f"- params: `{json.dumps(exp_runs[0].get('params', {}), sort_keys=True)}`\n",
                    (
                        f"- aggregated best_val_acc: {format_mean_std(agg['mean_best_val_acc'], agg['std_best_val_acc'])}\n"
                        if agg is not None
                        else ""
                    ),
                    (
                        f"- aggregated best_eval_acc: {format_mean_std(agg['mean_best_eval_acc'], agg['std_best_eval_acc'])}\n"
                        if agg is not None
                        else ""
                    ),
                ],
            }
        )

        table_rows = []
        for r in exp_runs:
            table_rows.append(
                {
                    "seed": r["seed"],
                    "epochs_ran": r["epochs_ran"],
                    "best_epoch": r["best_epoch"],
                    "best_val_acc": f"{r['best_val_acc']:.2f}",
                    "best_eval_acc": f"{r['best_eval_acc']:.2f}",
                    "best_val_loss": f"{r['best_val_loss']:.4f}",
                    "final_val_acc": f"{r['final_val_acc']:.2f}",
                    "final_train_acc": f"{r['final_train_acc']:.2f}",
                    "curve": f"![curve]({Path((out_dir / r['curves_png'])).as_posix()})",
                    "cm": f"![cm]({Path((out_dir / r['confusion_png'])).as_posix()})",
                }
            )

        md_table = to_markdown_table(
            table_rows,
            [
                ("seed", "Seed"),
                ("epochs_ran", "Epochs Ran"),
                ("best_epoch", "Best Epoch"),
                ("best_val_acc", "Best Val Acc"),
                ("best_eval_acc", "Best Eval Acc"),
                ("best_val_loss", "Best Val Loss"),
                ("final_val_acc", "Final Val Acc"),
                ("final_train_acc", "Final Train Acc"),
                ("curve", "Curve"),
                ("cm", "Confusion"),
            ],
        )
        detail_cells.append({"cell_type": "markdown", "metadata": {}, "source": [md_table + "\n"]})

    details_nb = {
        "cells": detail_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    summary_path = notebooks_dir / "01_benchmark_summary.ipynb"
    details_path = notebooks_dir / "02_detailed_runs.ipynb"
    summary_path.write_text(json.dumps(summary_nb, indent=1))
    details_path.write_text(json.dumps(details_nb, indent=1))

    return {"summary_notebook": str(summary_path), "details_notebook": str(details_path)}


# -----------------------------
# CLI
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Robust PointNetFull augmentation benchmark suite")

    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/nochi/NOCHI/M2_PAR/Apprenyissage_Pointcloud/Pointclouds-classification-with-the-POINTNET-Neural-network/PointNetLab/data/ModelNet10_PLY",
        help="Path to ModelNet*_PLY dataset root.",
    )

    parser.add_argument("--mode", type=str, default="benchmark", choices=["benchmark", "sweep"], help="Benchmark or sweep mode.")
    parser.add_argument("--sweep", type=str, default="jitter", choices=["jitter", "occlusion", "dropout"], help="Sweep family.")

    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Subset of experiment names in benchmark mode. Default: baseline plus_jitter plus_occlusion combo_full",
    )
    parser.add_argument("--max-experiments", type=int, default=0, help="If >0, keep only first N selected experiments.")

    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seeds list for robust aggregation.")

    parser.add_argument("--epochs-max", type=int, default=50, help="Maximum epochs per run.")
    parser.add_argument("--epochs", type=int, default=None, help="Backward-compatible alias for --epochs-max.")

    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (reduced default for longer training).")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")

    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")

    parser.add_argument("--early-stop-monitor", type=str, default="val_acc", choices=["val_acc", "val_loss"])
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)

    parser.add_argument("--use-scheduler", type=str2bool, default=True)
    parser.add_argument("--scheduler-step-size", type=int, default=20)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)

    parser.add_argument("--normalize-unit-sphere", type=str2bool, default=False)
    parser.add_argument("--deterministic", type=str2bool, default=True)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "tpu"])
    parser.add_argument("--probe-devices", action="store_true")

    parser.add_argument("--make-plots", type=str2bool, default=True)
    parser.add_argument("--make-gallery", type=str2bool, default=False)
    parser.add_argument("--gallery-indices", nargs="*", type=int, default=[0])
    parser.add_argument("--gallery-multishot", type=int, default=6)
    parser.add_argument("--make-error-analysis", type=str2bool, default=True)
    parser.add_argument("--top-confusions-k", type=int, default=12)
    parser.add_argument("--hard-examples-per-pair", type=int, default=5)

    parser.add_argument("--plots-dir", type=str, default="results/plots", help="Relative (to run dir) or absolute plots path.")

    parser.add_argument("--output-dir", type=str, default=None, help="Output root directory (new timestamp subdir is created).")
    parser.add_argument("--output-root", type=str, default=None, help="Backward-compatible alias for --output-dir.")

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------


def main():
    args = parse_args()

    if args.probe_devices:
        print_device_probe()
        return

    if args.epochs is not None:
        args.epochs_max = args.epochs

    # Validate dataset before starting long runs.
    ds_info = inspect_dataset_root(args.data_root)

    runtime = select_runtime(args.device)
    if runtime.device_kind == "tpu" and args.num_workers > 0:
        print("TPU detected: forcing num_workers=0 for stability.")
        args.num_workers = 0

    # Reproducibility base seed (per-run seeds are applied later).
    set_seed(args.seeds[0], deterministic=args.deterministic)

    out_base = args.output_dir or args.output_root or str(THIS_DIR / "reports")
    out_base = Path(out_base)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / f"pointnetfull_aug_suite_{ts}"
    ensure_dir(out_dir)

    print(f"Device: {runtime.device_label} (kind={runtime.device_kind}, amp={runtime.amp_enabled})")
    if runtime.device_kind == "cpu":
        print("Warning: CPU-only run. If possible, enable GPU/TPU for faster training.")
    print(f"Dataset check OK: classes={ds_info['num_classes']} train_ply={ds_info['train_ply']} test_ply={ds_info['test_ply']}")
    print(f"Output directory: {out_dir}")

    # Build experiment list
    registry = build_base_experiment_registry()

    if args.mode == "benchmark":
        selected = args.experiments if args.experiments is not None and len(args.experiments) > 0 else DEFAULT_TOP_EXPERIMENTS
        missing = [e for e in selected if e not in registry]
        if missing:
            raise ValueError(f"Unknown experiments: {missing}. Available: {sorted(registry.keys())}")
        exp_configs = [registry[e] for e in selected]
    else:
        exp_configs = build_sweep_configs(args.sweep)

    if args.max_experiments > 0:
        exp_configs = exp_configs[: args.max_experiments]

    print(f"Mode: {args.mode}")
    print(f"Experiments to run: {[e.name for e in exp_configs]}")
    print(f"Seeds: {args.seeds}")
    print(f"epochs_max={args.epochs_max}, batch_size={args.batch_size}, early_stop=({args.early_stop_monitor}, patience={args.early_stop_patience})")

    early_cfg = EarlyStoppingConfig(
        monitor=args.early_stop_monitor,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
    )

    run_results: List[Dict] = []

    for exp_cfg in exp_configs:
        for seed in args.seeds:
            print(f"\n=== Running experiment={exp_cfg.name} seed={seed} ===")
            rr = run_one_experiment_seed(
                exp_cfg=exp_cfg,
                seed=seed,
                args=args,
                runtime=runtime,
                out_dir=out_dir,
                early_cfg=early_cfg,
            )
            run_results.append(rr)

    # Summaries
    summaries_dir = out_dir / "summaries"
    run_summary_csv, run_summary_json = write_run_summary(run_results, summaries_dir)

    agg_records = aggregate_results(run_results)
    agg_paths = write_experiment_summary(agg_records, summaries_dir)

    # Comparative plots
    if Path(args.plots_dir).is_absolute():
        plots_dir = Path(args.plots_dir)
    else:
        plots_dir = out_dir / args.plots_dir
    ensure_dir(plots_dir)

    plot_paths: Dict[str, str] = {}
    if args.make_plots:
        plot_paths = generate_multi_seed_plots(run_results, agg_records, plots_dir)

    # Sweep-specific plots
    sweep_plot_paths: Dict[str, str] = {}
    sweep_plots_dir = plots_dir / "sweeps"
    if args.make_plots and args.mode == "sweep":
        sweep_plot_paths = generate_sweep_plots(agg_records, args.sweep, sweep_plots_dir)

    # Augmentation galleries
    gallery_files: List[str] = []
    if args.make_gallery:
        galleries_dir = plots_dir / "galleries"
        gallery_files = generate_augmentation_galleries(
            data_root=args.data_root,
            normalize_unit_sphere=args.normalize_unit_sphere,
            gallery_indices=args.gallery_indices,
            gallery_multishot=args.gallery_multishot,
            out_dir=galleries_dir,
        )

    # Error analysis
    error_artifacts: Dict[str, str] = {}
    if args.make_error_analysis:
        results_dir = out_dir / "results"
        error_artifacts = generate_error_analysis(
            run_results=run_results,
            out_dir=out_dir,
            results_dir=results_dir,
            plots_dir=plots_dir,
            top_k=args.top_confusions_k,
            hard_examples_per_pair=args.hard_examples_per_pair,
            data_root=args.data_root,
            normalize_unit_sphere=args.normalize_unit_sphere,
        )

    # Notebooks
    nb_paths = generate_report_notebooks(
        out_dir=out_dir,
        args=args,
        run_results=run_results,
        agg_records=agg_records,
        plot_paths=plot_paths,
        sweep_plot_paths=sweep_plot_paths,
        error_artifacts=error_artifacts,
        gallery_files=gallery_files,
    )

    # Metadata
    run_meta = {
        "timestamp": ts,
        "mode": args.mode,
        "sweep": args.sweep if args.mode == "sweep" else None,
        "device": runtime.device_label,
        "device_kind": runtime.device_kind,
        "device_policy": args.device,
        "device_probe": probe_devices(),
        "data_root": args.data_root,
        "dataset_info": ds_info,
        "experiments": [e.name for e in exp_configs],
        "seeds": args.seeds,
        "epochs_max": args.epochs_max,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "use_scheduler": args.use_scheduler,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "early_stop_monitor": args.early_stop_monitor,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "normalize_unit_sphere": args.normalize_unit_sphere,
        "deterministic": args.deterministic,
        "num_runs": len(run_results),
        "run_summary_csv": str(run_summary_csv),
        "run_summary_json": str(run_summary_json),
        **{k: str(v) for k, v in agg_paths.items()},
        **nb_paths,
        "plots": plot_paths,
        "sweep_plots": sweep_plot_paths,
        "gallery_files": gallery_files,
        "error_analysis": error_artifacts,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))

    print("\n=== Completed ===")
    print(f"Run summary CSV: {run_summary_csv}")
    print(f"Experiment summary CSV: {agg_paths['summary_experiments_csv']}")
    print(f"Experiment summary by eval CSV: {agg_paths['summary_experiments_by_eval_csv']}")
    print(f"Summary notebook: {nb_paths['summary_notebook']}")
    print(f"Detailed notebook: {nb_paths['details_notebook']}")

    top = sorted(agg_records, key=lambda x: x["mean_best_val_acc"], reverse=True)
    if top:
        print(f"Best experiment by mean(best_val_acc): {top[0]['experiment']} ({top[0]['mean_best_val_acc']:.2f} +- {top[0]['std_best_val_acc']:.2f})")


if __name__ == "__main__":
    main()
