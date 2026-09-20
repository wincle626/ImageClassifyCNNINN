"""Small shared helpers: config loading, seeding, device, checkpoint I/O."""

from __future__ import annotations

import contextlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class Config:
    # data
    data_root: str = "./data"
    subset_fraction: float = 1.0
    batch_size: int = 128
    num_workers: int = 0
    download: bool = True
    # model
    backbone: str = "resnet18"   # "resnet18" (main architecture) or "simple"
    pretrained: bool = True       # load ImageNet weights for resnet18
    small_input: bool = False     # CIFAR-style stem to preserve 48x48 resolution
    feature_dim: int = 256        # only used by the "simple" backbone
    temperature: float = 8.0
    # training
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    concept_loss_weight: float = 1.0
    class_loss_weight: float = 1.0
    seed: int = 0
    device: str = "auto"          # "auto" | "cuda" | "cuda:0" | "cpu"
    amp: bool = True              # mixed-precision training (only active on CUDA)
    # io
    out_dir: str = "./outputs"

    @staticmethod
    def load(path: str | None) -> "Config":
        cfg = Config()
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """Resolve the compute device.

    ``prefer`` is "auto" (CUDA if available, else CPU), or an explicit device
    string such as "cuda", "cuda:0" or "cpu". If CUDA is requested but not
    available, we warn and fall back to CPU rather than crashing.
    """
    if prefer in (None, "auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] requested device '{prefer}' but CUDA is not available; "
              "falling back to CPU. Install a CUDA build of PyTorch for GPU "
              "(see README).")
        return torch.device("cpu")
    return torch.device(prefer)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    return "cpu"


def configure_backends(device: torch.device) -> None:
    """Enable cuDNN autotuning for fixed-size inputs (a free speed-up on GPU)."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


class AmpHelper:
    """Tiny mixed-precision helper. On CUDA (with ``enabled``) it uses autocast
    + GradScaler; on CPU it is a no-op so the same training code runs anywhere."""

    def __init__(self, device: torch.device, enabled: bool = True):
        self.enabled = bool(enabled) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.enabled else None

    def autocast(self):
        if self.enabled:
            return torch.amp.autocast("cuda")
        return contextlib.nullcontext()

    def backward_step(self, loss: torch.Tensor, optimizer) -> None:
        if self.enabled:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
