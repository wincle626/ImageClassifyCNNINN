"""GTSRB data loading with concept labels attached.

Wraps ``torchvision.datasets.GTSRB`` (downloaded automatically) and augments
every sample with its ground-truth concept vector derived from
:mod:`src.concepts`.  Each item is ``(image, class_id, concept_vector)``.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB

from .concepts import concept_matrix

# GTSRB signs are small and roughly square; 48x48 keeps digits/symbols legible
# while staying cheap enough to train on CPU.
IMG_SIZE = 48

# Per-channel statistics are close to 0.5 for this dataset; using 0.5/0.5 keeps
# things simple and reproducible without a separate statistics pass.
_MEAN = (0.34, 0.31, 0.32)
_STD = (0.27, 0.26, 0.27)


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08),
                                    scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


class GTSRBWithConcepts(Dataset):
    """GTSRB wrapper that returns ``(image, class_id, concept_vector)``."""

    def __init__(self, root: str, split: str, download: bool = True):
        assert split in ("train", "test")
        self.base = GTSRB(root=root, split=split, download=download,
                          transform=build_transforms(train=(split == "train")))
        self.concept_m = torch.from_numpy(concept_matrix())  # [C, K]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        concepts = self.concept_m[label]
        return image, label, concepts


def _maybe_subset(ds: Dataset, fraction: float, seed: int) -> Dataset:
    if fraction >= 1.0:
        return ds
    n = len(ds)
    k = max(1, int(n * fraction))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:k].tolist()
    return Subset(ds, idx)


def make_dataloaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 0,
    subset_fraction: float = 1.0,
    seed: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, test_loader)``.

    ``subset_fraction`` (0-1) samples a random fraction of each split -- handy
    for a fast smoke run on CPU.
    """
    train_ds = GTSRBWithConcepts(root, "train", download=download)
    test_ds = GTSRBWithConcepts(root, "test", download=download)
    train_ds = _maybe_subset(train_ds, subset_fraction, seed)
    test_ds = _maybe_subset(test_ds, subset_fraction, seed + 1)

    # pin host memory when a CUDA GPU is present -> faster async H2D transfers
    pin = torch.cuda.is_available()
    persist = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
        persistent_workers=persist,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
        persistent_workers=persist,
    )
    return train_loader, test_loader
