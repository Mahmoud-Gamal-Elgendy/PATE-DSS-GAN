"""
Dataset wrappers for PATE-DSS-GAN.

Supports CelebA-HQ and AFHQ. Each dataset returns (image, label)
pairs where label is the class index used for stratified partitioning and
class-conditional generation via DSS-GAN's Directional Latent Routing (DLR).

HuggingFace sources
-------------------
- CelebA-HQ : korexyz/celeba-hq-256x256
- AFHQ      : huggan/AFHQ
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ---------------------------------------------------------------------------
# Standard transform factory
# ---------------------------------------------------------------------------

def _image_transform(image_size: int, augment: bool = False) -> transforms.Compose:
    tfms: list = []
    if augment:
        tfms += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    tfms += [
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
    ]
    return transforms.Compose(tfms)


# ---------------------------------------------------------------------------
# CelebA-HQ via HuggingFace  (korexyz/celeba-hq-256x256)
# ---------------------------------------------------------------------------

class CelebAHQDataset(Dataset):
    """
    CelebA-HQ loaded from HuggingFace Hub.

    Source  : korexyz/celeba-hq-256x256
    Label   : binary gender attribute (0 = female, 1 = male)
              derived from the 'Male' attribute column (index 20 in CelebA).

    Usage
    -----
    ds = CelebAHQDataset(split='train', image_size=128)
    """

    HF_REPO = "korexyz/celeba-hq-256x256"

    # CelebA attribute index for 'Male' (1-indexed in the original annotations)
    _MALE_ATTR_IDX = 20

    def __init__(
        self,
        split: str = "train",
        image_size: int = 256,
        augment: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Install the HuggingFace datasets library: pip install datasets"
            ) from exc

        print(f"[CelebAHQ] Loading {self.HF_REPO} (split={split}) ...")
        self._hf = load_dataset(self.HF_REPO, split=split, cache_dir=cache_dir)
        self._tfm = _image_transform(image_size, augment)

        # Build label list: 1 = male, 0 = female
        # The HF dataset exposes the image under the 'image' key and
        # attributes as separate columns. If no attribute column exists,
        # fall back to label 0 for all samples.
        self.targets: List[int] = self._build_targets()
        self.classes = ["female", "male"]
        self.class_to_idx: Dict[str, int] = {"female": 0, "male": 1}

    def _build_targets(self) -> List[int]:
        col_names = self._hf.column_names
        print(f"[CelebAHQ] Available columns: {col_names}")

        # Case 1: direct integer 'label' column (e.g. korexyz/celeba-hq-256x256)
        # label = 0 → female, label = 1 → male
        for col in ("label", "labels", "gender", "sex"):
            if col in col_names:
                raw = self._hf[col]
                if isinstance(raw[0], int):
                    counts = {0: raw.count(0), 1: raw.count(1)}
                    print(f"[CelebAHQ] Labels from column '{col}': {counts}")
                    return list(raw)

        # Case 2: direct 'Male' / 'male' integer column
        for col in ("Male", "male"):
            if col in col_names:
                raw = self._hf[col]
                print(f"[CelebAHQ] Labels from column '{col}'")
                return [int(v) for v in raw]

        # Case 3: nested attributes dict with 'Male' key
        for col in ("attributes", "attr"):
            if col in col_names:
                first = self._hf[0][col]
                if isinstance(first, dict) and "Male" in first:
                    print(f"[CelebAHQ] Labels from '{col}[\"Male\"]'")
                    return [int(row[col]["Male"]) for row in self._hf]

        # Fallback
        print(f"[CelebAHQ] No gender label found in columns {col_names}; assigning label 0 to all samples.")
        return [0] * len(self._hf)

    def __len__(self) -> int:
        return len(self._hf)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self._hf[idx]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        return self._tfm(img), self.targets[idx]


# ---------------------------------------------------------------------------
# AFHQ via HuggingFace  (huggan/AFHQ)
# ---------------------------------------------------------------------------

class AFHQDataset(Dataset):
    """
    AFHQ (Animal Faces HQ) loaded from HuggingFace Hub.

    Source  : huggan/AFHQ
    Labels  : 0 = cat, 1 = dog, 2 = wild
    Split   : 'train' or 'test'

    Usage
    -----
    ds = AFHQDataset(split='train', image_size=256)
    """

    HF_REPO = "huggan/AFHQ"

    _LABEL_MAP: Dict[str, int] = {"cat": 0, "dog": 1, "wild": 2}

    def __init__(
        self,
        split: str = "train",
        image_size: int = 256,
        augment: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Install the HuggingFace datasets library: pip install datasets"
            ) from exc

        print(f"[AFHQ] Loading {self.HF_REPO} (split={split}) ...")
        self._hf = load_dataset(self.HF_REPO, split=split, cache_dir=cache_dir)
        self._tfm = _image_transform(image_size, augment)

        self.targets: List[int] = self._build_targets()
        self.classes = ["cat", "dog", "wild"]
        self.class_to_idx = self._LABEL_MAP.copy()

    def _build_targets(self) -> List[int]:
        col_names = self._hf.column_names
        # huggan/AFHQ exposes a 'label' column (ClassLabel: cat/dog/wild)
        for col in ("label", "labels", "class", "category"):
            if col in col_names:
                raw = self._hf[col]
                # ClassLabel integers map directly; string labels need mapping
                if isinstance(raw[0], int):
                    return list(raw)
                return [self._LABEL_MAP.get(str(v).lower(), 0) for v in raw]
        print("[AFHQ] No label column found; assigning label 0 to all samples.")
        return [0] * len(self._hf)

    def __len__(self) -> int:
        return len(self._hf)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self._hf[idx]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        return self._tfm(img), self.targets[idx]


# ---------------------------------------------------------------------------
# ImageDatasetWrapper — unified interface + subsetting
# ---------------------------------------------------------------------------

class ImageDatasetWrapper(Dataset):
    """
    Thin wrapper exposing a unified (image, label) interface with
    index-based subsetting support (used by the stratified partitioner).
    """

    def __init__(self, base_dataset: Dataset, indices: Optional[List[int]] = None) -> None:
        self._ds = base_dataset
        self._indices: List[int] = indices if indices is not None else list(range(len(base_dataset)))

    @property
    def targets(self) -> List[int]:
        all_targets = getattr(self._ds, "targets", [])
        return [all_targets[i] for i in self._indices]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._ds[self._indices[idx]]

    def subset(self, sub_indices: List[int]) -> "ImageDatasetWrapper":
        mapped = [self._indices[i] for i in sub_indices]
        return ImageDatasetWrapper(self._ds, mapped)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_dataset(
    name: str,
    image_size: int = 128,
    train: bool = True,
    augment: bool = False,
    hf_cache_dir: Optional[str] = None,
) -> ImageDatasetWrapper:
    """
    Return an ImageDatasetWrapper for the requested dataset.

    Parameters
    ----------
    name : str
        One of 'celeba_hq', 'afhq'.
    image_size : int
        Target spatial resolution after resizing.
    train : bool
        Training split (maps to HuggingFace split='train' or 'test').
    augment : bool
        Enable random flip + colour jitter during loading.
    hf_cache_dir : str, optional
        Custom cache directory for HuggingFace datasets.

    Sources
    -------
    - celeba_hq : korexyz/celeba-hq-256x256  (HuggingFace)
    - afhq      : huggan/AFHQ                 (HuggingFace)
    """
    name = name.lower()
    split = "train" if train else "test"

    if name == "celeba_hq":
        ds = CelebAHQDataset(
            split=split,
            image_size=image_size,
            augment=augment,
            cache_dir=hf_cache_dir,
        )
    elif name == "afhq":
        ds = AFHQDataset(
            split=split,
            image_size=image_size,
            augment=augment,
            cache_dir=hf_cache_dir,
        )
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: celeba_hq, afhq.")

    return ImageDatasetWrapper(ds)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
