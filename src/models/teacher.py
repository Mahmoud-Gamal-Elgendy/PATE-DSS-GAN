"""
Simple CNN Teacher Discriminators for PATE-DSS-GAN.

Each teacher is a lightweight CNN that votes real (1) vs fake (0) on images.
Teachers are intentionally simple to keep the privacy analysis tractable and
focus model capacity on the Mamba student discriminator and generator.

Design constraints:
  - Teachers vote real/fake only (NOT class labels) — preserves GAN training signal.
  - Each teacher trained exclusively on its private shard.
  - Teachers retrained every `retrain_interval` outer iterations for fresh votes.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv → LeakyReLU → optional spectral norm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_spectral_norm: bool = True,
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        layers: list = [conv]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Teacher discriminator
# ---------------------------------------------------------------------------

class TeacherCNNDiscriminator(nn.Module):
    """
    Lightweight CNN teacher that outputs a single real/fake logit.

    Architecture: 5 strided conv layers → global average pool → linear.
    Input images are normalised to [-1, 1].

    Parameters
    ----------
    image_size : int
        Spatial resolution of input images (assumed square).
    in_channels : int
        Number of input channels (3 for RGB).
    base_channels : int
        Channel multiplier for the first conv layer.
    """

    def __init__(
        self,
        image_size: int = 128,
        in_channels: int = 3,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        C = base_channels

        self.features = nn.Sequential(
            # 128 → 64
            ConvBlock(in_channels, C, stride=2),
            # 64 → 32
            ConvBlock(C, C * 2, stride=2),
            # 32 → 16
            ConvBlock(C * 2, C * 4, stride=2),
            # 16 → 8
            ConvBlock(C * 4, C * 8, stride=2),
            # 8 → 4
            ConvBlock(C * 8, C * 8, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(C * 8, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Return raw logit (un-sigmoid'd) for real/fake classification."""
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.head(h)

    def vote(self, x: Tensor) -> Tensor:
        """
        Return binary vote: 1 = real, 0 = fake.
        Shape: (B,) integer tensor.
        """
        with torch.no_grad():
            logit = self.forward(x)
        return (logit.squeeze(1) > 0).long()


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------

class TeacherEnsemble(nn.Module):
    """
    Collection of k teacher discriminators, one per private data shard.

    Handles per-teacher training and provides collective voting interface
    for the PATE aggregation step.
    """

    def __init__(self, teachers: List[TeacherCNNDiscriminator]) -> None:
        super().__init__()
        self.teachers = nn.ModuleList(teachers)

    @property
    def num_teachers(self) -> int:
        return len(self.teachers)

    def train_teacher(
        self,
        teacher_idx: int,
        dataloader: DataLoader,
        fake_dataloader: DataLoader,
        num_epochs: int = 3,
        lr: float = 2e-4,
        device: torch.device = torch.device("cpu"),
    ) -> float:
        """
        Train a single teacher discriminator on its private shard.

        Teachers are trained with real images from their shard and fake images
        from the current generator. This is privacy-safe because:
          - Real images come only from the teacher's own shard.
          - Fake images are synthetic (public).

        Returns
        -------
        float
            Final training loss.
        """
        teacher = self.teachers[teacher_idx].to(device)
        teacher.train()
        opt = torch.optim.Adam(teacher.parameters(), lr=lr, betas=(0.0, 0.9))
        loss_fn = nn.BCEWithLogitsLoss()

        total_loss = 0.0
        steps = 0
        fake_iter = iter(fake_dataloader)

        for _ in range(num_epochs):
            for real_imgs, _ in dataloader:
                real_imgs = real_imgs.to(device)
                try:
                    fake_imgs = next(fake_iter)[0].to(device)
                except StopIteration:
                    fake_iter = iter(fake_dataloader)
                    fake_imgs = next(fake_iter)[0].to(device)

                real_labels = torch.ones(real_imgs.size(0), 1, device=device)
                fake_labels = torch.zeros(fake_imgs.size(0), 1, device=device)

                real_loss = loss_fn(teacher(real_imgs), real_labels)
                fake_loss = loss_fn(teacher(fake_imgs), fake_labels)
                loss = (real_loss + fake_loss) / 2.0

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()
                steps += 1

        teacher.eval()
        return total_loss / max(steps, 1)

    def train_all(
        self,
        shard_dataloaders: List[DataLoader],
        fake_dataloader: DataLoader,
        num_epochs: int = 3,
        lr: float = 2e-4,
        device: torch.device = torch.device("cpu"),
        verbose: bool = False,
    ) -> List[float]:
        """Train all k teachers. Returns per-teacher loss list."""
        losses = []
        for i in range(self.num_teachers):
            loss = self.train_teacher(
                i, shard_dataloaders[i], fake_dataloader, num_epochs, lr, device
            )
            losses.append(loss)
            if verbose:
                print(f"  Teacher {i:02d} loss: {loss:.4f}")
        return losses

    def collect_votes(self, fake_imgs: Tensor, device: torch.device) -> Tensor:
        """
        Collect binary votes from all teachers on a batch of fake images.

        Parameters
        ----------
        fake_imgs : Tensor
            Shape (B, C, H, W) — current generator output.
        device : torch.device

        Returns
        -------
        Tensor
            Shape (k, B) — each row is one teacher's votes (0 or 1).
        """
        fake_imgs = fake_imgs.to(device)
        votes = []
        for teacher in self.teachers:
            teacher.to(device).eval()
            votes.append(teacher.vote(fake_imgs))  # (B,)
        return torch.stack(votes, dim=0)  # (k, B)


def build_teacher_ensemble(
    num_teachers: int,
    image_size: int = 128,
    in_channels: int = 3,
    base_channels: int = 32,
) -> TeacherEnsemble:
    """Factory function to construct k identical teacher discriminators."""
    teachers = [
        TeacherCNNDiscriminator(image_size, in_channels, base_channels)
        for _ in range(num_teachers)
    ]
    return TeacherEnsemble(teachers)
