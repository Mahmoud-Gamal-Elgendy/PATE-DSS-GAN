"""
PATE Ensemble Manager for PATE-DSS-GAN.

Coordinates:
  - Teacher shard dataloaders
  - Periodic teacher retraining (every `retrain_interval` outer iterations)
  - Vote collection → aggregation pipeline

Design note: Teachers are retrained on the CURRENT generator output every
`retrain_interval` outer steps. This avoids anchoring teachers to the initial
generator (G_init bug from PATE-TabTransGAN), keeping their real/fake
judgement calibrated to the evolving generator.

Privacy safety: teachers see only their own shard (real images) + current
synthetic images (public). Retraining does not access other shards.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..models.teacher import TeacherEnsemble, build_teacher_ensemble
from ..data.dataset import make_dataloader, ImageDatasetWrapper
from .voting import PATEVoteAggregator


class PATEEnsembleManager:
    """
    High-level orchestrator for the PATE teacher ensemble + vote aggregation.

    Parameters
    ----------
    num_teachers : int
        k — number of teacher discriminators.
    shards : list of ImageDatasetWrapper
        Length-k list of private data shards.
    aggregator : PATEVoteAggregator
        Configured vote aggregator with calibrated sigma.
    image_size : int
        Spatial resolution of training images.
    base_channels : int
        CNN base channels for teacher architecture.
    batch_size : int
        Batch size for teacher training dataloaders.
    retrain_interval : int
        Retrain all teachers every this many outer GAN iterations.
    teacher_epochs : int
        Epochs per teacher retraining.
    teacher_lr : float
        Learning rate for teacher Adam optimizer.
    device : torch.device
    """

    def __init__(
        self,
        num_teachers: int,
        shards: List[ImageDatasetWrapper],
        aggregator: PATEVoteAggregator,
        image_size: int = 128,
        base_channels: int = 32,
        batch_size: int = 64,
        retrain_interval: int = 50,
        teacher_epochs: int = 3,
        teacher_lr: float = 2e-4,
        num_workers: int = 2,
        pin_memory: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        assert len(shards) == num_teachers, "Must provide one shard per teacher."

        self.num_teachers = num_teachers
        self.aggregator = aggregator
        self.retrain_interval = retrain_interval
        self.teacher_epochs = teacher_epochs
        self.teacher_lr = teacher_lr
        self.device = device or torch.device("cpu")

        # Build teacher ensemble
        self.ensemble = build_teacher_ensemble(
            num_teachers=num_teachers,
            image_size=image_size,
            base_channels=base_channels,
        )

        # Build per-shard dataloaders (persistent)
        self.shard_loaders: List[DataLoader] = [
            make_dataloader(
                shard,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            for shard in shards
        ]

        self._outer_step: int = 0

    def maybe_retrain_check(self, step: int) -> bool:
        """
        Return True if teachers should be retrained at this step.

        Separates the check from the actual retraining so the caller can
        refresh the fake image buffer ONLY when retraining is needed,
        avoiding the cost of generating buffer_size fake images every step.

        Parameters
        ----------
        step : int
            Current outer step index.
        """
        return step % self.retrain_interval == 0

    def do_retrain(self, fake_dataloader: DataLoader, step: int = 0) -> None:
        """
        Retrain all k teachers. Call only after maybe_retrain_check() returns True.

        Parameters
        ----------
        fake_dataloader : DataLoader
            Dataloader of current generator fake images (public).
        step : int
            Current outer step (for display only).
        """
        losses = self.ensemble.train_all(
            shard_dataloaders=self.shard_loaders,
            fake_dataloader=fake_dataloader,
            num_epochs=self.teacher_epochs,
            lr=self.teacher_lr,
            device=self.device,
            verbose=False,   # silence per-teacher lines; summary shown below
        )
        avg = sum(losses) / len(losses)
        lo, hi = min(losses), max(losses)
        print(f"[Step {step:05d}] [Teachers] retrained k={self.num_teachers} | "
              f"loss avg={avg:.4f}  min={lo:.4f}  max={hi:.4f}")

    def query(self, fake_imgs: Tensor) -> Tensor:
        """
        Query the ensemble: collect votes → aggregate → return noisy labels.

        This is the privacy-sensitive operation. Each call charges ε budget.

        Parameters
        ----------
        fake_imgs : Tensor  (B, C, H, W)
            Current generator fake images.

        Returns
        -------
        Tensor  (B,) long  — PATE noisy labels (1=real, 0=fake).
        """
        votes = self.ensemble.collect_votes(fake_imgs, self.device)  # (k, B)
        noisy_labels = self.aggregator.aggregate(votes)
        return noisy_labels

    def query_with_confidence(self, fake_imgs: Tensor) -> tuple[Tensor, Tensor]:
        """Query and also return pre-noise teacher confidence (monitoring only)."""
        votes = self.ensemble.collect_votes(fake_imgs, self.device)
        return self.aggregator.aggregate_with_confidence(votes)

    def step(self) -> None:
        """Increment outer iteration counter (call once per outer GAN step)."""
        self._outer_step += 1

    @property
    def outer_step(self) -> int:
        return self._outer_step
