"""
PATE-DSS-GAN Main Training Loop.

Training algorithm:
  Outer loop (privacy-accounting loop):
    1. [Every retrain_interval steps] Retrain teachers on current generator output.
    2. Sample batch z, c → Generator → fake images G(z, c).
    3. Query PATE ensemble: noisy_labels = aggregate(votes(fake_imgs)).
       → Records ε budget via accountant.
    4. Post-processing: perform n_s student update steps using PATE labels
       (post-processing is privacy-free — labels already released).
    5. Update Generator with student discriminator loss.
    6. Log privacy budget ε and training metrics.
    7. Stop when ε ≥ target_epsilon (privacy budget exhausted).

Privacy-correct design choices:
  - Teachers retrained on current generator (not anchored to G_init).
  - No confident threshold (all images queried, all labels recorded).
  - σ from calibrate_sigma() ensures ε budget is respected.
  - n_s student updates per query amortises the privacy cost.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from ..accountant import GNMaxRDPAccountant, calibrate_sigma
from ..data.dataset import make_dataloader, ImageDatasetWrapper
from ..data.partitioner import stratified_partition
from ..models.teacher import build_teacher_ensemble
from ..models.student import MambaStudentDiscriminator
from ..models.generator import DSSGANGenerator
from ..pate.voting import PATEVoteAggregator
from ..pate.ensemble import PATEEnsembleManager


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Dataset
    dataset_name: str = "celeba_hq"
    data_root: str = "./data"
    image_size: int = 128
    num_classes: int = 2

    # Privacy
    target_epsilon: float = 10.0
    delta: float = 1e-5
    num_teachers: int = 10
    num_queries: int = 5000       # estimated total PATE queries
    n_student_steps: int = 5      # post-processing student updates per query

    # Architecture
    latent_dim: int = 256
    base_channels_gen: int = 64
    base_channels_disc: int = 64
    base_channels_teacher: int = 32
    mamba_d_model: int = 256
    mamba_layers: int = 4
    scan_directions: List[str] = field(default_factory=lambda: ["row", "col", "diag"])

    # Training
    batch_size: int = 32
    gen_lr: float = 2e-4
    student_lr: float = 2e-4
    teacher_lr: float = 2e-4
    teacher_epochs: int = 3
    retrain_interval: int = 50
    max_outer_steps: int = 10000

    # Logging
    log_interval: int = 50
    save_interval: int = 500
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"

    # Misc
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PATEDSSGANTrainer:
    """
    PATE-DSS-GAN training orchestrator.

    Usage
    -----
    trainer = PATEDSSGANTrainer(config, dataset)
    trainer.train()
    """

    def __init__(self, config: TrainConfig, dataset: ImageDatasetWrapper) -> None:
        self.config = config
        self.dataset = dataset
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        torch.manual_seed(config.seed)

        # Set up output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._setup_privacy()
        self._setup_models()
        self._setup_data()
        self._setup_optimisers()

        self.metrics: Dict[str, List[float]] = {
            "epsilon": [],
            "g_loss": [],
            "d_loss": [],
            "teacher_confidence": [],
        }

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_privacy(self) -> None:
        cfg = self.config
        print(f"[Privacy] Calibrating σ for ε={cfg.target_epsilon}, "
              f"N={cfg.num_queries}, k={cfg.num_teachers}, δ={cfg.delta} ...")
        self.sigma = calibrate_sigma(
            target_epsilon=cfg.target_epsilon,
            num_queries=cfg.num_queries,
            num_teachers=cfg.num_teachers,
            delta=cfg.delta,
        )
        print(f"[Privacy] Calibrated σ = {self.sigma:.4f}")

        self.accountant = GNMaxRDPAccountant(
            num_teachers=cfg.num_teachers,
            sigma=self.sigma,
            delta=cfg.delta,
        )
        self.aggregator = PATEVoteAggregator(
            sigma=self.sigma,
            accountant=self.accountant,
            device=self.device,
        )

    def _setup_models(self) -> None:
        cfg = self.config

        self.generator = DSSGANGenerator(
            latent_dim=cfg.latent_dim,
            num_classes=cfg.num_classes,
            image_size=cfg.image_size,
            base_channels=cfg.base_channels_gen,
            mamba_d_model=cfg.mamba_d_model,
        ).to(self.device)

        self.student = MambaStudentDiscriminator(
            image_size=cfg.image_size,
            base_channels=cfg.base_channels_disc,
            mamba_d_model=cfg.mamba_d_model,
            mamba_layers=cfg.mamba_layers,
            scan_directions=cfg.scan_directions,
        ).to(self.device)

        print(f"[Models] Generator params: {_count_params(self.generator):,}")
        print(f"[Models] Student params:   {_count_params(self.student):,}")

    def _setup_data(self) -> None:
        cfg = self.config

        shards = stratified_partition(
            self.dataset,
            num_shards=cfg.num_teachers,
            seed=cfg.seed,
        )

        self.ensemble_manager = PATEEnsembleManager(
            num_teachers=cfg.num_teachers,
            shards=shards,
            aggregator=self.aggregator,
            image_size=cfg.image_size,
            base_channels=cfg.base_channels_teacher,
            batch_size=cfg.batch_size,
            retrain_interval=cfg.retrain_interval,
            teacher_epochs=cfg.teacher_epochs,
            teacher_lr=cfg.teacher_lr,
            device=self.device,
        )

    def _setup_optimisers(self) -> None:
        cfg = self.config
        self.opt_gen = torch.optim.Adam(
            self.generator.parameters(), lr=cfg.gen_lr, betas=(0.0, 0.9)
        )
        self.opt_student = torch.optim.Adam(
            self.student.parameters(), lr=cfg.student_lr, betas=(0.0, 0.9)
        )

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def _sample_fake(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """Sample random latent + class → generator output."""
        z = self.generator.sample_latent(batch_size, self.device)
        c = torch.randint(0, self.config.num_classes, (batch_size,), device=self.device)
        with torch.no_grad():
            fake_imgs = self.generator(z, c)
        return fake_imgs, c

    def _student_update(
        self,
        fake_imgs: Tensor,
        noisy_labels: Tensor,
        n_steps: int,
    ) -> float:
        """
        Post-processing student discriminator update (privacy-free).

        The PATE labels are already released, so n_s gradient steps
        on the student using these labels incur NO additional ε cost.
        """
        self.student.train()
        loss_fn = nn.BCEWithLogitsLoss()
        total_loss = 0.0

        for _ in range(n_steps):
            self.opt_student.zero_grad()
            logits = self.student(fake_imgs.detach()).squeeze(1)
            labels = noisy_labels.float()
            loss = loss_fn(logits, labels)
            loss.backward()
            self.opt_student.step()
            total_loss += loss.item()

        self.student.eval()
        return total_loss / n_steps

    def _generator_update(self, batch_size: int) -> float:
        """Update generator to fool the student discriminator."""
        self.generator.train()
        self.opt_gen.zero_grad()

        z = self.generator.sample_latent(batch_size, self.device)
        c = torch.randint(0, self.config.num_classes, (batch_size,), device=self.device)
        fake_imgs = self.generator(z, c)

        # Generator tries to maximise D(G(z)) → minimise -log(σ(D(G(z))))
        logits = self.student(fake_imgs).squeeze(1)
        # Non-saturating loss: G minimises E[-log(sigmoid(D(G(z))))]
        g_loss = F.softplus(-logits).mean()

        g_loss.backward()
        self.opt_gen.step()
        self.generator.eval()
        return g_loss.item()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.config
        print(f"\n{'='*60}")
        print("PATE-DSS-GAN Training")
        print(f"  Target ε: {cfg.target_epsilon} | σ: {self.sigma:.4f}")
        print(f"  Teachers: {cfg.num_teachers} | Image size: {cfg.image_size}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        # Build a fake image dataloader for teacher retraining
        fake_buffer = _FakeImageBuffer(
            generator=self.generator,
            num_classes=cfg.num_classes,
            latent_dim=cfg.latent_dim,
            buffer_size=cfg.batch_size * 20,
            device=self.device,
        )

        t0 = time.time()

        for step in range(cfg.max_outer_steps):
            # Periodic teacher retraining.
            # refresh() is called only when retraining actually happens to avoid
            # generating batch_size*20 fake images on every outer step.
            did_retrain = self.ensemble_manager.maybe_retrain_check(step)
            if did_retrain:
                fake_buffer.refresh(self.generator)
                self.ensemble_manager.do_retrain(fake_buffer.as_dataloader(cfg.batch_size), step=step)

            # Sample fake images
            fake_imgs, fake_classes = self._sample_fake(cfg.batch_size)

            # PATE query (charges ε budget)
            noisy_labels, confidence = self.ensemble_manager.query_with_confidence(fake_imgs)

            # Check privacy budget AFTER the query so no step is under-counted
            current_eps = self.accountant.get_epsilon()
            if current_eps >= cfg.target_epsilon:
                print(
                    f"\n[Step {step:05d}] Privacy budget exhausted: "
                    f"ε={current_eps:.3f} ≥ {cfg.target_epsilon:.0f}  "
                    f"(queries={self.accountant.steps})"
                )
                break

            # Student post-processing updates (privacy-free)
            d_loss = self._student_update(fake_imgs, noisy_labels, cfg.n_student_steps)

            # Generator update
            g_loss = self._generator_update(cfg.batch_size)

            # Logging
            if step % cfg.log_interval == 0:
                eps = self.accountant.get_epsilon()
                budget_pct = 100.0 * eps / cfg.target_epsilon
                elapsed = time.time() - t0
                conf_mean = confidence.float().mean().item()
                queries_so_far = self.accountant.steps
                print(
                    f"[Step {step:05d}] "
                    f"ε={eps:.3f}/{cfg.target_epsilon:.0f} ({budget_pct:.1f}%) | "
                    f"queries={queries_so_far} | "
                    f"G={g_loss:.4f} | D={d_loss:.4f} | "
                    f"TeacherConf={conf_mean:.3f} | {elapsed:.0f}s"
                )
                self.metrics["epsilon"].append(eps)
                self.metrics["g_loss"].append(g_loss)
                self.metrics["d_loss"].append(d_loss)
                self.metrics["teacher_confidence"].append(conf_mean)

            # Save samples and checkpoint
            if step % cfg.save_interval == 0 and step > 0:
                self._save_samples(step)
                self._save_checkpoint(step)

        print("\n[Trainer] Training complete.")
        self._save_checkpoint("final")
        self._save_samples("final")

    def _save_samples(self, step) -> None:
        cfg = self.config
        n = min(16, cfg.batch_size)
        z = self.generator.sample_latent(n, self.device)
        c = torch.arange(n, device=self.device) % cfg.num_classes
        with torch.no_grad():
            imgs = self.generator(z, c)
        path = os.path.join(cfg.output_dir, f"samples_step{step}.png")
        save_image(imgs * 0.5 + 0.5, path, nrow=4)
        print(f"[Trainer] Saved samples → {path}")

    def _save_checkpoint(self, step) -> None:
        cfg = self.config
        path = os.path.join(cfg.checkpoint_dir, f"ckpt_step{step}.pt")
        torch.save({
            "step": step,
            "generator": self.generator.state_dict(),
            "student": self.student.state_dict(),
            "opt_gen": self.opt_gen.state_dict(),
            "opt_student": self.opt_student.state_dict(),
            "accountant_steps": self.accountant.steps,
            "sigma": self.sigma,
            "epsilon": self.accountant.get_epsilon(),
        }, path)
        print(f"[Trainer] Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(ckpt["generator"])
        self.student.load_state_dict(ckpt["student"])
        self.opt_gen.load_state_dict(ckpt["opt_gen"])
        self.opt_student.load_state_dict(ckpt["opt_student"])
        self.accountant._steps = ckpt["accountant_steps"]
        print(f"[Trainer] Loaded checkpoint from {path} (ε={ckpt['epsilon']:.4f})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _FakeImageBuffer:
    """
    Maintains a small buffer of generator fake images for teacher retraining.
    Refreshed at each outer step from the current generator.
    """

    def __init__(
        self,
        generator: DSSGANGenerator,
        num_classes: int,
        latent_dim: int,
        buffer_size: int,
        device: torch.device,
    ) -> None:
        self.generator = generator
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.buffer_size = buffer_size
        self.device = device
        self._images: Optional[Tensor] = None
        self._classes: Optional[Tensor] = None

    def refresh(self, generator: DSSGANGenerator) -> None:
        generator.eval()
        with torch.no_grad():
            z = torch.randn(self.buffer_size, self.latent_dim, device=self.device)
            c = torch.randint(0, self.num_classes, (self.buffer_size,), device=self.device)
            imgs = generator(z, c).cpu()
        self._images = imgs
        self._classes = c.cpu()

    def as_dataloader(self, batch_size: int) -> DataLoader:
        if self._images is None:
            raise RuntimeError("Call refresh() before as_dataloader().")
        ds = TensorDataset(self._images, self._classes)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)
