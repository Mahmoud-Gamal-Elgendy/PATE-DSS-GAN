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

Training stabilisation (aligned with DSS_GAN-main):
  - EMA of generator weights.
  - R1 gradient penalty on the student discriminator.
  - Differentiable augmentation (brightness, contrast, flip) on fake images.
  - Gradient clipping for both G and D.
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from ..config import TrainConfig
from ..accountant import GNMaxRDPAccountant, calibrate_sigma
from ..data.dataset import make_dataloader, ImageDatasetWrapper
from ..data.partitioner import stratified_partition
from ..models.teacher import build_teacher_ensemble
from ..models.student import MambaStudentDiscriminator
from ..models.generator import DSSGANGenerator
from ..pate.voting import PATEVoteAggregator
from ..pate.ensemble import PATEEnsembleManager


# ---------------------------------------------------------------------------
# Training utilities (aligned with DSS_GAN-main/src/utils.py)
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


class DiffAug(nn.Module):
    """Differentiable augmentation (brightness, contrast, horizontal flip)."""

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        flip_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.flip_prob = flip_prob

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        if self.brightness > 0:
            b = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 2 * self.brightness
            x = x + b
        if self.contrast > 0:
            mean = x.mean(dim=[1, 2, 3], keepdim=True)
            c = 1.0 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 2 * self.contrast
            x = mean + (x - mean) * c
        if self.flip_prob > 0 and self.training:
            mask = torch.rand(B, 1, 1, 1, device=x.device) < self.flip_prob
            x = torch.where(mask, x.flip([3]), x)
        return x


def r1_gradient_penalty(d_out: Tensor, x_in: Tensor) -> Tensor:
    """R1 gradient penalty (Mescheder et al., 2018)."""
    grad = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad.pow(2).flatten(1).sum(1).mean()


def weights_init(m: nn.Module) -> None:
    """Xavier uniform init for Conv/Linear layers (matches DSS_GAN-main)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


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

        # Multi-GPU: put teachers on GPU 1 if available
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            self.teacher_device = torch.device("cuda:1")
            print(f"[Multi-GPU] Found {torch.cuda.device_count()} GPUs → "
                  f"G+D on {self.device}, Teachers on {self.teacher_device}")
        else:
            self.teacher_device = self.device

        torch.manual_seed(config.seed)

        # Set up output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._setup_privacy()
        self._setup_models()
        self._setup_data()
        self._setup_optimisers()
        self._setup_stabilisation()

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
            scan_directions=cfg.scan_directions,
        ).to(self.device)

        self.student = MambaStudentDiscriminator(
            image_size=cfg.image_size,
            base_channels=cfg.base_channels_disc,
            num_classes=cfg.num_classes,
            channel_max=cfg.channel_max,
            mamba_d_model=cfg.mamba_d_model,
            mamba_layers=cfg.mamba_layers,
            scan_directions=cfg.scan_directions,
        ).to(self.device)

        # Weight init matching DSS_GAN-main
        def _init_non_mamba(m: nn.Module) -> None:
            if "mamba" not in m.__class__.__name__.lower():
                weights_init(m)

        self.generator.apply(_init_non_mamba)
        self.student.apply(weights_init)

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
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            device=self.teacher_device,
        )

    def _setup_optimisers(self) -> None:
        cfg = self.config
        betas = tuple(cfg.optimizer_betas)
        self.opt_gen = torch.optim.Adam(
            self.generator.parameters(), lr=cfg.gen_lr, betas=betas
        )
        self.opt_student = torch.optim.Adam(
            self.student.parameters(), lr=cfg.student_lr, betas=betas
        )

    def _setup_stabilisation(self) -> None:
        cfg = self.config

        # EMA (two-phase schedule matching DSS-GAN Table 13)
        self.ema: Optional[EMA] = None
        self._gen_images_seen = 0
        self._ema_switched = False
        if cfg.use_ema:
            self.ema = EMA(self.generator, decay=cfg.ema_decay)
            print(
                f"[Stabilisation] EMA enabled "
                f"(decay={cfg.ema_decay} → {cfg.ema_decay_2} "
                f"after {cfg.ema_switch_images:,} images)"
            )

        # DiffAug
        self.diff_aug: Optional[DiffAug] = None
        if cfg.use_diffaug:
            self.diff_aug = DiffAug(
                brightness=cfg.diffaug_brightness,
                contrast=cfg.diffaug_contrast,
                flip_prob=cfg.diffaug_flip_prob,
            ).to(self.device)
            self.diff_aug.train()
            print(f"[Stabilisation] DiffAug enabled")

        # R1 + gradient clipping params stored for use in training step
        self._r1_step_counter = 0
        if cfg.use_r1:
            print(f"[Stabilisation] R1 enabled (γ={cfg.r1_gamma}, interval={cfg.r1_interval})")
        if cfg.gradient_clip_gen > 0:
            print(f"[Stabilisation] Grad clip: G={cfg.gradient_clip_gen}, D={cfg.gradient_clip_disc}")

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
        c: Optional[Tensor] = None,
    ) -> float:
        """
        Post-processing student discriminator update (privacy-free).

        The PATE labels are already released, so n_s gradient steps
        on the student using these labels incur NO additional ε cost.
        Class labels c are passed to enable projection conditioning.

        Includes R1 gradient penalty and DiffAug (both privacy-safe since
        they operate only on synthetic data).
        """
        cfg = self.config
        self.student.train()
        loss_fn = nn.BCEWithLogitsLoss()
        total_loss = 0.0

        for _ in range(n_steps):
            self._r1_step_counter += 1
            self.opt_student.zero_grad()

            imgs = fake_imgs.detach()
            if self.diff_aug is not None:
                imgs = self.diff_aug(imgs)

            logits = self.student(imgs, c).view(-1)
            labels = noisy_labels.float()
            loss = loss_fn(logits, labels)

            # R1 gradient penalty (on "real" = PATE-labelled-as-real fakes)
            if cfg.use_r1 and self._r1_step_counter % cfg.r1_interval == 0:
                real_mask = noisy_labels.bool()
                if real_mask.any():
                    real_imgs = imgs[real_mask].requires_grad_(True)
                    real_c = c[real_mask] if c is not None else None
                    d_real_out = self.student(real_imgs, real_c).view(-1)
                    r1_pen = r1_gradient_penalty(d_real_out, real_imgs)
                    loss = loss + 0.5 * cfg.r1_gamma * r1_pen

            loss.backward()
            if cfg.gradient_clip_disc > 0:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), cfg.gradient_clip_disc)
            self.opt_student.step()
            total_loss += loss.item()

        self.student.eval()
        return total_loss / n_steps

    def _generator_update(self, batch_size: int) -> float:
        """Update generator to fool the student discriminator (non-saturating logistic loss)."""
        cfg = self.config
        self.generator.train()
        self.opt_gen.zero_grad()

        z = self.generator.sample_latent(batch_size, self.device)
        c = torch.randint(0, cfg.num_classes, (batch_size,), device=self.device)
        fake_imgs = self.generator(z, c)

        if self.diff_aug is not None:
            fake_imgs_aug = self.diff_aug(fake_imgs)
        else:
            fake_imgs_aug = fake_imgs

        logits = self.student(fake_imgs_aug, c).view(-1)
        g_loss = F.softplus(-logits).mean()

        g_loss.backward()
        if cfg.gradient_clip_gen > 0:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), cfg.gradient_clip_gen)
        self.opt_gen.step()

        # EMA update (image-count switch matches DSS-GAN train.py)
        if self.ema is not None:
            self.ema.update(self.generator)
            self._gen_images_seen += batch_size
            if (
                not self._ema_switched
                and self._gen_images_seen >= cfg.ema_switch_images
            ):
                self.ema.decay = cfg.ema_decay_2
                self._ema_switched = True
                print(
                    f"[Stabilisation] EMA decay → {cfg.ema_decay_2} "
                    f"after {self._gen_images_seen:,} generator images"
                )

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
            d_loss = self._student_update(fake_imgs, noisy_labels, cfg.n_student_steps, c=fake_classes)

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

        # Use EMA generator for sample quality if available
        if self.ema is not None:
            ema_gen = copy.deepcopy(self.generator)
            self.ema.copy_to(ema_gen)
            ema_gen.eval()
            with torch.no_grad():
                imgs = ema_gen(z, c)
            del ema_gen
        else:
            with torch.no_grad():
                imgs = self.generator(z, c)

        path = os.path.join(cfg.output_dir, f"samples_step{step}.png")
        save_image(imgs * 0.5 + 0.5, path, nrow=4)
        print(f"[Trainer] Saved samples → {path}")

    def _save_checkpoint(self, step) -> None:
        cfg = self.config
        path = os.path.join(cfg.checkpoint_dir, f"ckpt_step{step}.pt")
        ckpt_dict = {
            "step": step,
            "generator": self.generator.state_dict(),
            "student": self.student.state_dict(),
            "opt_gen": self.opt_gen.state_dict(),
            "opt_student": self.opt_student.state_dict(),
            "accountant_steps": self.accountant.steps,
            "sigma": self.sigma,
            "epsilon": self.accountant.get_epsilon(),
        }
        if self.ema is not None:
            ckpt_dict["ema_shadow"] = self.ema.shadow
        torch.save(ckpt_dict, path)
        print(f"[Trainer] Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(ckpt["generator"])
        self.student.load_state_dict(ckpt["student"])
        self.opt_gen.load_state_dict(ckpt["opt_gen"])
        self.opt_student.load_state_dict(ckpt["opt_student"])
        self.accountant._steps = ckpt["accountant_steps"]
        if self.ema is not None and "ema_shadow" in ckpt:
            self.ema.shadow = ckpt["ema_shadow"]
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

    def refresh(self, generator: DSSGANGenerator, chunk_size: int = 64) -> None:
        """Regenerate the fake buffer in small chunks to avoid GPU OOM."""
        generator.eval()
        all_imgs: List[Tensor] = []
        all_classes: List[Tensor] = []
        with torch.no_grad():
            for start in range(0, self.buffer_size, chunk_size):
                n = min(chunk_size, self.buffer_size - start)
                z = torch.randn(n, self.latent_dim, device=self.device)
                c = torch.randint(0, self.num_classes, (n,), device=self.device)
                all_imgs.append(generator(z, c).cpu())
                all_classes.append(c.cpu())
        self._images = torch.cat(all_imgs, dim=0)
        self._classes = torch.cat(all_classes, dim=0)

    def as_dataloader(self, batch_size: int) -> DataLoader:
        if self._images is None:
            raise RuntimeError("Call refresh() before as_dataloader().")
        ds = TensorDataset(self._images, self._classes)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
