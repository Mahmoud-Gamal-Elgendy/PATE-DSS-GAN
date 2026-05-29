"""
AFHQ epsilon-milestone experiment for PATE-DSS-GAN.

Single training run on AFHQ at 128×128 with target ε = 10 and batch size 128.
Hyperparameters match DSS-GAN paper Table 13 (128×128 baseline).
When cumulative privacy spend reaches ε ∈ {1, 2, 4, 8, 10}, the script
saves a checkpoint and generates synthetic sample data at that budget.

Run from the project root:
    cd PATE-DSS-GAN
    python run_afhq_epsilon_experiment.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Set

import torch
from torchvision.utils import save_image

# Ensure project root is on sys.path when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data.dataset import get_dataset
from src.training.trainer import PATEDSSGANTrainer, _FakeImageBuffer


# ── Experiment constants ─────────────────────────────────────────────────────

TARGET_EPSILON = 10.0
BATCH_SIZE = 128
EPSILON_MILESTONES = [1.0, 2.0, 4.0, 8.0, 10.0]
NUM_SAMPLE_IMAGES = 256          # synthetic images saved per milestone
SAMPLE_GRID_SIZE = 16            # images in the preview grid

IMAGE_SIZE = 128

RESULTS_ROOT = PROJECT_ROOT / "results" / "afhq_128_eps_milestone_bs128"


def build_config() -> TrainConfig:
    """Inline AFHQ 128×128 config — DSS-GAN paper Table 13 (128 baseline)."""
    return TrainConfig(
        # Dataset
        dataset_name="afhq",
        image_size=IMAGE_SIZE,
        num_classes=3,

        # Privacy (PATE-specific — not in DSS-GAN paper)
        target_epsilon=TARGET_EPSILON,
        delta=1.0e-5,
        num_teachers=20,
        num_queries=5000,
        n_student_steps=1,          # aligned with DSS-GAN D_STEPS=1

        # Architecture — Table 13 @ 128×128
        # Latent: D_base=92, D_dir=20 → z_dim=92+20×3=152
        latent_dim=152,
        base_channels_gen=148,       # G channels 8×8–64×64; 128×128→168 via generator
        base_channels_disc=96,       # D base channels
        channel_max=512,             # D max channels
        base_channels_teacher=32,    # PATE teacher CNN (not in paper)
        scan_directions=["row_fwd", "col_bwd", "diag_left"],

        # Training — Table 13 @ 128×128
        batch_size=BATCH_SIZE,
        gen_lr=9e-5,
        student_lr=3e-5,
        teacher_lr=0.0002,
        teacher_epochs=3,
        retrain_interval=50,
        max_outer_steps=15000,
        optimizer_betas=[0.0, 0.99],

        # Stabilisation — Table 13 @ 128×128
        use_ema=True,
        ema_decay=0.999,
        ema_decay_2=0.9995,
        ema_switch_images=1_000_000,
        use_r1=True,
        r1_gamma=5.0,
        r1_interval=4,
        use_diffaug=True,
        diffaug_brightness=0.1,
        diffaug_contrast=0.1,
        diffaug_flip_prob=0.5,
        gradient_clip_gen=10.0,
        gradient_clip_disc=15.0,

        # Logging (milestones handled separately)
        log_interval=100,
        save_interval=999999,     # disable periodic saves; milestones only
        output_dir=str(RESULTS_ROOT / "samples"),
        checkpoint_dir=str(RESULTS_ROOT / "checkpoints"),

        # Misc
        seed=42,
        device="cuda",
        num_workers=2,
        pin_memory=True,
    )


class MilestoneTrainer(PATEDSSGANTrainer):
    """Trainer that snapshots model + samples at privacy-budget milestones."""

    def __init__(
        self,
        config: TrainConfig,
        dataset,
        milestones: List[float],
        num_sample_images: int = NUM_SAMPLE_IMAGES,
    ) -> None:
        super().__init__(config=config, dataset=dataset)
        self.milestones = sorted(milestones)
        self.num_sample_images = num_sample_images
        self._saved: Set[float] = set()
        self.milestone_log: List[dict] = []

    def _maybe_save_milestones(self, step: int) -> None:
        eps = self.accountant.get_epsilon()
        for milestone in self.milestones:
            if milestone in self._saved:
                continue
            if eps >= milestone:
                self._save_at_milestone(milestone, step, eps)
                self._saved.add(milestone)

    def _save_at_milestone(self, milestone: float, step: int, eps: float) -> None:
        tag = f"eps_{milestone:g}"
        ckpt_dir = Path(self.config.checkpoint_dir) / tag
        sample_dir = Path(self.config.output_dir) / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = ckpt_dir / f"ckpt_{tag}.pt"
        ckpt_dict = {
            "step": step,
            "milestone_epsilon": milestone,
            "generator": self.generator.state_dict(),
            "student": self.student.state_dict(),
            "opt_gen": self.opt_gen.state_dict(),
            "opt_student": self.opt_student.state_dict(),
            "accountant_steps": self.accountant.steps,
            "sigma": self.sigma,
            "epsilon": eps,
        }
        if self.ema is not None:
            ckpt_dict["ema_shadow"] = self.ema.shadow
        torch.save(ckpt_dict, ckpt_path)

        self._generate_samples(sample_dir, tag)

        record = {
            "milestone_epsilon": milestone,
            "achieved_epsilon": eps,
            "step": step,
            "queries": self.accountant.steps,
            "checkpoint": str(ckpt_path),
            "samples_dir": str(sample_dir),
        }
        self.milestone_log.append(record)

        print(
            f"\n[Milestone ε={milestone:g}] "
            f"achieved ε={eps:.4f} at step={step} | "
            f"checkpoint → {ckpt_path}\n"
        )

    @torch.no_grad()
    def _generate_samples(self, sample_dir: Path, tag: str) -> None:
        cfg = self.config
        n_grid = min(SAMPLE_GRID_SIZE, cfg.batch_size)
        n_total = self.num_sample_images
        chunk = 32

        # Use EMA weights if available; load into main generator temporarily
        orig_state = None
        if self.ema is not None:
            orig_state = {k: v.clone() for k, v in self.generator.state_dict().items()}
            self.ema.copy_to(self.generator)
        self.generator.eval()

        # Preview grid
        z_grid = self.generator.sample_latent(n_grid, self.device)
        c_grid = torch.arange(n_grid, device=self.device) % cfg.num_classes
        grid_imgs = self.generator(z_grid, c_grid)
        grid_path = sample_dir / f"grid_{tag}.png"
        save_image(grid_imgs * 0.5 + 0.5, str(grid_path), nrow=4)
        del grid_imgs, z_grid, c_grid
        print(f"[Milestone] Saved preview grid → {grid_path}")

        # Generate full synthetic batch in chunks to avoid OOM
        all_imgs, all_labels, all_z = [], [], []
        for start in range(0, n_total, chunk):
            n = min(chunk, n_total - start)
            z = self.generator.sample_latent(n, self.device)
            c = torch.randint(0, cfg.num_classes, (n,), device=self.device)
            imgs = self.generator(z, c)
            all_imgs.append(imgs.cpu())
            all_labels.append(c.cpu())
            all_z.append(z.cpu())

        all_imgs = torch.cat(all_imgs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_z = torch.cat(all_z, dim=0)

        images_path = sample_dir / f"synthetic_images_{tag}.pt"
        torch.save(
            {
                "images": all_imgs,
                "labels": all_labels,
                "latents": all_z,
                "num_samples": n_total,
                "epsilon_milestone": tag,
            },
            images_path,
        )
        print(f"[Milestone] Saved {n_total} synthetic images → {images_path}")

        # Export PNGs for quick inspection
        png_dir = sample_dir / "png"
        png_dir.mkdir(exist_ok=True)
        export_n = min(64, n_total)
        for i in range(export_n):
            save_image(
                all_imgs[i] * 0.5 + 0.5,
                str(png_dir / f"sample_{i:04d}_class{all_labels[i].item()}.png"),
            )
        print(f"[Milestone] Exported {export_n} PNGs → {png_dir}")

        del all_imgs, all_labels, all_z

        # Restore original weights if EMA was applied
        if orig_state is not None:
            self.generator.load_state_dict(orig_state)
            del orig_state

        torch.cuda.empty_cache()

    def train(self) -> None:
        cfg = self.config
        print(f"\n{'='*60}")
        print("PATE-DSS-GAN — AFHQ Epsilon Milestone Experiment")
        print(f"  Target ε: {cfg.target_epsilon} | σ: {self.sigma:.4f}")
        print(f"  Batch size: {cfg.batch_size}")
        print(f"  Milestones: {self.milestones}")
        print(f"  Teachers: {cfg.num_teachers} | Image size: {cfg.image_size}×{cfg.image_size}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        fake_buffer = _FakeImageBuffer(
            generator=self.generator,
            num_classes=cfg.num_classes,
            latent_dim=cfg.latent_dim,
            buffer_size=cfg.batch_size * 20,
            device=self.device,
        )

        t0 = time.time()

        for step in range(cfg.max_outer_steps):
            did_retrain = self.ensemble_manager.maybe_retrain_check(step)
            if did_retrain:
                fake_buffer.refresh(self.generator)
                self.ensemble_manager.do_retrain(
                    fake_buffer.as_dataloader(cfg.batch_size), step=step
                )

            fake_imgs, fake_classes = self._sample_fake(cfg.batch_size)
            noisy_labels, confidence = self.ensemble_manager.query_with_confidence(fake_imgs)

            current_eps = self.accountant.get_epsilon()
            self._maybe_save_milestones(step)

            if current_eps >= cfg.target_epsilon:
                self._maybe_save_milestones(step)
                print(
                    f"\n[Step {step:05d}] Privacy budget exhausted: "
                    f"ε={current_eps:.3f} ≥ {cfg.target_epsilon:.0f}  "
                    f"(queries={self.accountant.steps})"
                )
                break

            d_loss = self._student_update(
                fake_imgs, noisy_labels, cfg.n_student_steps, c=fake_classes
            )
            g_loss = self._generator_update(cfg.batch_size)

            if step % cfg.log_interval == 0:
                eps = self.accountant.get_epsilon()
                budget_pct = 100.0 * eps / cfg.target_epsilon
                elapsed = time.time() - t0
                conf_mean = confidence.float().mean().item()
                saved_tags = [f"ε={m:g}" for m in sorted(self._saved)]
                print(
                    f"[Step {step:05d}] "
                    f"ε={eps:.3f}/{cfg.target_epsilon:.0f} ({budget_pct:.1f}%) | "
                    f"queries={self.accountant.steps} | "
                    f"G={g_loss:.4f} | D={d_loss:.4f} | "
                    f"TeacherConf={conf_mean:.3f} | "
                    f"saved=[{', '.join(saved_tags) or 'none'}] | {elapsed:.0f}s"
                )
                self.metrics["epsilon"].append(eps)
                self.metrics["g_loss"].append(g_loss)
                self.metrics["d_loss"].append(d_loss)
                self.metrics["teacher_confidence"].append(conf_mean)

        print("\n[Experiment] Training complete.")

        log_path = RESULTS_ROOT / "milestone_log.json"
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(
                {
                    "target_epsilon": TARGET_EPSILON,
                    "batch_size": BATCH_SIZE,
                    "milestones_requested": EPSILON_MILESTONES,
                    "milestones_saved": sorted(self._saved),
                    "final_epsilon": self.accountant.get_epsilon(),
                    "sigma": self.sigma,
                    "records": self.milestone_log,
                },
                f,
                indent=2,
            )
        print(f"[Experiment] Milestone log → {log_path}")


def main() -> None:
    cfg = build_config()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] Loading AFHQ dataset ({IMAGE_SIZE}×{IMAGE_SIZE}, 3 classes) ...")
    dataset = get_dataset(
        name=cfg.dataset_name,
        image_size=cfg.image_size,
        train=True,
        augment=True,
    )
    print(f"[Setup] Dataset loaded — {len(dataset)} samples")

    trainer = MilestoneTrainer(
        config=cfg,
        dataset=dataset,
        milestones=EPSILON_MILESTONES,
        num_sample_images=NUM_SAMPLE_IMAGES,
    )
    trainer.train()

    print(f"\n[Done] Final ε={trainer.accountant.get_epsilon():.4f} | σ={trainer.sigma:.4f}")
    print(f"[Done] Milestones saved: {sorted(trainer._saved)}")
    print(f"[Done] Results root: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
