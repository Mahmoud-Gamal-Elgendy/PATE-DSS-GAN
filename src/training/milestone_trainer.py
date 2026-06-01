"""
Milestone-aware PATE-DSS-GAN trainer.

Extracted from run_afhq_epsilon_experiment.py so it can be reused by both the
single-run epsilon-milestone experiment and the per-class DP orchestration
(src/training/class_partitioned.py).

Behaviour is identical to the original inline MilestoneTrainer, with two
additions for reuse:
  - `results_root`, `milestones`, `num_sample_images`, `sample_grid_size` are
    constructor parameters (no module-level globals).
  - `fixed_sample_label`: when set, every synthetic sample is generated with
    this single class label instead of random labels. Used by per_class_dp so a
    per-class model is only ever sampled with its own (trained) class.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Set

import torch
from torchvision.utils import save_image

from ..config import TrainConfig
from .trainer import PATEDSSGANTrainer, _FakeImageBuffer


class MilestoneTrainer(PATEDSSGANTrainer):
    """Trainer that snapshots model + samples at privacy-budget milestones."""

    def __init__(
        self,
        config: TrainConfig,
        dataset,
        milestones: List[float],
        results_root: Path,
        num_sample_images: int = 256,
        sample_grid_size: int = 16,
        fixed_sample_label: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, dataset=dataset)
        self.milestones = sorted(milestones)
        self.results_root = Path(results_root)
        self.num_sample_images = num_sample_images
        self.sample_grid_size = sample_grid_size
        self.fixed_sample_label = fixed_sample_label
        self._saved: Set[float] = set()
        self.milestone_log: List[dict] = []

    # ------------------------------------------------------------------
    # Milestone handling
    # ------------------------------------------------------------------

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
        n_grid = min(self.sample_grid_size, cfg.batch_size)
        n_total = self.num_sample_images
        chunk = 32

        # Use EMA weights if available; load into main generator temporarily
        orig_state = None
        if self.ema is not None:
            orig_state = {k: v.clone() for k, v in self.generator.state_dict().items()}
            self.ema.copy_to(self.generator)
        self.generator.eval()

        # Helper: class labels for a batch of size n.
        def _labels(n: int) -> torch.Tensor:
            if self.fixed_sample_label is not None:
                return torch.full((n,), self.fixed_sample_label,
                                  dtype=torch.long, device=self.device)
            return torch.randint(0, cfg.num_classes, (n,), device=self.device)

        # Preview grid
        z_grid = self.generator.sample_latent(n_grid, self.device)
        if self.fixed_sample_label is not None:
            c_grid = torch.full((n_grid,), self.fixed_sample_label,
                                dtype=torch.long, device=self.device)
        else:
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
            c = _labels(n)
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
                "fixed_label": self.fixed_sample_label,
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.config
        print(f"\n{'='*60}")
        print("PATE-DSS-GAN — Epsilon Milestone Training")
        print(f"  Target ε: {cfg.target_epsilon} | σ: {self.sigma:.4f}")
        print(f"  Batch size: {cfg.batch_size}")
        print(f"  Milestones: {self.milestones}")
        print(f"  Teachers: {cfg.num_teachers} | Image size: {cfg.image_size}×{cfg.image_size}")
        if self.fixed_sample_label is not None:
            print(f"  Fixed sample label: {self.fixed_sample_label}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        import time

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

        # Per-run milestone log (matches original single-run format)
        log_path = self.results_root / "milestone_log.json"
        self.results_root.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(
                {
                    "target_epsilon": cfg.target_epsilon,
                    "batch_size": cfg.batch_size,
                    "milestones_requested": self.milestones,
                    "milestones_saved": sorted(self._saved),
                    "final_epsilon": self.accountant.get_epsilon(),
                    "sigma": self.sigma,
                    "fixed_sample_label": self.fixed_sample_label,
                    "records": self.milestone_log,
                },
                f,
                indent=2,
            )
        print(f"[Experiment] Milestone log → {log_path}")
