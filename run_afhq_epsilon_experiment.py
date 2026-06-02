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

import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data.dataset import get_dataset
from src.training.milestone_trainer import MilestoneTrainer


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
        query_reuse_factor=250,     # 250 free post-processing updates per charged query

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
        retrain_interval=5,          # retrain every 5 outer steps (~8× during run)
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
        diffaug_brightness=0.5,      # stronger aug → prevents color saturation
        diffaug_contrast=0.5,
        diffaug_flip_prob=0.5,
        gradient_clip_gen=1.0,       # tighter clip → prevents tanh saturation
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

    # MilestoneTrainer now lives in src/training/milestone_trainer.py (shared
    # with the per-class DP orchestration). fixed_sample_label=None preserves
    # the original behaviour of sampling random class labels.
    trainer = MilestoneTrainer(
        config=cfg,
        dataset=dataset,
        milestones=EPSILON_MILESTONES,
        results_root=RESULTS_ROOT,
        num_sample_images=NUM_SAMPLE_IMAGES,
        sample_grid_size=SAMPLE_GRID_SIZE,
        fixed_sample_label=None,
    )
    trainer.train()

    print(f"\n[Done] Final ε={trainer.accountant.get_epsilon():.4f} | σ={trainer.sigma:.4f}")
    print(f"[Done] Milestones saved: {sorted(trainer._saved)}")
    print(f"[Done] Results root: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
