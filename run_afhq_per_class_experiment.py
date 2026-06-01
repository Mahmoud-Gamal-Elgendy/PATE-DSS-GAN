"""
AFHQ per-class DP experiment for PATE-DSS-GAN (`training_mode = "per_class_dp"`).

Trains one isolated PATE-DSS-GAN per AFHQ class (cat / dog / wild), each under
its own ε = 10 budget, then merges the synthetic outputs at public class ratios.
By parallel composition the merged release is (ε = 10, δ = 1e-5)-DP overall.

Run from the project root:
    cd PATE-DSS-GAN
    python run_afhq_per_class_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data.dataset import get_dataset
from src.training.class_partitioned import ClassPartitionedRunner


# ── Experiment constants ─────────────────────────────────────────────────────

TARGET_EPSILON = 10.0
BATCH_SIZE = 128
EPSILON_MILESTONES = [1.0, 2.0, 4.0, 8.0, 10.0]
NUM_SAMPLE_IMAGES = 256
SAMPLE_GRID_SIZE = 16
IMAGE_SIZE = 128
CLASS_NAMES = ["cat", "dog", "wild"]

RESULTS_ROOT = PROJECT_ROOT / "results" / "afhq_128_eps10_per_class"


def build_config() -> TrainConfig:
    """AFHQ 128×128 per-class config — DSS-GAN paper Table 13 (128 baseline)."""
    return TrainConfig(
        dataset_name="afhq",
        image_size=IMAGE_SIZE,
        num_classes=3,                      # architecture kept 3-class (fixed label per run)

        # Privacy / mode
        training_mode="per_class_dp",
        target_epsilon=TARGET_EPSILON,
        per_class_target_epsilon=TARGET_EPSILON,
        delta=1.0e-5,
        num_teachers=20,
        num_queries=5000,
        n_student_steps=1,
        merge_ratio_mode="public_counts",
        num_synthetic_per_class=NUM_SAMPLE_IMAGES,

        # Architecture — Table 13 @ 128×128
        latent_dim=152,
        base_channels_gen=148,
        base_channels_disc=96,
        channel_max=512,
        base_channels_teacher=32,
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

        # Logging (milestones handled by MilestoneTrainer)
        log_interval=100,
        save_interval=999999,

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

    runner = ClassPartitionedRunner(
        base_config=cfg,
        dataset=dataset,
        results_root=RESULTS_ROOT,
        class_names=CLASS_NAMES,
        milestones=EPSILON_MILESTONES,
        num_sample_images=NUM_SAMPLE_IMAGES,
        sample_grid_size=SAMPLE_GRID_SIZE,
    )
    summary = runner.run()

    print(f"\n[Done] Overall ε (parallel composition) = {summary['overall_epsilon']:.4f}")
    print(f"[Done] Results root: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
