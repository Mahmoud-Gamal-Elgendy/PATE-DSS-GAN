"""
Centralized configuration for PATE-DSS-GAN.

All experiment-level knobs live here so that experiment scripts can
``from src.config import TrainConfig, load_config`` without touching
the training internals.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, fields
from typing import List, Optional


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Dataset
    dataset_name: str = "celeba_hq"
    image_size: int = 128
    num_classes: int = 2

    # Privacy
    target_epsilon: float = 10.0
    delta: float = 1e-5
    num_teachers: int = 10
    num_queries: int = 5000
    n_student_steps: int = 1

    # Training mode — "single" (default, whole-dataset) or "per_class_dp"
    # (one isolated model per class; merged via parallel composition).
    training_mode: str = "single"
    # Per-class ε budget under per_class_dp. By parallel composition the merged
    # release is max over classes = this value (NOT the sum).
    per_class_target_epsilon: float = 10.0
    # Synthetic-merge controls (per_class_dp only). "public_counts" uses the
    # dataset's real per-class counts (public for AFHQ) as mixing ratios;
    # "equal" mixes 1:1:1; "custom" uses merge_class_ratios.
    merge_ratio_mode: str = "public_counts"
    merge_class_ratios: Optional[List[float]] = None
    num_synthetic_per_class: int = 5000

    # Architecture (aligned with DSS_GAN-main)
    latent_dim: int = 152
    base_channels_gen: int = 148
    base_channels_disc: int = 96
    channel_max: int = 512
    base_channels_teacher: int = 32
    mamba_d_model: int = 256
    mamba_layers: int = 4
    scan_directions: List[str] = field(default_factory=lambda: ["row_fwd", "col_bwd", "diag_left"])

    # Training (aligned with DSS_GAN-main)
    batch_size: int = 32
    gen_lr: float = 9e-5
    student_lr: float = 3e-5
    teacher_lr: float = 2e-4
    teacher_epochs: int = 3
    retrain_interval: int = 50
    max_outer_steps: int = 10000
    optimizer_betas: List[float] = field(default_factory=lambda: [0.0, 0.99])

    # Regularization & stabilization (aligned with DSS_GAN-main)
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_decay_2: float = 0.9995
    ema_switch_images: int = 1_000_000
    use_r1: bool = True
    r1_gamma: float = 5.0
    r1_interval: int = 4
    use_diffaug: bool = True
    diffaug_brightness: float = 0.1
    diffaug_contrast: float = 0.1
    diffaug_flip_prob: float = 0.5
    gradient_clip_gen: float = 10.0
    gradient_clip_disc: float = 15.0

    # Logging
    log_interval: int = 50
    save_interval: int = 500
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"

    # Misc
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 2
    pin_memory: bool = True


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    num_samples: int = 10000
    compute_pr: bool = True
    compute_tstr: bool = True
    tstr_num_synth_samples: int = 10000
    tstr_cnn_epochs: int = 30
    tstr_cnn_lr: float = 1e-3
    batch_size: int = 64


# ---------------------------------------------------------------------------
# YAML loader with keyword overrides
# ---------------------------------------------------------------------------

def load_config(config_path: str, **overrides) -> TrainConfig:
    """
    Load a YAML config file into a ``TrainConfig``, applying any keyword
    overrides on top.  Unknown keys in the YAML are silently ignored so
    that config files can carry extra metadata without breaking.

    Usage::

        cfg = load_config("configs/celeba_hq_128.yaml", target_epsilon=5.0)
    """
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    cfg_dict.update({k: v for k, v in overrides.items() if v is not None})

    valid_fields = {f.name for f in fields(TrainConfig)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_fields}

    return TrainConfig(**cfg_dict)
