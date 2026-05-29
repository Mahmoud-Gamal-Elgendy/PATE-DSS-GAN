"""
High-level pipeline API for PATE-DSS-GAN experiments.

Provides three entry-points so that experiment scripts can run the full
train → evaluate workflow in a single function call::

    from src.config import TrainConfig, EvalConfig, load_config
    from src.pipeline import run_experiment

    cfg = load_config("configs/celeba_hq_128.yaml", target_epsilon=5.0)
    result = run_experiment(cfg)
    print(result.metrics["fid"], result.epsilon)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import TrainConfig, EvalConfig
from .data.dataset import get_dataset, make_dataloader
from .models.generator import DSSGANGenerator
from .training.trainer import PATEDSSGANTrainer
from .training.evaluation import Evaluator


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Returned by :func:`run_training`."""
    trainer: PATEDSSGANTrainer
    metrics: Dict[str, Any]
    epsilon: float
    sigma: float
    checkpoint_path: str


@dataclass
class ExperimentResult:
    """Returned by :func:`run_experiment`."""
    metrics: Dict[str, float]
    epsilon: float
    sigma: float
    config: TrainConfig
    checkpoint_path: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_training(
    config: TrainConfig,
    resume: Optional[str] = None,
) -> TrainingResult:
    """
    Load dataset, build the PATE-DSS-GAN trainer, and run the full
    privacy-aware training loop.

    Parameters
    ----------
    config : TrainConfig
        Complete training configuration.
    resume : str, optional
        Path to a checkpoint to resume from.

    Returns
    -------
    TrainingResult
        Contains the trainer instance, logged metrics, final epsilon/sigma,
        and the path to the final checkpoint.
    """
    dataset = get_dataset(
        name=config.dataset_name,
        image_size=config.image_size,
        train=True,
        augment=True,
    )
    print(f"[Pipeline] Dataset '{config.dataset_name}' loaded — {len(dataset)} samples")

    trainer = PATEDSSGANTrainer(config=config, dataset=dataset)

    if resume:
        trainer.load_checkpoint(resume)

    trainer.train()

    final_eps = trainer.accountant.get_epsilon()
    final_sigma = trainer.sigma
    ckpt_path = os.path.join(config.checkpoint_dir, "ckpt_stepfinal.pt")

    return TrainingResult(
        trainer=trainer,
        metrics=trainer.metrics,
        epsilon=final_eps,
        sigma=final_sigma,
        checkpoint_path=ckpt_path,
    )


def run_evaluation(
    config: TrainConfig,
    checkpoint: str,
    eval_config: Optional[EvalConfig] = None,
) -> Dict[str, float]:
    """
    Load a generator checkpoint and evaluate it against the held-out
    test split.

    Parameters
    ----------
    config : TrainConfig
        Used for dataset/architecture parameters.
    checkpoint : str
        Path to a saved ``.pt`` checkpoint.
    eval_config : EvalConfig, optional
        Controls num_samples, precision/recall, and TSTR/TRTR toggles.
        Defaults to ``EvalConfig()`` (10 000 samples, all metrics).

    Returns
    -------
    dict
        Generative quality keys: ``fid``, ``kid_mean``, ``kid_std``,
        ``precision``, ``recall``.
        TSTR/TRTR keys (when ``eval_config.compute_tstr`` is True):
        ``trtr_{clf}``, ``tstr_{clf}`` for each of the 5 classifiers,
        plus ``trtr_mean_acc``, ``tstr_mean_acc``.
        Checkpoint metadata: ``epsilon``, ``sigma``, ``step``.
    """
    if eval_config is None:
        eval_config = EvalConfig()

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint, map_location=device)
    sigma = ckpt.get("sigma", "N/A")
    epsilon = ckpt.get("epsilon", "N/A")
    step = ckpt.get("step", "N/A")
    print(f"[Pipeline/Eval] Checkpoint step={step} | ε={epsilon} | σ={sigma}")

    generator = DSSGANGenerator(
        latent_dim=config.latent_dim,
        num_classes=config.num_classes,
        image_size=config.image_size,
        base_channels=config.base_channels_gen,
        scan_directions=config.scan_directions,
    ).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    test_dataset = get_dataset(
        name=config.dataset_name,
        image_size=config.image_size,
        train=False,
        augment=False,
    )
    real_loader = make_dataloader(test_dataset, batch_size=eval_config.batch_size, shuffle=False)

    evaluator = Evaluator(device=device, batch_size=eval_config.batch_size)

    # Generative quality: FID, KID, Precision, Recall
    results = evaluator.evaluate(
        generator=generator,
        real_dataloader=real_loader,
        num_samples=eval_config.num_samples,
        num_classes=config.num_classes,
        compute_pr=eval_config.compute_pr,
    )

    # Downstream utility: TSTR vs TRTR
    if eval_config.compute_tstr:
        train_dataset = get_dataset(
            name=config.dataset_name,
            image_size=config.image_size,
            train=True,
            augment=False,
        )
        train_loader = make_dataloader(
            train_dataset, batch_size=eval_config.batch_size, shuffle=False
        )
        tstr_results = evaluator.evaluate_tstr_trtr(
            generator=generator,
            train_dataloader=train_loader,
            test_dataloader=real_loader,
            num_synth_samples=eval_config.tstr_num_synth_samples,
            num_classes=config.num_classes,
            cnn_epochs=eval_config.tstr_cnn_epochs,
            cnn_lr=eval_config.tstr_cnn_lr,
        )
        results.update(tstr_results)

    results["epsilon"] = float(epsilon) if isinstance(epsilon, (int, float)) else epsilon
    results["sigma"] = float(sigma) if isinstance(sigma, (int, float)) else sigma
    results["step"] = int(step) if isinstance(step, int) else step

    return results


def run_experiment(
    config: TrainConfig,
    eval_config: Optional[EvalConfig] = None,
    resume: Optional[str] = None,
) -> ExperimentResult:
    """
    End-to-end train **then** evaluate in a single call.

    Parameters
    ----------
    config : TrainConfig
        Full training + dataset configuration.
    eval_config : EvalConfig, optional
        Evaluation settings (defaults to ``EvalConfig()``).
    resume : str, optional
        Checkpoint to resume training from.

    Returns
    -------
    ExperimentResult
        Aggregated metrics, achieved epsilon/sigma, config snapshot, and
        path to the final checkpoint.
    """
    train_result = run_training(config, resume=resume)

    eval_metrics = run_evaluation(
        config,
        checkpoint=train_result.checkpoint_path,
        eval_config=eval_config,
    )

    return ExperimentResult(
        metrics=eval_metrics,
        epsilon=train_result.epsilon,
        sigma=train_result.sigma,
        config=config,
        checkpoint_path=train_result.checkpoint_path,
    )
