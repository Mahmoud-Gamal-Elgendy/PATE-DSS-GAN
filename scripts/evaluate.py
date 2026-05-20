"""
Evaluation script for PATE-DSS-GAN.

Computes FID, KID, Precision, Recall, Density, Coverage for a trained
generator checkpoint. Also reports the privacy budget ε at evaluation time.

Usage
-----
python scripts/evaluate.py \\
    --config configs/celeba_hq_128.yaml \\
    --checkpoint checkpoints/celeba_hq_128/ckpt_final.pt \\
    --num_samples 10000

# Evaluate at multiple epsilon levels (for privacy-utility curve):
python scripts/evaluate.py \\
    --config configs/celeba_hq_128.yaml \\
    --checkpoint_dir checkpoints/celeba_hq_128/ \\
    --epsilon_curve
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import json
import torch
from pathlib import Path
from dataclasses import fields

from src.data.dataset import get_dataset, make_dataloader
from src.models.generator import DSSGANGenerator
from src.training.evaluation import Evaluator
from src.training.trainer import TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PATE-DSS-GAN")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--output", type=str, default=None, help="JSON output path for results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_pr", action="store_true", help="Skip Precision/Recall (slow)")
    parser.add_argument("--no_dc", action="store_true", help="Skip Density/Coverage (slow)")
    return parser.parse_args()


def load_config(config_path: str) -> TrainConfig:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    valid_fields = {f.name for f in fields(TrainConfig)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    return TrainConfig(**cfg_dict)


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.device:
        config.device = args.device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"[Eval] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    sigma = ckpt.get("sigma", "N/A")
    epsilon = ckpt.get("epsilon", "N/A")
    step = ckpt.get("step", "N/A")
    print(f"[Eval] Step: {step} | ε: {epsilon} | σ: {sigma}")

    # Build generator
    generator = DSSGANGenerator(
        latent_dim=config.latent_dim,
        num_classes=config.num_classes,
        image_size=config.image_size,
        base_channels=config.base_channels_gen,
        mamba_d_model=config.mamba_d_model,
    ).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    # Load held-out test split for honest FID evaluation.
    # train=False ensures the evaluator never sees training images.
    print(f"[Eval] Loading test split from {config.data_root} ...")
    test_dataset = get_dataset(
        name=config.dataset_name,
        root=config.data_root,
        image_size=config.image_size,
        train=False,
        augment=False,
    )
    real_loader = make_dataloader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate
    evaluator = Evaluator(device=device, batch_size=64)
    results = evaluator.evaluate(
        generator=generator,
        real_dataloader=real_loader,
        num_samples=args.num_samples,
        num_classes=config.num_classes,
        compute_pr=not args.no_pr,
        compute_dc=not args.no_dc,
    )

    # Add privacy metadata
    results["epsilon"] = float(epsilon) if isinstance(epsilon, (int, float)) else epsilon
    results["sigma"] = float(sigma) if isinstance(sigma, (int, float)) else sigma
    results["step"] = int(step) if isinstance(step, int) else step

    # Save results
    out_path = args.output or os.path.join(
        config.output_dir,
        f"eval_step{step}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Eval] Results saved to {out_path}")


if __name__ == "__main__":
    main()
