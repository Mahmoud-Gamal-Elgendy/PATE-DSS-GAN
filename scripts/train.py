"""
PATE-DSS-GAN Training Script.

Usage
-----
# Start with CIFAR-10 for fast pipeline verification:
python scripts/train.py --config configs/cifar10_32.yaml

# CelebA-HQ 128x128 proof of concept:
python scripts/train.py --config configs/celeba_hq_128.yaml

# Resume from checkpoint:
python scripts/train.py --config configs/celeba_hq_128.yaml \\
    --resume checkpoints/celeba_hq_128/ckpt_step500.pt

# Override privacy budget from command line:
python scripts/train.py --config configs/celeba_hq_128.yaml --epsilon 5.0
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from dataclasses import fields

from src.data.dataset import get_dataset
from src.training.trainer import PATEDSSGANTrainer, TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train PATE-DSS-GAN")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--epsilon", type=float, default=None, help="Override target_epsilon")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    return parser.parse_args()


def load_config(config_path: str, overrides: dict) -> TrainConfig:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    cfg_dict.update({k: v for k, v in overrides.items() if v is not None})

    valid_fields = {f.name for f in fields(TrainConfig)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_fields}

    return TrainConfig(**cfg_dict)


def main():
    args = parse_args()

    overrides = {
        "target_epsilon": args.epsilon,
        "device": args.device,
        "output_dir": args.output_dir,
    }

    config = load_config(args.config, overrides)

    print("=" * 60)
    print("PATE-DSS-GAN Configuration")
    print("=" * 60)
    for f in fields(config):
        print(f"  {f.name:25s}: {getattr(config, f.name)}")
    print("=" * 60)

    # Load dataset
    print(f"\n[Data] Loading {config.dataset_name} from {config.data_root} ...")
    dataset = get_dataset(
        name=config.dataset_name,
        root=config.data_root,
        image_size=config.image_size,
        train=True,
        augment=True,
        download=(config.dataset_name == "cifar10"),
    )
    print(f"[Data] Dataset size: {len(dataset)} samples")

    # Build trainer
    trainer = PATEDSSGANTrainer(config=config, dataset=dataset)

    # Optionally resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
