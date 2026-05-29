"""
PATE-DSS-GAN Training Script.

Usage
-----
# From the project root:
python scripts/train.py --config configs/celeba_hq_128.yaml

# Resume from checkpoint:
python scripts/train.py --config configs/celeba_hq_128.yaml \
    --resume checkpoints/celeba_hq_128/ckpt_step500.pt

# Override privacy budget from command line:
python scripts/train.py --config configs/celeba_hq_128.yaml --epsilon 5.0
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.pipeline import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train PATE-DSS-GAN")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--epsilon", type=float, default=None, help="Override target_epsilon")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(
        args.config,
        target_epsilon=args.epsilon,
        device=args.device,
        output_dir=args.output_dir,
    )

    result = run_training(config, resume=args.resume)
    print(f"\n[Done] Final ε={result.epsilon:.4f} | σ={result.sigma:.4f}")
    print(f"[Done] Checkpoint → {result.checkpoint_path}")


if __name__ == "__main__":
    main()
