"""
Evaluation script for PATE-DSS-GAN.

Computes FID, KID, Precision, Recall, and TSTR/TRTR (5 classifiers) for a
trained generator checkpoint. Also reports the privacy budget at evaluation
time.

Usage
-----
python scripts/evaluate.py \
    --config configs/celeba_hq_128.yaml \
    --checkpoint checkpoints/celeba_hq_128/ckpt_final.pt \
    --num_samples 10000
"""

import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, EvalConfig
from src.pipeline import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PATE-DSS-GAN")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--tstr_num_synth", type=int, default=10000,
                        help="Synthetic samples for TSTR training sets")
    parser.add_argument("--output", type=str, default=None, help="JSON output path for results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_pr", action="store_true", help="Skip Precision/Recall")
    parser.add_argument("--no_tstr", action="store_true", help="Skip TSTR/TRTR evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config, device=args.device)

    eval_config = EvalConfig(
        num_samples=args.num_samples,
        compute_pr=not args.no_pr,
        compute_tstr=not args.no_tstr,
        tstr_num_synth_samples=args.tstr_num_synth,
    )

    results = run_evaluation(config, checkpoint=args.checkpoint, eval_config=eval_config)

    out_path = args.output or os.path.join(
        config.output_dir,
        f"eval_step{results.get('step', 'unknown')}.json",
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Eval] Results saved to {out_path}")


if __name__ == "__main__":
    main()
