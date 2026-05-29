"""
Ablation study script for PATE-DSS-GAN.

Ablation 1 — Scan directions (1-dir vs 3-dir under privacy constraints)
Ablation 2 — Privacy-utility curve (ε sweep)
Ablation 3 — Number of teachers k

Usage
-----
python scripts/ablation.py \
    --config configs/celeba_hq_128.yaml \
    --ablation epsilon \
    --epsilon_values 1.0 3.0 5.0 8.0 10.0
"""

import argparse
import ast
import json
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, EvalConfig
from src.pipeline import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="PATE-DSS-GAN Ablation Study")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ablation", type=str, required=True,
                        choices=["scan_directions", "epsilon", "num_teachers"])
    parser.add_argument("--variants", nargs="*", help="Scan direction variants (Python lists)")
    parser.add_argument("--epsilon_values", nargs="*", type=float,
                        default=[1.0, 3.0, 5.0, 8.0, 10.0])
    parser.add_argument("--k_values", nargs="*", type=int, default=[5, 10, 20])
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="./ablation_results")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def plot_results(x_values, y_values, x_label, y_label, title, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, "o-", color="#2196F3", linewidth=2, markersize=8)
    plt.xlabel(x_label, fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[Ablation] Plot saved → {output_path}")


def main():
    args = parse_args()
    config = load_config(args.config, device=args.device)
    eval_cfg = EvalConfig(num_samples=args.num_samples, compute_pr=True,
                          compute_tstr=True, tstr_num_synth_samples=args.num_samples)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    all_results = {}

    if args.ablation == "scan_directions":
        variants = args.variants or [
            "['row']",
            "['row', 'col']",
            "['row', 'col', 'diag']",
        ]
        fid_values, labels = [], []
        for variant_str in variants:
            directions = ast.literal_eval(variant_str)
            tag = "_".join(directions)
            cfg = replace(config,
                          scan_directions=directions,
                          output_dir=os.path.join(args.output_dir, f"scan_{tag}", "samples"),
                          checkpoint_dir=os.path.join(args.output_dir, f"scan_{tag}", "ckpts"))
            print(f"\n{'='*50}\nAblation: scan={directions}\n{'='*50}")
            result = run_experiment(cfg, eval_config=eval_cfg)
            all_results[tag] = result.metrics
            fid_values.append(result.metrics["fid"])
            labels.append(tag)
            print(f"[{tag}] FID={result.metrics['fid']:.2f}, ε={result.epsilon:.4f}")

        plot_results(range(len(labels)), fid_values,
                     "Scan Direction Configuration", "FID",
                     "FID vs Scan Directions (Privacy-Constrained)",
                     os.path.join(args.output_dir, "ablation_scan_directions.png"))

    elif args.ablation == "epsilon":
        fid_values = []
        for eps in args.epsilon_values:
            cfg = replace(config,
                          target_epsilon=eps,
                          output_dir=os.path.join(args.output_dir, f"eps_{eps:.1f}", "samples"),
                          checkpoint_dir=os.path.join(args.output_dir, f"eps_{eps:.1f}", "ckpts"))
            print(f"\n{'='*50}\nAblation: ε={eps}\n{'='*50}")
            result = run_experiment(cfg, eval_config=eval_cfg)
            tag = f"eps_{eps:.1f}"
            all_results[tag] = result.metrics
            fid_values.append(result.metrics["fid"])
            print(f"[ε={eps}] FID={result.metrics['fid']:.2f}, σ={result.sigma:.4f}")

        plot_results(args.epsilon_values, fid_values,
                     "Privacy Budget ε", "FID",
                     "Privacy-Utility Curve (FID vs ε)",
                     os.path.join(args.output_dir, "ablation_privacy_utility_curve.png"))

    elif args.ablation == "num_teachers":
        fid_values = []
        for k in args.k_values:
            cfg = replace(config,
                          num_teachers=k,
                          output_dir=os.path.join(args.output_dir, f"k_{k}", "samples"),
                          checkpoint_dir=os.path.join(args.output_dir, f"k_{k}", "ckpts"))
            print(f"\n{'='*50}\nAblation: k={k}\n{'='*50}")
            result = run_experiment(cfg, eval_config=eval_cfg)
            tag = f"k_{k}"
            all_results[tag] = result.metrics
            fid_values.append(result.metrics["fid"])
            print(f"[k={k}] FID={result.metrics['fid']:.2f}, ε={result.epsilon:.4f}")

        plot_results(args.k_values, fid_values,
                     "Number of Teachers (k)", "FID",
                     "FID vs Number of Teachers",
                     os.path.join(args.output_dir, "ablation_num_teachers.png"))

    out_json = os.path.join(args.output_dir, f"ablation_{args.ablation}_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Ablation] All results saved → {out_json}")


if __name__ == "__main__":
    main()
