"""
Ablation study script for PATE-DSS-GAN.

Ablation 1: Scan directions (1-dir vs 3-dir under privacy constraints)
  - Train three models: row-only, col-only, row+col+diag
  - Compare FID at matched ε budget

Ablation 2: Privacy-utility curve
  - Train models with ε ∈ {1, 3, 5, 8, 10}
  - Plot FID vs ε (honest reporting: flat curve is also publishable)

Ablation 3: Number of teachers k
  - k ∈ {5, 10, 20} at fixed ε
  - Shows effect on teacher vote quality

Usage
-----
# Scan direction ablation:
python scripts/ablation.py \\
    --config configs/celeba_hq_128.yaml \\
    --ablation scan_directions \\
    --variants "['row']" "['col']" "['row','col','diag']"

# Privacy-utility curve:
python scripts/ablation.py \\
    --config configs/celeba_hq_128.yaml \\
    --ablation epsilon \\
    --epsilon_values 1.0 3.0 5.0 8.0 10.0
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import json
import ast
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import fields, replace
from typing import List

from src.data.dataset import get_dataset, make_dataloader
from src.training.trainer import PATEDSSGANTrainer, TrainConfig
from src.training.evaluation import Evaluator


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


def load_config(config_path: str) -> TrainConfig:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    valid_fields = {f.name for f in fields(TrainConfig)}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_fields}
    return TrainConfig(**cfg_dict)


def run_experiment(config: TrainConfig, train_dataset, test_dataset, tag: str, output_dir: str) -> dict:
    """Run a single training + evaluation experiment."""
    cfg = replace(
        config,
        output_dir=os.path.join(output_dir, tag, "samples"),
        checkpoint_dir=os.path.join(output_dir, tag, "checkpoints"),
    )

    trainer = PATEDSSGANTrainer(config=cfg, dataset=train_dataset)
    trainer.train()

    # Evaluate on the held-out TEST split — not the training set.
    # Using train=False ensures FID is not inflated by train-set memorisation.
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(device=device, batch_size=32)
    test_loader = make_dataloader(test_dataset, batch_size=32, shuffle=False)
    metrics = evaluator.evaluate(
        generator=trainer.generator,
        real_dataloader=test_loader,
        num_samples=5000,
        num_classes=cfg.num_classes,
        compute_pr=True,
        compute_dc=False,
    )
    metrics["epsilon_achieved"] = trainer.accountant.get_epsilon()
    metrics["sigma"] = trainer.sigma
    return metrics


def plot_results(x_values, y_values, x_label: str, y_label: str,
                 title: str, output_path: str) -> None:
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
    config = load_config(args.config)
    if args.device:
        config.device = args.device

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training split (private data for teachers + generator)
    train_dataset = get_dataset(
        name=config.dataset_name,
        root=config.data_root,
        image_size=config.image_size,
        train=True,
        augment=False,
    )

    # Held-out test split for honest FID/KID evaluation.
    # This is the split passed to the evaluator — never seen by teachers or generator.
    test_dataset = get_dataset(
        name=config.dataset_name,
        root=config.data_root,
        image_size=config.image_size,
        train=False,
        augment=False,
    )

    all_results = {}

    if args.ablation == "scan_directions":
        variants = args.variants or [
            "['row']",
            "['row', 'col']",
            "['row', 'col', 'diag']",
        ]
        fid_values = []
        labels = []
        for variant_str in variants:
            directions = ast.literal_eval(variant_str)
            tag = "_".join(directions)
            cfg = replace(config, scan_directions=directions)
            print(f"\n{'='*50}\nAblation: scan={directions}\n{'='*50}")
            metrics = run_experiment(cfg, train_dataset, test_dataset, f"scan_{tag}", args.output_dir)
            all_results[tag] = metrics
            fid_values.append(metrics["fid"])
            labels.append(tag)
            print(f"[{tag}] FID={metrics['fid']:.2f}, ε={metrics['epsilon_achieved']:.4f}")

        plot_results(
            range(len(labels)), fid_values,
            "Scan Direction Configuration", "FID",
            "FID vs Scan Directions (Privacy-Constrained)",
            os.path.join(args.output_dir, "ablation_scan_directions.png"),
        )

    elif args.ablation == "epsilon":
        epsilon_values = args.epsilon_values
        fid_values = []
        for eps in epsilon_values:
            cfg = replace(config, target_epsilon=eps)
            tag = f"eps_{eps:.1f}"
            print(f"\n{'='*50}\nAblation: ε={eps}\n{'='*50}")
            metrics = run_experiment(cfg, train_dataset, test_dataset, tag, args.output_dir)
            all_results[tag] = metrics
            fid_values.append(metrics["fid"])
            print(f"[ε={eps}] FID={metrics['fid']:.2f}, σ={metrics['sigma']:.4f}")

        plot_results(
            epsilon_values, fid_values,
            "Privacy Budget ε", "FID",
            "Privacy-Utility Curve (FID vs ε)",
            os.path.join(args.output_dir, "ablation_privacy_utility_curve.png"),
        )

    elif args.ablation == "num_teachers":
        k_values = args.k_values
        fid_values = []
        for k in k_values:
            cfg = replace(config, num_teachers=k)
            tag = f"k_{k}"
            print(f"\n{'='*50}\nAblation: k={k}\n{'='*50}")
            metrics = run_experiment(cfg, train_dataset, test_dataset, tag, args.output_dir)
            all_results[tag] = metrics
            fid_values.append(metrics["fid"])
            print(f"[k={k}] FID={metrics['fid']:.2f}, ε={metrics['epsilon_achieved']:.4f}")

        plot_results(
            k_values, fid_values,
            "Number of Teachers (k)", "FID",
            "FID vs Number of Teachers",
            os.path.join(args.output_dir, "ablation_num_teachers.png"),
        )

    # Save all results as JSON
    out_json = os.path.join(args.output_dir, f"ablation_{args.ablation}_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Ablation] All results saved → {out_json}")


if __name__ == "__main__":
    main()
