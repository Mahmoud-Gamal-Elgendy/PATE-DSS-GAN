"""
Privacy analysis script: σ calibration and ε-σ relationship.

Run this BEFORE training to understand the privacy-utility trade-off
for your specific (N, k, δ) setting.

Usage
-----
python scripts/sigma_analysis.py \\
    --num_queries 5000 \\
    --num_teachers 10 \\
    --delta 1e-5 \\
    --epsilon_values 1.0 3.0 5.0 8.0 10.0

Outputs a table and plots σ vs ε and ε_remaining vs training step.
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.accountant import calibrate_sigma, compute_epsilon_at_sigma, GNMaxRDPAccountant


def parse_args():
    parser = argparse.ArgumentParser(description="PATE σ calibration analysis")
    parser.add_argument("--num_queries", type=int, default=5000)
    parser.add_argument("--num_teachers", type=int, default=10)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--epsilon_values", nargs="*", type=float,
                        default=[1.0, 3.0, 5.0, 8.0, 10.0, 20.0])
    parser.add_argument("--output_dir", type=str, default="./privacy_analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"PATE σ Calibration Analysis")
    print(f"  num_queries (N) = {args.num_queries}")
    print(f"  num_teachers (k)= {args.num_teachers}")
    print(f"  delta (δ)       = {args.delta}")
    print(f"{'='*65}")
    print(f"\n{'ε':>8} | {'σ':>12} | {'ε/query':>10} | {'Feasibility'}")
    print("-" * 55)

    sigmas = []
    for eps in args.epsilon_values:
        try:
            sigma = calibrate_sigma(
                target_epsilon=eps,
                num_queries=args.num_queries,
                num_teachers=args.num_teachers,
                delta=args.delta,
            )
            eps_per_query = eps / args.num_queries
            feasibility = "OK" if eps >= 5.0 else ("RISKY" if eps >= 3.0 else "UTILITY COLLAPSE LIKELY")
            print(f"{eps:>8.1f} | {sigma:>12.4f} | {eps_per_query:>10.6f} | {feasibility}")
            sigmas.append(sigma)
        except ValueError as e:
            print(f"{eps:>8.1f} | {'INFEASIBLE':>12} | {'N/A':>10} | {str(e)[:30]}")
            sigmas.append(None)

    # ε consumed over training steps (for a given σ at ε=10)
    try:
        sigma_10 = calibrate_sigma(10.0, args.num_queries, args.num_teachers, args.delta)
        steps = np.arange(0, args.num_queries + 1, max(1, args.num_queries // 100))
        epsilons = []
        for n in steps:
            if n == 0:
                epsilons.append(0.0)
            else:
                epsilons.append(compute_epsilon_at_sigma(sigma_10, int(n), args.num_teachers, args.delta))

        plt.figure(figsize=(8, 4))
        plt.plot(steps, epsilons, color="#E53935", linewidth=2)
        plt.axhline(y=10.0, linestyle="--", color="#333", alpha=0.5, label="ε=10 budget")
        plt.axhline(y=5.0, linestyle=":", color="#555", alpha=0.4, label="ε=5 threshold")
        plt.xlabel("Training Queries (N)", fontsize=12)
        plt.ylabel("Cumulative ε", fontsize=12)
        plt.title(f"Privacy Budget Consumption (σ={sigma_10:.3f}, k={args.num_teachers})", fontsize=13)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "epsilon_over_queries.png"), dpi=150)
        print(f"\n[Analysis] Budget consumption plot saved.")
    except ValueError:
        print("\n[Analysis] Skipping budget plot (infeasible at ε=10).")

    # σ vs ε plot
    valid_eps = [e for e, s in zip(args.epsilon_values, sigmas) if s is not None]
    valid_sigma = [s for s in sigmas if s is not None]
    if len(valid_eps) >= 2:
        plt.figure(figsize=(7, 4))
        plt.plot(valid_eps, valid_sigma, "o-", color="#1565C0", linewidth=2, markersize=8)
        plt.xlabel("Target ε", fontsize=12)
        plt.ylabel("Required σ", fontsize=12)
        plt.title(f"σ vs ε (k={args.num_teachers}, N={args.num_queries})", fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "sigma_vs_epsilon.png"), dpi=150)
        print(f"[Analysis] σ vs ε plot saved.")

    print(f"\n[Analysis] Recommendation:")
    print(f"  Start validation at ε ∈ {{5, 8, 10}} before testing ε ∈ {{1, 3}}.")
    print(f"  The privacy-utility curve is a publishable finding even if flat.")


if __name__ == "__main__":
    main()
