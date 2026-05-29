"""
Multi-dataset privacy-utility ablation for PATE-DSS-GAN.

Demonstrates the modular experiment pattern:
  - No base-code modifications required.
  - All configuration via ``TrainConfig`` / ``EvalConfig`` dataclasses.
  - Each experiment variant is a ``dataclasses.replace()`` call.

Run from the project root:
    cd PATE-DSS-GAN
    python run_ablation_experiment.py
"""

import json
from dataclasses import replace
from pathlib import Path

from src.config import TrainConfig, EvalConfig, load_config
from src.pipeline import run_experiment


# ── Experiment grid ────────────────────────────────────────────────────────

DATASETS = {
    "celeba_hq": load_config("configs/celeba_hq_128.yaml"),
    "afhq":      load_config("configs/afhq_256.yaml"),
}

EPSILONS = [5.0, 8.0, 10.0]

EVAL = EvalConfig(num_samples=5000, compute_pr=True, compute_tstr=True, tstr_num_synth_samples=5000)

RESULTS_ROOT = Path("./results/ablation_multi_dataset")


# ── Run loop ───────────────────────────────────────────────────────────────

def main():
    all_results = {}

    for ds_name, base_cfg in DATASETS.items():
        for eps in EPSILONS:
            tag = f"{ds_name}_eps{eps}"
            out_dir = RESULTS_ROOT / ds_name / f"eps_{eps}"

            cfg = replace(
                base_cfg,
                target_epsilon=eps,
                output_dir=str(out_dir / "samples"),
                checkpoint_dir=str(out_dir / "checkpoints"),
            )

            print(f"\n{'='*60}")
            print(f"  Experiment: {tag}")
            print(f"  Dataset={ds_name}  ε={eps}  k={cfg.num_teachers}")
            print(f"{'='*60}\n")

            result = run_experiment(cfg, eval_config=EVAL)

            summary = {
                "dataset": ds_name,
                "target_epsilon": eps,
                "achieved_epsilon": result.epsilon,
                "sigma": result.sigma,
                **result.metrics,
            }
            all_results[tag] = summary

            print(f"\n>>> [{tag}] FID={result.metrics['fid']:.2f}  "
                  f"ε_achieved={result.epsilon:.4f}  σ={result.sigma:.4f}\n")

    # ── Save aggregate results ─────────────────────────────────────────────
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_ROOT / "all_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] All results saved → {out_path}")


if __name__ == "__main__":
    main()
