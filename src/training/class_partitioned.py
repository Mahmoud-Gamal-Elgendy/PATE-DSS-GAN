"""
Class-partitioned DP training orchestration for PATE-DSS-GAN.

Implements the `training_mode = "per_class_dp"` experiment mode: instead of one
model over the whole dataset, train one isolated PATE-DSS-GAN per class, each
under its own ε budget, then merge the synthetic outputs at public class ratios.

Privacy model (parallel composition)
-------------------------------------
The private records are partitioned disjointly by their *public* class label.
Each class model is an isolated (ε, δ)-DP mechanism that reads ONLY its own
class subset — a fresh trainer (own generator/student/teachers/accountant/σ) is
built per class and destroyed before the next, so no mutable state is shared.
By parallel composition (McSherry 2009) the joint release of all class models is
(max_i ε_i, δ)-DP — NOT the sum. Merging the synthetic outputs at public ratios
is post-processing and incurs no additional privacy cost.

This orchestrator deliberately introduces no cross-class statistic: dataset
normalization is a fixed constant, σ-calibration uses no data, and each teacher
sees only its own shard. Isolation therefore holds by construction.
"""

from __future__ import annotations

import copy
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..config import TrainConfig
from ..data.dataset import ImageDatasetWrapper
from .milestone_trainer import MilestoneTrainer
from .synthetic_merge import SyntheticMerger


class ClassPartitionedRunner:
    """Train one isolated PATE-DSS-GAN per class, then merge synthetic outputs."""

    def __init__(
        self,
        base_config: TrainConfig,
        dataset: ImageDatasetWrapper,
        results_root: Path,
        class_names: Optional[List[str]] = None,
        milestones: Optional[List[float]] = None,
        num_sample_images: int = 256,
        sample_grid_size: int = 16,
    ) -> None:
        self.base_config = base_config
        self.dataset = dataset
        self.results_root = Path(results_root)
        self.class_names = class_names
        self.milestones = sorted(milestones) if milestones else [base_config.per_class_target_epsilon]
        self.num_sample_images = num_sample_images
        self.sample_grid_size = sample_grid_size

        self.per_class_records: Dict[int, dict] = {}

    # ------------------------------------------------------------------

    def _class_name(self, class_id: int) -> str:
        if self.class_names and class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def _resolve_ratios(self, class_counts: Dict[int, int]) -> Dict[int, float]:
        cfg = self.base_config
        if cfg.merge_ratio_mode == "equal":
            return {c: 1.0 for c in class_counts}
        if cfg.merge_ratio_mode == "custom" and cfg.merge_class_ratios:
            return {c: float(cfg.merge_class_ratios[i]) for i, c in enumerate(sorted(class_counts))}
        # default: public per-class counts
        return {c: float(n) for c, n in class_counts.items()}

    # ------------------------------------------------------------------

    def run(self) -> dict:
        cfg = self.base_config
        self.results_root.mkdir(parents=True, exist_ok=True)

        # 1) Disjoint class partition (asserts exhaustive + disjoint internally)
        subsets = self.dataset.get_class_subsets()
        class_counts = self.dataset.class_counts()
        ratios = self._resolve_ratios(class_counts)

        # 2) Top-level metadata (written before training)
        metadata = {
            "training_mode": "per_class_dp",
            "dataset_name": cfg.dataset_name,
            "image_size": cfg.image_size,
            "num_classes": cfg.num_classes,
            "class_names": self.class_names,
            "per_class_counts": {str(c): int(n) for c, n in class_counts.items()},
            "per_class_target_epsilon": cfg.per_class_target_epsilon,
            "delta": cfg.delta,
            "milestones": self.milestones,
            "merge": {
                "ratio_mode": cfg.merge_ratio_mode,
                "ratios": {str(c): float(r) for c, r in ratios.items()},
                "num_synthetic_per_class": cfg.num_synthetic_per_class,
            },
            "privacy_composition_statement": (
                "Records are partitioned disjointly by public class label; each "
                f"class model is an isolated (ε={cfg.per_class_target_epsilon}, "
                f"δ={cfg.delta})-DP mechanism over its own subset. By parallel "
                f"composition the joint release is (ε={cfg.per_class_target_epsilon}, "
                f"δ={cfg.delta})-DP (max over classes, not sum). Synthetic merge at "
                "public class ratios is post-processing and incurs no additional "
                "privacy cost."
            ),
        }
        with open(self.results_root / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[PerClassDP] Metadata → {self.results_root / 'metadata.json'}")

        # 3) Sequential per-class training — fresh isolated trainer each
        class_synthetic_paths: Dict[int, Path] = {}
        for class_id in sorted(subsets):
            name = self._class_name(class_id)
            class_root = self.results_root / name
            class_root.mkdir(parents=True, exist_ok=True)

            cfg_c = copy.deepcopy(cfg)
            cfg_c.target_epsilon = cfg.per_class_target_epsilon
            cfg_c.output_dir = str(class_root / "milestones")
            cfg_c.checkpoint_dir = str(class_root / "checkpoints")

            # snapshot per-class config
            with open(class_root / "config.json", "w") as f:
                json.dump(asdict(cfg_c), f, indent=2)

            print(f"\n{'#'*60}\n[PerClassDP] Training class {class_id} ({name}) "
                  f"| {len(subsets[class_id])} samples\n{'#'*60}")

            t0 = time.time()
            trainer = MilestoneTrainer(
                config=cfg_c,
                dataset=subsets[class_id],
                milestones=self.milestones,
                results_root=class_root,
                num_sample_images=self.num_sample_images,
                sample_grid_size=self.sample_grid_size,
                fixed_sample_label=class_id,   # sample only this trained class
            )
            trainer.train()
            elapsed = time.time() - t0

            # per-class accountant log
            achieved_eps = trainer.accountant.get_epsilon()
            acct_log = {
                "class_id": class_id,
                "class_name": name,
                "achieved_epsilon": achieved_eps,
                "sigma": trainer.sigma,
                "queries": trainer.accountant.steps,
                "milestones_saved": sorted(trainer._saved),
                "epsilon_trajectory": trainer.metrics["epsilon"],
                "teacher_confidence": trainer.metrics["teacher_confidence"],
                "wall_time_sec": elapsed,
            }
            with open(class_root / "accountant_log.json", "w") as f:
                json.dump(acct_log, f, indent=2)

            self.per_class_records[class_id] = {
                "name": name,
                "achieved_epsilon": achieved_eps,
                "sigma": trainer.sigma,
                "queries": trainer.accountant.steps,
                "milestones_saved": sorted(trainer._saved),
            }

            # locate the highest-milestone synthetic tensor for the merge
            target_tag = f"eps_{cfg.per_class_target_epsilon:g}"
            synth = class_root / "milestones" / target_tag / f"synthetic_images_{target_tag}.pt"
            if not synth.exists():
                # fall back to the last milestone that was actually saved
                saved = sorted(trainer._saved)
                if saved:
                    tag = f"eps_{saved[-1]:g}"
                    synth = class_root / "milestones" / tag / f"synthetic_images_{tag}.pt"
            if synth.exists():
                class_synthetic_paths[class_id] = synth
            else:
                print(f"[PerClassDP] WARNING: no synthetic tensor found for class {class_id}")

            # explicit isolation: tear down before next class
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 4) Merge (post-processing)
        merge_dist = None
        if class_synthetic_paths:
            total = cfg.num_synthetic_per_class * len(class_synthetic_paths)
            merge_ratios = {c: ratios[c] for c in class_synthetic_paths}
            merge_dist = SyntheticMerger(seed=cfg.seed).merge(
                class_synthetic_paths=class_synthetic_paths,
                ratios=merge_ratios,
                total=total,
                output_dir=self.results_root / "merged",
                ratio_source=cfg.merge_ratio_mode,
            )

        # 5) Summary report (overall ε = max over classes)
        overall_eps = max((r["achieved_epsilon"] for r in self.per_class_records.values()),
                          default=float("nan"))
        summary = {
            "overall_epsilon": overall_eps,
            "composition": "parallel (ε = max over disjoint class partitions)",
            "per_class": {str(c): r for c, r in self.per_class_records.items()},
            "merge": merge_dist,
        }
        with open(self.results_root / "summary_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[PerClassDP] Summary → {self.results_root / 'summary_report.json'}")
        print(f"[PerClassDP] Overall ε (parallel composition) = {overall_eps:.4f}")
        return summary
