"""
Synthetic dataset merge for per-class DP training (pure post-processing).

Each per-class PATE-DSS-GAN model writes a synthetic tensor at its target ε
milestone (``synthetic_images_eps_<m>.pt``). This module loads those generated
tensors and concatenates them into a single labelled synthetic dataset, sampling
per-class counts according to public mixing ratios.

Privacy note
------------
This operation is **post-processing** of already-released, DP-protected
generator outputs (Dwork & Roth, Prop. 2.1). It reads ONLY generated ``.pt``
tensors — never the private training data. The mixing ratios are derived from
public per-class counts, so releasing them incurs no additional privacy cost.
The merged dataset therefore carries the same (ε, δ) guarantee as the per-class
models under parallel composition (ε = max over classes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch


class SyntheticMerger:
    """Merge per-class synthetic tensors into one dataset at public ratios."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    @staticmethod
    def _counts_from_ratios(ratios: Dict[int, float], total: int) -> Dict[int, int]:
        """Largest-remainder apportionment so per-class counts sum exactly to total."""
        s = sum(ratios.values())
        raw = {c: (total * r / s) for c, r in ratios.items()}
        floored = {c: int(v) for c, v in raw.items()}
        remainder = total - sum(floored.values())
        # distribute the remaining slots to the largest fractional parts
        fracs = sorted(raw, key=lambda c: raw[c] - floored[c], reverse=True)
        for i in range(remainder):
            floored[fracs[i % len(fracs)]] += 1
        return floored

    def merge(
        self,
        class_synthetic_paths: Dict[int, Path],
        ratios: Dict[int, float],
        total: int,
        output_dir: Path,
        ratio_source: str = "public_counts",
    ) -> Dict:
        """
        Parameters
        ----------
        class_synthetic_paths : {class_id: path to synthetic_images_*.pt}
        ratios : {class_id: relative weight}  (e.g. public per-class counts)
        total : desired number of samples in the merged dataset
        output_dir : where to write merged/{synthetic_dataset.pt, class_distribution.json}
        ratio_source : provenance string logged for auditability

        Returns
        -------
        dict : the class_distribution.json content (also written to disk).
        """
        g = torch.Generator().manual_seed(self.seed)
        target_counts = self._counts_from_ratios(ratios, total)

        merged_imgs: List[torch.Tensor] = []
        merged_labels: List[torch.Tensor] = []
        actual_counts: Dict[int, int] = {}

        for class_id, path in class_synthetic_paths.items():
            blob = torch.load(path, map_location="cpu", weights_only=False)
            imgs = blob["images"]
            n_avail = imgs.shape[0]
            want = target_counts.get(class_id, 0)

            if want <= n_avail:
                sel = torch.randperm(n_avail, generator=g)[:want]
            else:
                # not enough generated samples: sample with replacement
                sel = torch.randint(0, n_avail, (want,), generator=g)
                print(
                    f"[Merge] class {class_id}: requested {want} > available "
                    f"{n_avail}; sampling with replacement."
                )

            merged_imgs.append(imgs[sel])
            merged_labels.append(torch.full((want,), class_id, dtype=torch.long))
            actual_counts[class_id] = want

        images = torch.cat(merged_imgs, dim=0)
        labels = torch.cat(merged_labels, dim=0)

        # Shuffle the combined set (deterministic)
        perm = torch.randperm(images.shape[0], generator=g)
        images, labels = images[perm], labels[perm]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ds_path = output_dir / "synthetic_dataset.pt"
        torch.save(
            {"images": images, "labels": labels, "num_samples": int(images.shape[0])},
            ds_path,
        )

        distribution = {
            "total": int(images.shape[0]),
            "ratio_source": ratio_source,
            "ratios_used": {str(c): float(r) for c, r in ratios.items()},
            "counts_per_class": {str(c): int(n) for c, n in actual_counts.items()},
            "privacy_note": (
                "Mixing ratios derived from public per-class counts; merge is "
                "post-processing of DP-protected generator outputs → no "
                "additional privacy cost."
            ),
            "synthetic_dataset": str(ds_path),
        }
        dist_path = output_dir / "class_distribution.json"
        with open(dist_path, "w") as f:
            json.dump(distribution, f, indent=2)

        print(f"[Merge] Wrote {images.shape[0]} samples → {ds_path}")
        print(f"[Merge] Distribution → {dist_path}")
        return distribution
