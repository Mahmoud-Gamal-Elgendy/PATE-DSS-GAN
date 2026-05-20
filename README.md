# PATE-DSS-GAN

**Privacy-Preserving High-Resolution Image Synthesis via PATE with Mamba Backbone**

> First differentially private GAN with both a Mamba generator (DSS-GAN) and a Mamba student discriminator, using the PATE privacy framework.

## Proposal Context

This codebase implements the research proposal for combining:
- **PATE** (Papernot et al. 2018) — differentially private teacher ensemble
- **DSS-GAN** (Section 3) — hierarchical Mamba generator with Directional Latent Routing (DLR)

---

## Project Structure

```
PATE-DSS-GAN/
├── src/
│   ├── accountant.py           # GNMax RDP accountant + calibrate_sigma() [CRITICAL]
│   ├── data/
│   │   ├── dataset.py          # CelebA-HQ / AFHQ / CIFAR-10 loaders
│   │   └── partitioner.py      # Stratified k-shard partitioner
│   ├── models/
│   │   ├── teacher.py          # Simple CNN teacher discriminators
│   │   ├── student.py          # Mamba student discriminator
│   │   └── generator.py        # DSS-GAN generator with DLR
│   ├── pate/
│   │   ├── voting.py           # GNMax vote aggregation (no threshold)
│   │   └── ensemble.py         # Teacher ensemble orchestration
│   └── training/
│       ├── trainer.py          # Main training loop
│       └── evaluation.py       # FID, KID, Precision, Recall, D&C
├── configs/
│   ├── cifar10_32.yaml         # Fast prototyping
│   ├── celeba_hq_128.yaml      # Proof of concept [START HERE]
│   ├── celeba_hq_256.yaml      # Primary experiment
│   └── afhq_256.yaml           # Secondary experiment
├── scripts/
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation with FID/KID/PR
│   ├── ablation.py             # Ablation studies
│   └── sigma_analysis.py       # σ calibration analysis [RUN FIRST]
├── docs/
│   └── ARCHITECTURE.md         # Detailed component reference
└── requirements.txt
```

---


### Privacy-Correct Vote Aggregation

From `src/pate/voting.py`:
- **No confident threshold** — all images are queried and all labels recorded
- Gaussian noise `N(0, σ²)` added to both real/fake vote counts (GNMax)
- Every release charges the privacy budget via `GNMaxRDPAccountant.step()`

### Mamba Student Discriminator

From `src/models/student.py`:
- CNN downsampling to 8×8 bottleneck
- Mamba SSM blocks applied to 3 scan directions (row / col / diagonal)
- Directional outputs pooled and combined → scalar real/fake logit
- Ablation: `scan_directions=['row']` for 1-direction baseline

---

## Datasets

All datasets load automatically — no manual download steps required.

| Dataset | Source | Classes | Size | Notes |
|---------|--------|---------|------|-------|
| CelebA-HQ | `korexyz/celeba-hq-256x256` on HuggingFace | 2 (female / male) | 30K | Primary experiment |
| AFHQ | `huggan/AFHQ` on HuggingFace | 3 (cat / dog / wild) | 15K | Secondary experiment |


### Loading datasets manually

```python
from datasets import load_dataset

# CelebA-HQ 256×256
ds = load_dataset("korexyz/celeba-hq-256x256")

# AFHQ (cat / dog / wild)
ds = load_dataset("huggan/AFHQ")
```

Both are wrapped automatically by `src/data/dataset.py` when you call `get_dataset()`.
HuggingFace caches the data under `~/.cache/huggingface/` after the first download.

---

## Comparison Baselines

| Method | Type | Notes |
|--------|------|-------|
| SPTI | Text-guided DP synthesis | Compare FID at matched ε |
| DP-Diffusion | DP diffusion model | Compare FID at matched ε |

---
