# PATE-DSS-GAN

**Privacy-Preserving High-Resolution Image Synthesis via PATE with Mamba Backbone**

> First differentially private GAN with both a Mamba generator (DSS-GAN) and a Mamba student discriminator, using the PATE privacy framework.

## Proposal Context

This codebase implements the research proposal for combining:
- **PATE** (Papernot et al. 2018) — differentially private teacher ensemble
- **DSS-GAN** (Section 3) — hierarchical Mamba generator with Directional Latent Routing (DLR)
- **Novel contribution** — first Mamba-based student discriminator in a PATE framework

Target venue: **ICDM 2026** (June 5th)

---

## Novelty

1. First PATE framework with **Mamba-based discriminator** (O(N) vs O(N²) attention)
2. First **differentially private GAN** with Mamba in both generator and discriminator
3. **Directional privacy-utility trade-off**: DLR provides class-specific spatial priors under privacy constraints
4. **Scalable**: DSS-GAN demonstrated 512×512; PATE-DSS-GAN maintains this with DP guarantees

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

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# For Mamba CUDA kernel (requires CUDA 11.6+):
pip install mamba-ssm
# A pure-PyTorch fallback is used automatically if mamba-ssm is unavailable.
```

### 2. Analyse Privacy Budget (Run This First)

Before training, understand the σ-ε trade-off for your configuration:

```bash
python scripts/sigma_analysis.py \
    --num_queries 5000 \
    --num_teachers 10 \
    --delta 1e-5 \
    --epsilon_values 1.0 3.0 5.0 8.0 10.0
```

This outputs the required σ for each ε budget and plots the budget consumption curve.

### 3. Verify Pipeline on CIFAR-10

```bash
python scripts/train.py --config configs/cifar10_32.yaml
```

CIFAR-10 is downloaded automatically via torchvision.

### 4. Train on CelebA-HQ 128×128 (Proof of Concept)

No manual download needed. The dataset streams from HuggingFace Hub on first use:

```bash
python scripts/train.py --config configs/celeba_hq_128.yaml
```

### 5. Train on AFHQ 256×256

```bash
python scripts/train.py --config configs/afhq_256.yaml
```

### 6. Evaluate

```bash
python scripts/evaluate.py \
    --config configs/celeba_hq_128.yaml \
    --checkpoint checkpoints/celeba_hq_128/ckpt_final.pt \
    --num_samples 10000
```

### 7. Run Ablation Studies

```bash
# Privacy-utility curve (key result):
python scripts/ablation.py \
    --config configs/celeba_hq_128.yaml \
    --ablation epsilon \
    --epsilon_values 1.0 3.0 5.0 8.0 10.0

# Scan direction ablation:
python scripts/ablation.py \
    --config configs/celeba_hq_128.yaml \
    --ablation scan_directions \
    --variants "['row']" "['row','col']" "['row','col','diag']"
```

---

## Key Implementation Details

### `calibrate_sigma()` — Critical Path

Located in `src/accountant.py`. Derives σ via binary search from `(ε, N, k, δ)`.
**This must be called at training start**, not hardcoded per dataset.

```python
from src.accountant import calibrate_sigma

sigma = calibrate_sigma(
    target_epsilon=10.0,
    num_queries=5000,
    num_teachers=10,
    delta=1e-5,
)
```

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
| CIFAR-10 | torchvision (auto-download) | 10 | 60K | Fast prototyping |

**LSUN Bedroom excluded** — no class labels; DLR conditioning is ineffective.

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

## Privacy Expectations (Honest Reporting)

| ε | σ (typical) | Expected Utility |
|---|-------------|-----------------|
| 10 | ~1.0–2.0 | Functional GAN output |
| 5–8 | ~2.0–4.0 | Moderate utility loss |
| 3 | ~5.0–10.0 | Possible degradation |
| 1 | very large | Near-random labels likely |

**The privacy-utility curve, even if flat, is a publishable finding.**

---

## Comparison Baselines

| Method | Type | Notes |
|--------|------|-------|
| SPTI | Text-guided DP synthesis | Compare FID at matched ε |
| DP-Diffusion | DP diffusion model | Compare FID at matched ε |
| PATE-GAN | Tabular PATE-GAN | Image adaptation baseline |

---

## Citation

If you use this code, please cite the works it builds on:
- PATE: Papernot et al. (2018), *Scalable Private Learning with PATE*
- DSS-GAN: *Directional State-Space GAN*
- Mamba: Gu & Dao (2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
