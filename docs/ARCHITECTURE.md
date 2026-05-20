# PATE-DSS-GAN Architecture Reference

## Overview

PATE-DSS-GAN combines:
- **PATE** (Private Aggregation of Teacher Ensembles) for differential privacy
- **DSS-GAN's Mamba backbone** for high-resolution image generation
- **Novel Mamba-based student discriminator** (first in PATE literature)

## Component Map

```
src/
├── accountant.py           GNMax RDP accountant + calibrate_sigma()
├── data/
│   ├── dataset.py          CelebA-HQ / AFHQ / CIFAR-10 loaders
│   └── partitioner.py      Stratified k-shard partitioner
├── models/
│   ├── teacher.py          Simple CNN teacher discriminators
│   ├── student.py          Mamba student discriminator (CNN + Mamba@8x8)
│   └── generator.py        DSS-GAN generator with DLR
├── pate/
│   ├── voting.py           GNMax vote aggregation (no threshold)
│   └── ensemble.py         Ensemble orchestration + retraining
└── training/
    ├── trainer.py          Main PATE-DSS-GAN training loop
    └── evaluation.py       FID, KID, Precision, Recall, Density, Coverage
```

## Privacy Correctness

| Issue | Bug | Fix |
|-------|-----|-----|
| σ calibration | Hardcoded per dataset | `calibrate_sigma()` binary search from (ε,N,k,δ) |
| Confident threshold | Privacy leak (mask before noise) | **Removed** — all images queried, all labels recorded |
| Teacher anchoring | Trained on G_init output only | Retrained every `retrain_interval` steps on current G |
| Data partitioning | Random split → degenerate shards | **Stratified** by class label |
| Seed selection | Test set for best-seed search | Validation set only |

## Student Discriminator Architecture

```
Input (B, 3, H, W)
    ↓ CNN DiscriminatorBlocks (H → 8)
Bottleneck (B, C, 8, 8)
    ↓ Row scan  → Mamba → pooling
    ↓ Col scan  → Mamba → pooling   } concatenate → project
    ↓ Diag scan → Mamba → pooling
Linear → (B, 1) logit
```

Key properties:
- O(N) complexity at bottleneck (vs O(N²) attention)
- Directional scanning with ablation support (1-dir / 3-dir)
- Mamba-ssm CUDA kernel used when available; pure-PyTorch fallback for CPU dev

## Generator Architecture

```
z (B, latent_dim)  +  c (B,) class label
    ↓ Embedding + MLP mapping network → w (style)
Learned constant (B, C, 8, 8)
    ↓ DLRBlock × n_ups (8×8 → target resolution)
      Each DLRBlock: Mamba(row) + Mamba(col) + Mamba(diag)
                     weighted by class-conditional routing
StyleGAN2 refinement (AdaIN)
ToRGB → Tanh → (B, 3, H, W) ∈ [-1, 1]
```

## Training Loop

```
Outer step t:
  1. [Every 50 steps] Retrain k teachers on shard + current G output
  2. Sample z, c → G(z,c) = fake_imgs
  3. k teachers vote real/fake on fake_imgs → votes (k, B)
  4. GNMax: add N(0,σ²) to vote counts → noisy_labels (B,)   ← charges ε
  5. Post-processing: n_s=5 student updates on (fake_imgs, noisy_labels)
  6. Generator update: minimise -log(σ(D_student(G(z))))
  7. Check ε ≥ target_epsilon → stop if exhausted
```

## Datasets

| Dataset | Resolution | Classes | Notes |
|---------|-----------|---------|-------|
| CelebA-HQ | 128/256 | 2 (binary) | Primary experiment |
| AFHQ | 256 | 3 (cat/dog/wild) | Secondary experiment |
| CIFAR-10 | 32 | 10 | Fast prototyping only |

LSUN Bedroom excluded (no class labels; DLR is ineffective without conditioning).

## Privacy Expectations (Honest Reporting)

Tight privacy (ε ≤ 1) may yield near-random labels due to large σ.

| ε | Expectation |
|---|------------|
| 10 | Utility likely; validate here first |
| 5-8 | Acceptable utility trade-off |
| 3 | Possible utility degradation |
| 1 | Likely near-random labels; publishable as negative result |

**The privacy-utility curve, even if flat, is a publishable finding.**
Report honestly: do not cherry-pick ε values.
