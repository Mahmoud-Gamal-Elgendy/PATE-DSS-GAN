# PATE-DSS-GAN

**Differentially Private High-Resolution Image Synthesis via State-Space Generative Adversarial Networks**

*Wrocław University of Science and Technology × Yunnan University — ICDM 2026*

---

## What It Is

PATE-DSS-GAN is a differentially private GAN that generates high-resolution images with formal (ε, δ)-DP guarantees. It combines two ideas:

- **PATE** (Private Aggregation of Teacher Ensembles) — privacy is enforced through noisy voting, not gradient perturbation, so the generator's learning signal is never corrupted.
- **DSS-GAN** — a Mamba (state-space model) based generator that captures long-range spatial dependencies, paired with a hybrid CNN-Mamba student discriminator.

---

## How It Works

```
Private Dataset
      │
      ▼
  k Stratified Shards
      │
      ▼
  k CNN Teacher Discriminators  (each trained on one private shard)
      │  ← fake images from G
      ▼
  GNMax Aggregation:  noisy_vote = Σ teacher_votes + N(0, σ²)
      │                            ↑ only step that touches private data
      │  RDP Accountant tracks ε budget
      ▼
  Student Discriminator  (CNN + Mamba)  trained on (fake_img, noisy_label)
      │
      ▼
  Generator G  (DSS-GAN / Mamba backbone)  updated against student
```

**Privacy guarantee:** Only the GNMax aggregation step interacts with private data. Everything downstream — student training and generator updates — is post-processing over already-released noisy labels, so it adds zero privacy cost.

---

## Architecture

### Generator (DSS-GAN)
- Mamba tokeniser maps latent `z` + class `c` into spatial tokens at 8×8
- Directional Latent Routing (DLR) blocks upsample: 8×8 → 16×16 → 32×32 → 64×64 → 128×128
- StyleGAN2 refinement head recovers fine detail at the final resolution

### Student Discriminator (Hybrid CNN-Mamba)
- CNN front-end downsamples input to 8×8 feature map
- Mamba blocks scan along row, column, and diagonal directions for global context
- Projection head produces a real/fake scalar logit

### Teacher Ensemble
- `k` lightweight CNN discriminators, each trained on a private data shard
- Retrained every 50 steps using current generator outputs as augmentation (privacy-safe: generator is public)
- Votes are aggregated via GNMax; noise σ is calibrated before training via binary search to meet the target ε

---

## Privacy Accounting

Noise σ is calibrated before training:

```
σ ← calibrate_sigma(ε, N, k, δ)
```

At each step, teachers vote on a fake batch → votes are noised → one binary label is released. The RDP accountant accumulates the per-step cost. Training stops when `accountant.get_epsilon() ≥ ε`.

---

## Datasets

| Dataset | Source | Resolution | Classes |
|---|---|---|---|
| CelebA-HQ | `korexyz/celeba-hq-256x256` | 128 / 256 | 2 (female / male) |
| AFHQ | `huggan/AFHQ` | 128 / 256 | 3 (cat / dog / wild) |

---

## Project Structure

```
PATE-DSS-GAN/
├── src/
│   ├── accountant.py           # GNMax RDP accountant + calibrate_sigma()
│   ├── config.py               # TrainConfig / EvalConfig dataclasses
│   ├── data/
│   │   ├── dataset.py          # CelebA-HQ / AFHQ loaders
│   │   └── partitioner.py      # Stratified k-shard partitioner
│   ├── models/
│   │   ├── teacher.py          # Lightweight CNN teachers
│   │   ├── student.py          # Hybrid CNN-Mamba student discriminator
│   │   └── generator.py        # DSS-GAN generator with DLR blocks
│   ├── pate/
│   │   ├── voting.py           # GNMax aggregation
│   │   └── ensemble.py         # Teacher ensemble orchestration
│   └── training/
│       ├── trainer.py          # Main training loop
│       └── evaluation.py       # FID, KID, Precision, Recall
├── configs/
│   ├── celeba_hq_128.yaml
│   ├── celeba_hq_256.yaml
│   └── afhq_256.yaml
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── run_afhq_epsilon_experiment.py   # AFHQ ε-milestone experiment
└── requirements.txt
```

---

## Quick Start

```bash
# AFHQ epsilon-milestone experiment (saves checkpoints at ε = 1, 2, 4, 8, 10)
cd PATE-DSS-GAN
python run_afhq_epsilon_experiment.py
```

Or via config:

```bash
python scripts/train.py --config configs/celeba_hq_256.yaml
```

---

## Citation

M. Youssef, Xin Jin, M. Woźniak. *PATE-DSS-GAN: Differentially Private High-Resolution Image Synthesis via State-Space Generative Adversarial Networks.* Technical Report, Wrocław University of Science and Technology & Yunnan University, 2026.

---

<p align="center">
  <sub>Wrocław University of Science and Technology &nbsp;×&nbsp; Yunnan University &nbsp;|&nbsp; ICDM 2026</sub>
</p>
