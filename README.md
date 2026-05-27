# PATE-DSS-GAN

### Differentially Private High-Resolution Image Synthesis via State-Space Generative Adversarial Networks

> **The first differentially private GAN built entirely on state-space models** — combining the PATE privacy framework with a Mamba-backbone DSS-GAN generator and a novel hybrid CNN-Mamba student discriminator, delivering formal $(\varepsilon, \delta)$-DP guarantees for high-resolution image synthesis.

**Research Collaboration:** Wrocław University of Science and Technology × Yunnan University

---

## Table of Contents

- [Abstract](#abstract)
- [Key Contributions](#key-contributions)
- [Background: PATE-TabTransGAN](#background-pate-tabtransgan)
- [Proposed Architecture](#proposed-architecture)
  - [System Overview](#system-overview)
  - [Generator: DSS-GAN with Mamba Backbone](#generator-dss-gan-with-mamba-backbone)
  - [Student Discriminator: Hybrid CNN-Mamba](#student-discriminator-hybrid-cnn-mamba)
  - [Teacher Ensemble](#teacher-ensemble)
- [Theoretical Framework & Privacy Accounting](#theoretical-framework--privacy-accounting)
- [Training Algorithm](#training-algorithm)
- [Project Structure](#project-structure)
- [Datasets & Baselines](#datasets--baselines)
- [Current Status & Collaboration](#current-status--collaboration)
- [Citation](#citation)

---

## Abstract

Differentially private image synthesis with Generative Adversarial Networks faces a fundamental tension: gradient-noise injection — the standard mechanism for enforcing DP in GAN training — directly corrupts the generator's learning signal, causing severe degradation in sample quality, particularly at high resolutions. Existing approaches that apply DP-SGD to the discriminator are architecturally constrained by the locality of convolutional receptive fields and suffer from an unfavorable privacy-utility trade-off as resolution scales.

**PATE-DSS-GAN** addresses these limitations through a principled combination of two complementary advances. First, we adopt the **Private Aggregation of Teacher Ensembles (PATE)** framework, which entirely avoids gradient perturbation by releasing only post-processed, noisy binary labels to the student discriminator — preserving the integrity of the generator's training signal while enforcing formal privacy guarantees through the GNMax aggregation mechanism. Second, we replace the conventional CNN discriminator with a **hybrid CNN-Mamba architecture** that exploits the long-range sequence modeling capabilities of selective state-space models (Mamba/SSM), enabling efficient directional spatial scanning over high-resolution feature maps. The generator follows the **DSS-GAN** design: a Mamba tokeniser with Directional Latent Routing (DLR) blocks for class-conditional hierarchical upsampling, refined by a StyleGAN2 head for high-frequency detail at $256 \times 256$ resolution.

Formal $(\varepsilon, \delta)$-DP guarantees are inherited through post-processing from the GNMax aggregation step, tracked via Rényi Differential Privacy (RDP) composition, and calibrated end-to-end via a binary-search noise function ported from our prior PATE-TabTransGAN GNMax RDP accountant.

---

## Key Contributions

- **First fully SSM-based DP-GAN.** PATE-DSS-GAN is the first differentially private GAN in which both the generator and the discriminator are built on state-space models (Mamba), eliminating the inductive locality bias of convolutional DP-GANs.

- **Gradient-noise-free privacy enforcement.** By adopting PATE rather than DP-SGD, the generator's learning signal is never perturbed. Privacy cost is charged exclusively at the teacher-ensemble aggregation step, not during backpropagation.

- **Hybrid CNN-Mamba student discriminator.** A CNN front-end efficiently downsamples $256 \times 256$ inputs to an $8 \times 8$ bottleneck; subsequent Mamba blocks perform directional spatial scanning (row, column, and diagonal trajectories), capturing global structural dependencies that standard CNNs miss.

- **Direct reuse of GNMax RDP accountant.** The `calibrate_sigma(ε, N, k, δ)` binary-search calibration function and the step-wise RDP composition logic are ported directly from PATE-TabTransGAN, providing a verified, battle-tested privacy accounting stack.

- **Privacy-safe teacher refresh.** Teachers are periodically retrained on their private shards using current generator outputs as additional signal. Because the generator is public and the shards are fixed private partitions, this incurs no additional privacy cost.

- **Formal end-to-end DP guarantee.** The entire pipeline satisfies $(\varepsilon, \delta)$-DP by construction: shard partitioning is post-processing, teacher training is local, GNMax charges the budget, and all downstream operations (student training, generator update) are post-processing over already-released labels.

---

## Background: PATE-TabTransGAN

PATE-DSS-GAN builds directly on our prior work **PATE-TabTransGAN**, which established the PATE-GAN paradigm for high-fidelity synthetic *tabular* data generation.

In PATE-TabTransGAN, an ensemble of Logistic Regression teachers — each trained on a disjoint partition of the private dataset — supervised a Transformer-based student discriminator via noisy-aggregated labels using the GNMax mechanism. A residual generator was optimised adversarially against this differentially private student, inheriting formal $(\varepsilon, \delta)$-DP guarantees through the post-processing immunity property of DP.

Evaluated on four tabular benchmarks against PATE-GAN, DP-GAN, and DP-CTGAN under matched privacy budgets, **PATE-TabTransGAN achieved the best or tied-best AUROC on all four datasets**, validating both the PATE-GAN training paradigm and the RDP accounting methodology.

PATE-DSS-GAN extends this validated framework to the image domain, replacing tabular Transformer components with Mamba state-space blocks while preserving the exact same privacy accounting stack.

---

## Proposed Architecture

### System Overview

```
Private Dataset D
        │
        ▼
  Stratified k-shard Partitioner
        │
   D_1 ... D_k
        │
        ▼
  CNN Teacher Discriminators T_1, ..., T_k
  (each trained on one private shard)
        │
        │   ← synthetic batch x̃ from G
        ▼
  GNMax Aggregation
  votes v_i ∈ {0,1}  →  n_real + N(0, σ²)  →  noisy label ŷ
        │
        │   RDP Accountant: record_query(v_1,...,v_k)
        ▼
  Student Discriminator D_S  (hybrid CNN-Mamba)
  trained on (x̃, ŷ)  — post-processing, privacy-free
        │
        ▼
  Generator G  (DSS-GAN / Mamba backbone)
  updated adversarially against D_S
```

The key insight is that the only step that touches private data after partitioning is the GNMax vote release. Everything downstream — student training and generator updates — operates exclusively on publicly released noisy labels and synthetic images, and therefore inherits the privacy guarantee at no additional cost.

---

### Generator: DSS-GAN with Mamba Backbone

The generator follows the **DSS-GAN** architecture, a hierarchical Mamba-based image synthesis network designed for class-conditional high-resolution generation.

**Components:**

- **Mamba Tokeniser.** A learned projection maps a latent vector $z \sim \mathcal{N}(0, I)$ and class embedding $c$ into a sequence of spatial tokens that serve as the initial state for the SSM backbone.

- **Directional Latent Routing (DLR) Blocks.** DLR blocks perform class-conditional hierarchical upsampling through selective state-space transitions. Each block routes latent information along multiple spatial directions (row-wise, column-wise, diagonal), enabling the generator to model both local texture and global structure coherently. Resolution is doubled at each DLR stage: $8 \times 8 \rightarrow 16 \times 16 \rightarrow 32 \times 32 \rightarrow 64 \times 64 \rightarrow 128 \times 128 \rightarrow 256 \times 256$.

- **StyleGAN2 Refinement Head.** The final stage applies a StyleGAN2-style modulated convolution head to recover fine-grained high-frequency detail at $256 \times 256$ resolution, compensating for the inherent smoothness of SSM-decoded sequences.

**Forward pass summary:**

$$G(z, c, z_{\text{dir}}) : \mathbb{R}^{d_z} \times \mathcal{C} \times \mathbb{R}^{d_{\text{dir}}} \rightarrow [0,1]^{3 \times 256 \times 256}$$

where $z_{\text{dir}}$ is a directional routing variable that conditions the DLR blocks on spatial trajectory preferences.

---

### Student Discriminator: Hybrid CNN-Mamba

The student discriminator $D_S$ is a novel hybrid architecture designed to capture both local texture (via CNN) and long-range spatial dependencies (via Mamba) from synthetic images labeled by the teacher ensemble.

**Architecture:**

1. **CNN Front-End (Downsampling Encoder).** A stack of strided convolutional layers progressively downsamples the $256 \times 256$ input image to an $8 \times 8$ spatial feature map, extracting local edge and texture features efficiently.

2. **Mamba Directional Scanning Blocks.** The $8 \times 8$ feature map (flattened into a sequence of $64$ patch tokens) is processed by Mamba SSM blocks along three directional scan trajectories:
   - **Row-wise scan** — left-to-right horizontal context
   - **Column-wise scan** — top-to-bottom vertical context
   - **Diagonal scan** — diagonal long-range context

   Each direction produces an independent contextual representation. The three outputs are concatenated and projected.

3. **Global Average Pooling → Scalar Logit.** A global average pooling layer followed by a linear projection produces a single real/fake scalar logit $D_S(\tilde{x}) \in \mathbb{R}$.

**Why Mamba for discrimination?** Standard CNN discriminators are inherently limited by their local receptive fields — detecting global image coherence (e.g., consistent lighting, structural symmetry) requires very deep stacks. Mamba's selective state-space mechanism models arbitrarily long-range dependencies in linear time, making it well-suited to assessing global realism in high-resolution synthetic images.

---

### Teacher Ensemble

- **$k$ CNN Teacher Discriminators**, each independently trained on a disjoint stratified shard $\mathcal{D}_i \subset \mathcal{D}$.
- Teachers are **lightweight CNNs** (not Mamba-based) — privacy cost scales with the number of teacher queries, not teacher complexity.
- Shards are created via **stratified partitioning by class label**, ensuring balanced class representation across all $k$ teachers.
- Teachers are **periodically refreshed** (every 50 generator iterations) by retraining on their private shard augmented with current generator outputs. This is privacy-safe because the generator is public and the shard boundaries are fixed.

---

## Theoretical Framework & Privacy Accounting

### Formal DP Guarantee

The full pipeline satisfies **$(\varepsilon, \delta)$-Differential Privacy** with respect to the private training dataset $\mathcal{D}$.

> **Definition ($(\varepsilon,\delta)$-DP).** A randomised mechanism $\mathcal{M}$ satisfies $(\varepsilon, \delta)$-DP if for all adjacent datasets $\mathcal{D}, \mathcal{D}'$ differing in one record, and all measurable output sets $S$:
> $$\Pr[\mathcal{M}(\mathcal{D}) \in S] \leq e^{\varepsilon} \cdot \Pr[\mathcal{M}(\mathcal{D}') \in S] + \delta$$

The guarantee follows from three structural properties:

1. **Disjoint sharding** is post-processing and does not consume privacy budget.
2. **GNMax aggregation** is the sole mechanism that operates on private data outputs. Each query releases one noisy binary label $\hat{y} = \mathbf{1}[\tilde{n} > k/2]$ where $\tilde{n} = n_{\text{real}} + \mathcal{N}(0, \sigma^2)$.
3. **Student training and generator updates** are functions of the released labels only — by the post-processing theorem of DP, they add zero privacy cost.

### GNMax Mechanism

At each training step, the teacher vote count for "real" is:

$$n_{\text{real}}(\tilde{x}) = \sum_{i=1}^{k} \mathbf{1}[T_i(\tilde{x}) = 1]$$

Gaussian noise is added to produce the noisy count:

$$\tilde{n} = n_{\text{real}} + \mathcal{N}(0, \sigma^2)$$

The released binary label is:

$$\hat{y} = \mathbf{1}\!\left[\tilde{n} > \frac{k}{2}\right]$$

This is the **Gaussian NoisyMax (GNMax)** mechanism. The $\ell_2$-sensitivity of $n_{\text{real}}$ is $1$ (a single teacher's vote can change by at most $1$ between adjacent datasets), so the noise scale $\sigma$ directly controls the per-query privacy expenditure.

### RDP Composition

Privacy cost is tracked via **Rényi Differential Privacy (RDP)**, which provides tight composition bounds. A mechanism satisfying $(\alpha, \rho)$-RDP for order $\alpha$ satisfies $(\rho + \frac{\log(1/\delta)}{\alpha - 1}, \delta)$-DP for any $\delta > 0$.

After $N$ queries each at noise level $\sigma$, the accumulated RDP budget at order $\alpha$ is:

$$\rho_{\text{total}}(\alpha) = N \cdot \rho_{\text{GNMax}}(\alpha, \sigma, k)$$

where $\rho_{\text{GNMax}}(\alpha, \sigma, k)$ is the per-step RDP cost of the GNMax mechanism, computed analytically from the Gaussian mechanism RDP formula with sensitivity $1/\sigma$.

The final $(\varepsilon, \delta)$-DP guarantee is obtained by converting the accumulated RDP:

$$\varepsilon = \min_{\alpha > 1} \left[ \rho_{\text{total}}(\alpha) + \frac{\log(1/\delta)}{\alpha - 1} \right]$$

### Noise Calibration

The noise parameter $\sigma$ is determined before training via binary search to exactly meet the target budget:

$$\sigma \leftarrow \texttt{calibrate\_sigma}(\varepsilon,\; N,\; k,\; \delta)$$

This function — ported directly from the verified PATE-TabTransGAN GNMax RDP accountant — performs a binary search over $\sigma$ values, simulating $N$ steps of RDP composition at each candidate $\sigma$ and returning the smallest $\sigma$ such that the total privacy cost does not exceed $(\varepsilon, \delta)$.

Training halts when `accountant.get_epsilon() ≥ ε`, ensuring the privacy budget is never exceeded regardless of training duration.

---

## Training Algorithm

```
Algorithm: PATE-DSS-GAN Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Require:
  D          — private image dataset
  (ε, δ)     — target privacy budget
  k          — number of teacher discriminators
  N          — total query budget
  n_s        — number of student update steps per query

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1:  Partition D into k stratified shards D_1, ..., D_k
  2:  Train CNN teacher discriminators T_1, ..., T_k on respective shards
  3:  σ ← calibrate_sigma(ε, N, k, δ)            ▷ binary-search noise calibration
  4:  Initialise generator G (with DLR), student D_S, accountant A

  while A.get_epsilon() < ε do

    5:  Sample z ~ N(0,I), class c, directional routing z_dir
    6:  Generate fake batch  x̃ ~ G(z, c, z_dir)

    ── Teacher Voting ──────────────────────────────────────────
    7:  Collect votes: v_i ← T_i(x̃)  for i = 1, ..., k
    8:  Aggregate:    n_real ← Σ_i 1[v_i = 1]

    ── GNMax Mechanism (charges privacy budget) ────────────────
    9:  Perturb:      ñ ← n_real + N(0, σ²)
   10:  Noisy label:  ŷ ← 1[ñ > k/2]
   11:  A.record_query(v_1, ..., v_k)              ▷ RDP composition step

    ── Student Update (post-processing — free) ─────────────────
    for n_s steps do
   12:    Update D_S on (x̃, ŷ)
    end for

    ── Generator Update ────────────────────────────────────────
   13:  Update G adversarially against D_S

    ── Periodic Teacher Refresh (privacy-safe) ─────────────────
   14:  if iteration ≡ 0 (mod 50) then
   15:    Retrain T_1,...,T_k on D_i ∪ {G outputs}  ▷ public G + fixed shards
        end if

  end while

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Privacy correctness notes:**
- Lines 1–2: Partitioning and teacher training are local to each shard — no cross-shard information leakage.
- Lines 9–11: The *only* interaction with private information after training. Every release is metered by the RDP accountant.
- Lines 12–13: Operate solely on $(x̃, \hat{y})$ — synthetic images and a post-processed label. Zero additional privacy cost by the post-processing theorem.
- Line 15: Teacher refresh uses the *public* generator's outputs as augmentation. No new private information is released; existing shard boundaries are not crossed.

---

## Project Structure

```
PATE-DSS-GAN/
├── src/
│   ├── accountant.py           # GNMax RDP accountant + calibrate_sigma()
│   ├── data/
│   │   ├── dataset.py          # CelebA-HQ / AFHQ loaders
│   │   └── partitioner.py      # Stratified k-shard partitioner
│   ├── models/
│   │   ├── teacher.py          # Lightweight CNN teacher discriminators
│   │   ├── student.py          # Hybrid CNN-Mamba student discriminator
│   │   └── generator.py        # DSS-GAN generator with DLR blocks
│   ├── pate/
│   │   ├── voting.py           # GNMax vote aggregation
│   │   └── ensemble.py         # Teacher ensemble orchestration
│   └── training/
│       ├── trainer.py          # Main training loop (Algorithm 1)
│       └── evaluation.py       # FID, KID, Precision, Recall, D&C
├── configs/
│   ├── cifar10_32.yaml         # Fast prototyping
│   ├── celeba_hq_128.yaml      # Proof-of-concept
│   ├── celeba_hq_256.yaml      # Primary experiment
│   └── afhq_256.yaml           # Secondary experiment
├── scripts/
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # FID / KID / Precision / Recall evaluation
│   ├── ablation.py             # Ablation studies
│   └── sigma_analysis.py       # σ calibration analysis
├── docs/
│   └── ARCHITECTURE.md         # Detailed component reference
└── requirements.txt
```

---

## Datasets & Baselines

### Datasets

| Dataset | Source | Resolution | Classes | Size | Role |
|---|---|---|---|---|---|
| CelebA-HQ | `korexyz/celeba-hq-256x256` (HuggingFace) | 256×256 | 2 (female / male) | 30K | Primary experiment |
| AFHQ | `huggan/AFHQ` (HuggingFace) | 256×256 | 3 (cat / dog / wild) | 15K | Secondary experiment |
| CIFAR-10 | standard | 32×32 | 10 | 50K | Fast prototyping |

```python
from datasets import load_dataset

ds_celeba = load_dataset("korexyz/celeba-hq-256x256")
ds_afhq   = load_dataset("huggan/AFHQ")
```

### Comparison Baselines

| Method | Mechanism | Notes |
|---|---|---|
| PATE-GAN | PATE + CNN | Tabular-origin; CNN discriminator baseline |
| DP-GAN | DP-SGD on discriminator | Gradient-noise injection approach |
| DP-CTGAN | DP-SGD | Tabular; included for cross-domain context |
| SPTI | Text-guided DP synthesis | Compare FID at matched $\varepsilon$ |
| DP-Diffusion | DP diffusion model | Compare FID at matched $\varepsilon$ |

All comparisons are conducted at **matched privacy budgets** $(\varepsilon, \delta)$.

---

## Current Status & Collaboration

> **Status: Core architecture under active development and implementation.**

This repository serves as the official conceptual framework and architectural baseline for an ongoing research collaboration between:

- **Wrocław University of Science and Technology** (Wrocław, Poland)
- **Yunnan University** (Kunming, China)

The foundational components — the GNMax RDP accountant (ported from PATE-TabTransGAN), the stratified data partitioner, and the CNN teacher ensemble — are complete and verified. Implementation of the hybrid CNN-Mamba student discriminator and integration of the DSS-GAN generator with the PATE training loop are currently in progress as part of the joint collaboration.

The target venue for the full empirical results is **ICDM 2026 (June 5th, 2026)**.

---


**Plain text:**

M. Youssef, Xin Jin, M. Woźniak. *PATE-DSS-GAN: Differentially Private High-Resolution Image Synthesis via State-Space Generative Adversarial Networks.* Technical Report, Wrocław University of Science and Technology & Yunnan University, 2026.

---

<p align="center">
  <sub>Wrocław University of Science and Technology &nbsp;×&nbsp; Yunnan University &nbsp;|&nbsp; ICDM 2026</sub>
</p>
