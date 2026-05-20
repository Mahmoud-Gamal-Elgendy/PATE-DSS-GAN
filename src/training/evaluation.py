"""
Evaluation metrics for PATE-DSS-GAN.

Implements:
  - FID  (Fréchet Inception Distance)
  - KID  (Kernel Inception Distance)
  - Precision & Recall (Kynkäänniemi et al.)
  - Density & Coverage (Naeem et al.)

All metrics use Inception-v3 features by default (torchvision).
Clean-FID (pip install cleanfid) is used when available for more
accurate FID estimation on non-standard resolutions.

Usage
-----
evaluator = Evaluator(device=device)
metrics = evaluator.evaluate(generator, real_dataloader, num_samples=10000)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np

try:
    from scipy.linalg import sqrtm as scipy_sqrtm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from cleanfid import fid as cleanfid_module
    CLEANFID_AVAILABLE = True
except ImportError:
    CLEANFID_AVAILABLE = False


# ---------------------------------------------------------------------------
# Inception feature extractor
# ---------------------------------------------------------------------------

class InceptionFeatureExtractor(nn.Module):
    """
    Extracts 2048-dim pool3 features from Inception-v3.
    Input images should be in [0, 1] range (NOT [-1, 1]).
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        # Remove classification head; keep up to global average pool
        self.features = nn.Sequential(
            model.Conv2d_1a_3x3, model.Conv2d_2a_3x3, model.Conv2d_2b_3x3,
            nn.MaxPool2d(3, 2), model.Conv2d_3b_1x1, model.Conv2d_4a_3x3,
            nn.MaxPool2d(3, 2), model.Mixed_5b, model.Mixed_5c, model.Mixed_5d,
            model.Mixed_6a, model.Mixed_6b, model.Mixed_6c, model.Mixed_6d,
            model.Mixed_6e, model.Mixed_7a, model.Mixed_7b, model.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, 3, H, W) in [0, 1] → (B, 2048)."""
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return self.features(x).flatten(1)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _sqrtm(A: np.ndarray) -> np.ndarray:
    """
    Matrix square root of a symmetric PSD matrix.

    Uses scipy.linalg.sqrtm when available (handles near-singular matrices
    more robustly). Falls back to eigendecomposition for environments without
    scipy (e.g. minimal Docker images).
    """
    if SCIPY_AVAILABLE:
        result = scipy_sqrtm(A)
        if np.iscomplexobj(result):
            result = result.real
        return result
    # Eigendecomposition fallback (valid for symmetric PSD matrices)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _compute_fid(mu_r: np.ndarray, sigma_r: np.ndarray,
                  mu_g: np.ndarray, sigma_g: np.ndarray) -> float:
    """Compute FID between two Gaussians (real, generated)."""
    diff = mu_r - mu_g
    covmean = _sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))
    return fid


def _compute_kid(feats_r: np.ndarray, feats_g: np.ndarray,
                  num_subsets: int = 100, max_subset: int = 1000) -> tuple[float, float]:
    """
    Kernel Inception Distance (polynomial kernel MMD).
    Returns (mean, std) over bootstrap subsets.
    """
    n = min(len(feats_r), len(feats_g), max_subset)
    rng = np.random.default_rng(0)
    kids = []
    for _ in range(num_subsets):
        idx_r = rng.choice(len(feats_r), n, replace=False)
        idx_g = rng.choice(len(feats_g), n, replace=False)
        r, g = feats_r[idx_r], feats_g[idx_g]
        # Polynomial kernel: k(x, y) = (x·y/d + 1)³
        d = r.shape[1]
        K_rr = (r @ r.T / d + 1) ** 3
        K_gg = (g @ g.T / d + 1) ** 3
        K_rg = (r @ g.T / d + 1) ** 3
        kid = (K_rr.mean() + K_gg.mean() - 2 * K_rg.mean())
        kids.append(kid)
    return float(np.mean(kids)), float(np.std(kids))


def _compute_precision_recall(
    feats_r: np.ndarray,
    feats_g: np.ndarray,
    k: int = 3,
) -> tuple[float, float]:
    """
    Precision & Recall (Kynkäänniemi et al. 2019).
    Precision: fraction of fake samples in real manifold.
    Recall: fraction of real samples in fake manifold.
    """
    from sklearn.neighbors import NearestNeighbors

    def _manifold_radius(feats: np.ndarray) -> np.ndarray:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(feats)
        dists, _ = nn.kneighbors(feats)
        return dists[:, -1]  # distance to k-th nearest neighbour

    r_rad = _manifold_radius(feats_r)
    g_rad = _manifold_radius(feats_g)

    nn_r = NearestNeighbors(n_neighbors=1).fit(feats_r)
    nn_g = NearestNeighbors(n_neighbors=1).fit(feats_g)

    # Precision: for each fake, check if nearest real is within real's radius
    d_g_to_r, idx_g = nn_r.kneighbors(feats_g)
    precision = float((d_g_to_r[:, 0] <= r_rad[idx_g[:, 0]]).mean())

    # Recall: for each real, check if nearest fake is within fake's radius
    d_r_to_g, idx_r = nn_g.kneighbors(feats_r)
    recall = float((d_r_to_g[:, 0] <= g_rad[idx_r[:, 0]]).mean())

    return precision, recall


def _compute_density_coverage(
    feats_r: np.ndarray,
    feats_g: np.ndarray,
    k: int = 5,
) -> tuple[float, float]:
    """Density & Coverage (Naeem et al. 2020)."""
    from sklearn.neighbors import NearestNeighbors

    nn_r = NearestNeighbors(n_neighbors=k + 1).fit(feats_r)
    dists_r, _ = nn_r.kneighbors(feats_r)
    r_rad = dists_r[:, -1]

    # Density: average number of real neighbours within radius
    d_g_to_r, _ = nn_r.kneighbors(feats_g, n_neighbors=k)
    in_ball = (d_g_to_r <= r_rad[None, :k]).any(axis=1)  # simplified
    density = float(in_ball.mean())

    # Coverage: fraction of real samples with at least one fake neighbour in ball
    d_r_to_g, _ = NearestNeighbors(n_neighbors=1).fit(feats_g).kneighbors(feats_r)
    coverage = float((d_r_to_g[:, 0] <= r_rad).mean())

    return density, coverage


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Computes FID, KID, Precision, Recall, Density, Coverage for a generator.

    Parameters
    ----------
    device : torch.device
    batch_size : int
        Batch size for feature extraction.
    """

    def __init__(self, device: torch.device, batch_size: int = 64) -> None:
        self.device = device
        self.batch_size = batch_size
        self.inception = InceptionFeatureExtractor(device)

    @torch.no_grad()
    def _extract_features(self, dataloader: DataLoader, max_samples: int = 50000) -> np.ndarray:
        feats = []
        n = 0
        for imgs, *_ in dataloader:
            if n >= max_samples:
                break
            imgs = imgs.to(self.device)
            imgs = imgs * 0.5 + 0.5   # [-1,1] → [0,1]
            imgs = imgs.clamp(0, 1)
            f = self.inception(imgs).cpu().numpy()
            feats.append(f)
            n += len(f)
        return np.concatenate(feats, axis=0)[:max_samples]

    @torch.no_grad()
    def _extract_gen_features(
        self,
        generator,
        num_samples: int,
        num_classes: int,
        latent_dim: int,
    ) -> np.ndarray:
        feats = []
        n = 0
        generator.eval()
        while n < num_samples:
            bs = min(self.batch_size, num_samples - n)
            z = generator.sample_latent(bs, self.device)
            c = torch.randint(0, num_classes, (bs,), device=self.device)
            imgs = generator(z, c)
            imgs = imgs * 0.5 + 0.5
            imgs = imgs.clamp(0, 1)
            f = self.inception(imgs).cpu().numpy()
            feats.append(f)
            n += bs
        return np.concatenate(feats, axis=0)

    def evaluate(
        self,
        generator,
        real_dataloader: DataLoader,
        num_samples: int = 10000,
        num_classes: int = 2,
        compute_pr: bool = True,
        compute_dc: bool = True,
    ) -> Dict[str, float]:
        """
        Run full evaluation suite.

        Parameters
        ----------
        generator : DSSGANGenerator
        real_dataloader : DataLoader
            Real image loader for reference statistics.
        num_samples : int
            Number of generated samples to evaluate.
        num_classes : int
        compute_pr : bool
            Whether to compute Precision/Recall (requires sklearn).
        compute_dc : bool
            Whether to compute Density/Coverage (requires sklearn).

        Returns
        -------
        dict with keys: fid, kid_mean, kid_std, precision, recall, density, coverage
        """
        print("[Eval] Extracting real features...")
        feats_r = self._extract_features(real_dataloader, max_samples=num_samples)

        print(f"[Eval] Extracting generated features ({num_samples} samples)...")
        feats_g = self._extract_gen_features(
            generator, num_samples, num_classes, generator.latent_dim
        )

        print("[Eval] Computing FID...")
        mu_r, sigma_r = feats_r.mean(0), np.cov(feats_r, rowvar=False)
        mu_g, sigma_g = feats_g.mean(0), np.cov(feats_g, rowvar=False)
        fid = _compute_fid(mu_r, sigma_r, mu_g, sigma_g)

        print("[Eval] Computing KID...")
        kid_mean, kid_std = _compute_kid(feats_r, feats_g)

        results: Dict[str, float] = {
            "fid": fid,
            "kid_mean": kid_mean,
            "kid_std": kid_std,
        }

        if compute_pr:
            try:
                print("[Eval] Computing Precision/Recall...")
                precision, recall = _compute_precision_recall(feats_r, feats_g)
                results["precision"] = precision
                results["recall"] = recall
            except ImportError:
                print("[Eval] sklearn not installed; skipping Precision/Recall.")

        if compute_dc:
            try:
                print("[Eval] Computing Density/Coverage...")
                density, coverage = _compute_density_coverage(feats_r, feats_g)
                results["density"] = density
                results["coverage"] = coverage
            except ImportError:
                print("[Eval] sklearn not installed; skipping Density/Coverage.")

        print("\n[Eval] Results:")
        for k, v in results.items():
            print(f"  {k:15s}: {v:.4f}")

        return results

    def privacy_utility_curve(
        self,
        checkpoint_paths: List[str],
        real_dataloader: DataLoader,
        generator_factory,
        num_classes: int,
        num_samples: int = 5000,
    ) -> Dict[str, List[float]]:
        """
        Evaluate FID and KID across a set of per-ε checkpoints.

        Produces the privacy-utility curve for the paper: FID vs ε at
        different privacy budgets, using held-out real images for reference.

        Parameters
        ----------
        checkpoint_paths : list of str
            Paths to saved checkpoints, one per ε value. Each checkpoint must
            contain 'generator', 'epsilon', and 'sigma' keys (saved by trainer).
        real_dataloader : DataLoader
            Held-out TEST split dataloader (NOT the training set). This ensures
            honest FID reporting on unseen real images.
        generator_factory : callable
            `generator_factory(ckpt) -> DSSGANGenerator` — builds and loads a
            generator from a checkpoint dict. Example:
                def factory(ckpt):
                    g = DSSGANGenerator(...)
                    g.load_state_dict(ckpt['generator'])
                    return g
        num_classes : int
            Number of output classes (used for latent sampling).
        num_samples : int
            Number of generated samples per ε checkpoint.

        Returns
        -------
        dict with keys: 'epsilon', 'fid', 'kid_mean', 'kid_std'
            Each value is a list aligned with checkpoint_paths.

        Example
        -------
        curve = evaluator.privacy_utility_curve(
            checkpoint_paths=[
                'checkpoints/eps1/ckpt_final.pt',
                'checkpoints/eps5/ckpt_final.pt',
                'checkpoints/eps10/ckpt_final.pt',
            ],
            real_dataloader=test_loader,   # train=False split
            generator_factory=my_factory,
            num_classes=2,
            num_samples=5000,
        )
        # Plot: plt.plot(curve['epsilon'], curve['fid'])
        """
        import torch

        results: Dict[str, List[float]] = {
            "epsilon": [], "fid": [], "kid_mean": [], "kid_std": []
        }

        print(f"[Eval] Privacy-utility curve over {len(checkpoint_paths)} checkpoints...")
        print("[Eval] Using held-out test split for real reference features.")

        # Extract real features once (shared across all ε values)
        print("[Eval] Extracting real features from test split...")
        feats_r = self._extract_features(real_dataloader, max_samples=num_samples)
        mu_r = feats_r.mean(0)
        sigma_r = np.cov(feats_r, rowvar=False)

        for ckpt_path in checkpoint_paths:
            print(f"\n[Eval] Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            eps = float(ckpt.get("epsilon", float("nan")))
            print(f"[Eval]   ε = {eps:.4f}")

            generator = generator_factory(ckpt).to(self.device)
            generator.eval()

            feats_g = self._extract_gen_features(
                generator, num_samples, num_classes, generator.latent_dim
            )
            mu_g = feats_g.mean(0)
            sigma_g = np.cov(feats_g, rowvar=False)

            fid = _compute_fid(mu_r, sigma_r, mu_g, sigma_g)
            kid_mean, kid_std = _compute_kid(feats_r, feats_g)

            results["epsilon"].append(eps)
            results["fid"].append(fid)
            results["kid_mean"].append(kid_mean)
            results["kid_std"].append(kid_std)
            print(f"[Eval]   FID={fid:.2f}  KID={kid_mean:.4f}±{kid_std:.4f}")

        # Sort by ascending ε for clean plotting
        order = sorted(range(len(results["epsilon"])), key=lambda i: results["epsilon"][i])
        for key in results:
            results[key] = [results[key][i] for i in order]

        return results
