"""
Evaluation metrics for PATE-DSS-GAN.

Implements:
  - FID  (Fréchet Inception Distance)
  - KID  (Kernel Inception Distance)
  - Precision & Recall (Kynkäänniemi et al. 2019)
  - TSTR / TRTR  (Train on Synthetic/Real, Test on Real)
      Three CNN classifiers trained end-to-end on raw images:
        ResNet-18, EfficientNet-B0, MobileNetV3-Small
      Each is initialised from ImageNet weights, fine-tuned on the
      training set (synthetic for TSTR, real for TRTR), and tested on
      the held-out real test split.

All generative metrics use Inception-v3 features (torchvision).

Usage
-----
evaluator = Evaluator(device=device)

# Generative quality (FID, KID, Precision, Recall)
metrics = evaluator.evaluate(generator, test_loader, num_samples=10000)

# Downstream utility (TSTR vs TRTR)
tstr_metrics = evaluator.evaluate_tstr_trtr(
    generator, train_loader, test_loader,
    num_synth_samples=10000, num_classes=2,
)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import inception_v3, Inception_V3_Weights

try:
    from scipy.linalg import sqrtm as scipy_sqrtm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ImageNet normalisation constants (for the 3 CNN classifiers)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Inception feature extractor  (used for FID / KID / Precision / Recall)
# ---------------------------------------------------------------------------

class InceptionFeatureExtractor(nn.Module):
    """
    Extracts 2048-dim pool3 features from Inception-v3.
    Input images should be in [0, 1] range (NOT [-1, 1]).
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
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
# Statistical helpers  (FID / KID)
# ---------------------------------------------------------------------------

def _sqrtm(A: np.ndarray) -> np.ndarray:
    """Matrix square root of a symmetric PSD matrix."""
    if SCIPY_AVAILABLE:
        result = scipy_sqrtm(A)
        if np.iscomplexobj(result):
            result = result.real
        return result
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _compute_fid(mu_r: np.ndarray, sigma_r: np.ndarray,
                  mu_g: np.ndarray, sigma_g: np.ndarray) -> float:
    diff = mu_r - mu_g
    covmean = _sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean))


def _compute_kid(feats_r: np.ndarray, feats_g: np.ndarray,
                  num_subsets: int = 100, max_subset: int = 1000) -> tuple[float, float]:
    """Kernel Inception Distance (polynomial kernel MMD). Returns (mean, std)."""
    n = min(len(feats_r), len(feats_g), max_subset)
    rng = np.random.default_rng(0)
    kids = []
    for _ in range(num_subsets):
        idx_r = rng.choice(len(feats_r), n, replace=False)
        idx_g = rng.choice(len(feats_g), n, replace=False)
        r, g = feats_r[idx_r], feats_g[idx_g]
        d = r.shape[1]
        K_rr = (r @ r.T / d + 1) ** 3
        K_gg = (g @ g.T / d + 1) ** 3
        K_rg = (r @ g.T / d + 1) ** 3
        kids.append(K_rr.mean() + K_gg.mean() - 2 * K_rg.mean())
    return float(np.mean(kids)), float(np.std(kids))


def _compute_precision_recall(
    feats_r: np.ndarray,
    feats_g: np.ndarray,
    k: int = 3,
) -> tuple[float, float]:
    """Precision & Recall (Kynkäänniemi et al. 2019)."""
    from sklearn.neighbors import NearestNeighbors

    def _manifold_radius(feats: np.ndarray) -> np.ndarray:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(feats)
        dists, _ = nn.kneighbors(feats)
        return dists[:, -1]

    r_rad = _manifold_radius(feats_r)
    g_rad = _manifold_radius(feats_g)

    nn_r = NearestNeighbors(n_neighbors=1).fit(feats_r)
    nn_g = NearestNeighbors(n_neighbors=1).fit(feats_g)

    d_g_to_r, idx_g = nn_r.kneighbors(feats_g)
    precision = float((d_g_to_r[:, 0] <= r_rad[idx_g[:, 0]]).mean())

    d_r_to_g, idx_r = nn_g.kneighbors(feats_r)
    recall = float((d_r_to_g[:, 0] <= g_rad[idx_r[:, 0]]).mean())

    return precision, recall


# ---------------------------------------------------------------------------
# CNN classifier helpers  (TSTR / TRTR)
# ---------------------------------------------------------------------------

def _imagenet_normalize(x: Tensor) -> Tensor:
    """
    Convert images from the generator's [-1, 1] range to ImageNet-normalised
    [~N(0,1)] expected by torchvision pretrained CNNs.
    """
    x = x * 0.5 + 0.5                                          # → [0, 1]
    x = x.clamp(0.0, 1.0)
    mean = _IMAGENET_MEAN.to(x.device)
    std  = _IMAGENET_STD.to(x.device)
    return (x - mean) / std


def _build_cnn_classifiers(num_classes: int) -> Dict[str, nn.Module]:
    """
    Return three fresh CNN classifiers, each initialised from ImageNet weights
    with its final classification head replaced to output ``num_classes`` logits.

    Models
    ------
    ResNet-18        — lightweight universal baseline
    EfficientNet-B0  — best accuracy / compute tradeoff
    MobileNetV3-Small — different architectural inductive bias, fast
    """
    from torchvision.models import (
        resnet18, ResNet18_Weights,
        efficientnet_b0, EfficientNet_B0_Weights,
        mobilenet_v3_small, MobileNet_V3_Small_Weights,
    )

    # ResNet-18: replace final FC
    m_r = resnet18(weights=ResNet18_Weights.DEFAULT)
    m_r.fc = nn.Linear(m_r.fc.in_features, num_classes)

    # EfficientNet-B0: replace classifier[1]
    m_e = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    m_e.classifier[1] = nn.Linear(m_e.classifier[1].in_features, num_classes)

    # MobileNetV3-Small: replace classifier[3]
    m_m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    m_m.classifier[3] = nn.Linear(m_m.classifier[3].in_features, num_classes)

    return {
        "ResNet18":         m_r,
        "EfficientNet_B0":  m_e,
        "MobileNetV3_Small": m_m,
    }


def _train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
) -> nn.Module:
    """
    Fine-tune ``model`` on ``train_loader`` for ``epochs`` epochs.

    Optimiser : AdamW  (weight_decay=1e-4)
    Schedule  : Cosine annealing over all epochs
    Images    : re-normalised to ImageNet stats and resized to 224×224
    """
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_loss, correct, n = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs   = _imagenet_normalize(imgs.to(device))
            imgs   = F.interpolate(imgs, size=(224, 224),
                                   mode="bilinear", align_corners=False)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                total_loss += loss.item() * len(imgs)
                correct    += (model(imgs).argmax(1) == labels).sum().item()
                n          += len(imgs)
        scheduler.step()
        if epoch % 10 == 0:
            print(f"    epoch {epoch:3d}/{epochs}  "
                  f"loss={total_loss/n:.4f}  acc={correct/n:.4f}")

    return model


@torch.no_grad()
def _eval_cnn(model: nn.Module,
              test_loader: DataLoader,
              device: torch.device) -> float:
    """Return top-1 accuracy of ``model`` on ``test_loader``."""
    model = model.to(device).eval()
    correct, n = 0, 0
    for imgs, labels in test_loader:
        imgs   = _imagenet_normalize(imgs.to(device))
        imgs   = F.interpolate(imgs, size=(224, 224),
                               mode="bilinear", align_corners=False)
        labels = labels.to(device)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        n += len(imgs)
    return correct / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Computes FID, KID, Precision, Recall and TSTR/TRTR for a generator.

    Parameters
    ----------
    device : torch.device
    batch_size : int
        Batch size used for Inception feature extraction and CNN training.
    """

    def __init__(self, device: torch.device, batch_size: int = 64) -> None:
        self.device = device
        self.batch_size = batch_size
        self.inception = InceptionFeatureExtractor(device)

    # ------------------------------------------------------------------
    # Feature extraction  (Inception — for FID / KID / PR)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_features(self, dataloader: DataLoader,
                           max_samples: int = 50000) -> np.ndarray:
        feats: List[np.ndarray] = []
        n = 0
        for imgs, *_ in dataloader:
            if n >= max_samples:
                break
            imgs = imgs.to(self.device)
            imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
            feats.append(self.inception(imgs).cpu().numpy())
            n += len(imgs)
        return np.concatenate(feats, axis=0)[:max_samples]

    @torch.no_grad()
    def _extract_gen_features(
        self,
        generator,
        num_samples: int,
        num_classes: int,
        latent_dim: int,
    ) -> np.ndarray:
        feats: List[np.ndarray] = []
        n = 0
        generator.eval()
        while n < num_samples:
            bs = min(self.batch_size, num_samples - n)
            z  = generator.sample_latent(bs, self.device)
            c  = torch.randint(0, num_classes, (bs,), device=self.device)
            imgs = generator(z, c)
            imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
            feats.append(self.inception(imgs).cpu().numpy())
            n += bs
        return np.concatenate(feats, axis=0)

    # ------------------------------------------------------------------
    # Synthetic DataLoader builder  (for TSTR CNN training)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _make_synth_dataloader(
        self,
        generator,
        num_samples: int,
        num_classes: int,
    ) -> DataLoader:
        """
        Generate ``num_samples`` synthetic images and wrap them in a
        shuffled DataLoader suitable for CNN training.

        Images are stored as CPU tensors in [-1, 1] (same range as
        the real train DataLoader) so the same ``_imagenet_normalize``
        preprocessing applies to both TSTR and TRTR.
        """
        imgs_list:   List[Tensor] = []
        labels_list: List[Tensor] = []
        n = 0
        generator.eval()
        while n < num_samples:
            bs = min(self.batch_size, num_samples - n)
            z  = generator.sample_latent(bs, self.device)
            c  = torch.randint(0, num_classes, (bs,), device=self.device)
            imgs = generator(z, c).cpu()
            imgs_list.append(imgs)
            labels_list.append(c.cpu())
            n += bs
        imgs_t   = torch.cat(imgs_list,   dim=0)[:num_samples]
        labels_t = torch.cat(labels_list, dim=0)[:num_samples]
        ds = TensorDataset(imgs_t, labels_t)
        return DataLoader(ds, batch_size=self.batch_size,
                          shuffle=True, drop_last=False)

    # ------------------------------------------------------------------
    # Generative quality: FID, KID, Precision, Recall
    # ------------------------------------------------------------------

    def evaluate(
        self,
        generator,
        real_dataloader: DataLoader,
        num_samples: int = 10000,
        num_classes: int = 2,
        compute_pr: bool = True,
    ) -> Dict[str, float]:
        """
        Run generative-quality evaluation: FID, KID, Precision, Recall.

        Parameters
        ----------
        generator : DSSGANGenerator
        real_dataloader : DataLoader
            Held-out test split used as the reference distribution.
        num_samples : int
            Number of generated samples.
        num_classes : int
        compute_pr : bool
            Whether to compute Precision/Recall (requires sklearn).

        Returns
        -------
        dict with keys: fid, kid_mean, kid_std, [precision, recall]
        """
        print("[Eval] Extracting real features...")
        feats_r = self._extract_features(real_dataloader, max_samples=num_samples)

        print(f"[Eval] Generating {num_samples} samples for feature extraction...")
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
                results["recall"]    = recall
            except ImportError:
                print("[Eval] sklearn not installed; skipping Precision/Recall.")

        print("\n[Eval] Generative Quality Results:")
        for k, v in results.items():
            print(f"  {k:15s}: {v:.4f}")

        return results

    # ------------------------------------------------------------------
    # Downstream utility: TSTR vs TRTR  (3 CNN classifiers)
    # ------------------------------------------------------------------

    def evaluate_tstr_trtr(
        self,
        generator,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        num_synth_samples: int = 10000,
        num_classes: int = 2,
        cnn_epochs: int = 30,
        cnn_lr: float = 1e-3,
    ) -> Dict[str, float]:
        """
        TSTR (Train on Synthetic, Test on Real) vs
        TRTR (Train on Real,      Test on Real).

        Three CNN classifiers are trained **end-to-end on raw images**
        (not on Inception features), each initialised from ImageNet
        pretrained weights:

          - ResNet-18
          - EfficientNet-B0
          - MobileNetV3-Small

        Both TSTR and TRTR are evaluated on the **same held-out real
        test split**, so the accuracy gap directly measures how much
        utility is lost when substituting real training data with
        synthetic data.

        Parameters
        ----------
        generator : DSSGANGenerator
        train_dataloader : DataLoader
            Real training split — used **only** for TRTR.
        test_dataloader : DataLoader
            Held-out real test split — shared evaluation target.
        num_synth_samples : int
            Number of synthetic images generated for the TSTR training set.
        num_classes : int
        cnn_epochs : int
            Fine-tuning epochs for each CNN (default 30).
        cnn_lr : float
            Initial learning rate for AdamW (default 1e-3).

        Returns
        -------
        dict
            Keys: ``trtr_{clf}``, ``tstr_{clf}`` for each of the 3 CNNs,
            plus ``trtr_mean_acc`` and ``tstr_mean_acc``.

        Example
        -------
        tstr = evaluator.evaluate_tstr_trtr(gen, train_loader, test_loader)
        for clf in ["ResNet18", "EfficientNet_B0", "MobileNetV3_Small"]:
            print(f"TRTR {clf}: {tstr['trtr_'+clf]:.4f}  "
                  f"TSTR {clf}: {tstr['tstr_'+clf]:.4f}")
        """
        clf_names = ["ResNet18", "EfficientNet_B0", "MobileNetV3_Small"]
        results: Dict[str, float] = {}

        # Synthetic DataLoader — generated once, reused for all 3 TSTR classifiers
        print(f"[TSTR/TRTR] Generating {num_synth_samples} synthetic images...")
        synth_loader = self._make_synth_dataloader(
            generator, num_synth_samples, num_classes
        )

        for clf_name, model in _build_cnn_classifiers(num_classes).items():
            print(f"\n[TSTR/TRTR] ── {clf_name} ──")

            # TRTR: train on real
            print(f"  [TRTR] Fine-tuning on real training data...")
            model_trtr = _train_cnn(model, train_dataloader,
                                     self.device, cnn_epochs, cnn_lr)
            trtr_acc = _eval_cnn(model_trtr, test_dataloader, self.device)
            results[f"trtr_{clf_name}"] = trtr_acc
            print(f"  [TRTR] Test accuracy: {trtr_acc:.4f}")

            # TSTR: train on synthetic  (fresh copy of the same init)
            print(f"  [TSTR] Fine-tuning on synthetic data...")
            _, fresh_model = list(_build_cnn_classifiers(num_classes).items())[
                clf_names.index(clf_name)
            ]
            model_tstr = _train_cnn(fresh_model, synth_loader,
                                     self.device, cnn_epochs, cnn_lr)
            tstr_acc = _eval_cnn(model_tstr, test_dataloader, self.device)
            results[f"tstr_{clf_name}"] = tstr_acc
            print(f"  [TSTR] Test accuracy: {tstr_acc:.4f}")

        results["trtr_mean_acc"] = float(
            np.mean([results[f"trtr_{c}"] for c in clf_names])
        )
        results["tstr_mean_acc"] = float(
            np.mean([results[f"tstr_{c}"] for c in clf_names])
        )

        print("\n[TSTR/TRTR] Summary:")
        print(f"  {'Classifier':22s}  {'TRTR':>8}  {'TSTR':>8}  {'Gap (TSTR−TRTR)':>16}")
        print(f"  {'-'*60}")
        for clf in clf_names:
            trtr_acc = results[f"trtr_{clf}"]
            tstr_acc = results[f"tstr_{clf}"]
            print(f"  {clf:22s}  {trtr_acc:8.4f}  {tstr_acc:8.4f}  {tstr_acc - trtr_acc:+16.4f}")
        print(f"  {'Mean':22s}  {results['trtr_mean_acc']:8.4f}  "
              f"{results['tstr_mean_acc']:8.4f}  "
              f"{results['tstr_mean_acc'] - results['trtr_mean_acc']:+16.4f}")

        return results

    # ------------------------------------------------------------------
    # Privacy-utility curve (FID / KID vs epsilon)
    # ------------------------------------------------------------------

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

        Parameters
        ----------
        checkpoint_paths : list of str
            Paths to saved checkpoints, one per ε value.
        real_dataloader : DataLoader
            Held-out TEST split dataloader.
        generator_factory : callable
            ``generator_factory(ckpt) -> DSSGANGenerator``.
        num_classes : int
        num_samples : int

        Returns
        -------
        dict with keys: 'epsilon', 'fid', 'kid_mean', 'kid_std'
        """
        results: Dict[str, List[float]] = {
            "epsilon": [], "fid": [], "kid_mean": [], "kid_std": []
        }

        print(f"[Eval] Privacy-utility curve over {len(checkpoint_paths)} checkpoints...")
        print("[Eval] Extracting real features from test split...")
        feats_r  = self._extract_features(real_dataloader, max_samples=num_samples)
        mu_r     = feats_r.mean(0)
        sigma_r  = np.cov(feats_r, rowvar=False)

        for ckpt_path in checkpoint_paths:
            print(f"\n[Eval] Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            eps  = float(ckpt.get("epsilon", float("nan")))
            print(f"[Eval]   ε = {eps:.4f}")

            generator = generator_factory(ckpt).to(self.device)
            generator.eval()

            feats_g = self._extract_gen_features(
                generator, num_samples, num_classes, generator.latent_dim
            )
            fid          = _compute_fid(mu_r, sigma_r,
                                        feats_g.mean(0), np.cov(feats_g, rowvar=False))
            kid_mean, kid_std = _compute_kid(feats_r, feats_g)

            results["epsilon"].append(eps)
            results["fid"].append(fid)
            results["kid_mean"].append(kid_mean)
            results["kid_std"].append(kid_std)
            print(f"[Eval]   FID={fid:.2f}  KID={kid_mean:.4f}±{kid_std:.4f}")

        order = sorted(range(len(results["epsilon"])),
                       key=lambda i: results["epsilon"][i])
        for key in results:
            results[key] = [results[key][i] for i in order]

        return results
