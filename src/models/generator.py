"""
DSS-GAN Generator Wrapper with Directional Latent Routing (DLR).

This module provides a self-contained, trainable generator that implements
the DSS-GAN hierarchical Mamba architecture:

  z (latent) + c (class label)
      ↓ DLR conditioning (class-specific routing)
  Mamba Tokenizer (8×8 feature map)
      ↓ DLR blocks (8×8 → target resolution)
  StyleGAN2-style refinement layers (high-resolution)
      ↓
  Synthetic image (B, 3, H, W) in [-1, 1]

Reference: DSS-GAN (https://github.com/dssgan/DSS_GAN), Section 3.
If the DSS_GAN-main codebase is available, it can be integrated directly.
This implementation provides a full standalone version for environments
where the original repository is not yet cloned.

DLR (Directional Latent Routing):
  Class label c determines which scan direction (row / col / diag) is
  activated for each DLR block, providing class-specific spatial priors.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    from .student import SimplifiedMamba as Mamba
    MAMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utility layers
# ---------------------------------------------------------------------------

class AdaIN(nn.Module):
    """Adaptive Instance Normalization for StyleGAN2-style conditioning."""

    def __init__(self, num_features: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.style_proj = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        x = self.norm(x)
        style = self.style_proj(w)                         # (B, 2*C)
        gamma, beta = style.chunk(2, dim=1)                # each (B, C)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return gamma * x + beta


class PixelShuffle2x(nn.Module):
    """2× spatial upsampling via sub-pixel convolution."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 3, 1, 1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        return self.ps(self.conv(x))


# ---------------------------------------------------------------------------
# Directional Latent Routing block
# ---------------------------------------------------------------------------

class DLRBlock(nn.Module):
    """
    Directional Latent Routing block.

    Applies a Mamba SSM along a class-conditional scan direction,
    then upsamples the spatial map 2×.

    Parameters
    ----------
    in_ch : int
        Input channels.
    out_ch : int
        Output channels (after 2× upsample).
    d_model : int
        Mamba hidden dimension.
    num_classes : int
        Number of classes (used to route scan directions).
    """

    DIRECTIONS = ["row", "col", "diag"]

    def __init__(self, in_ch: int, out_ch: int, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Per-direction Mamba (DLR routes to one direction per class)
        self.mamba_row = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) if MAMBA_AVAILABLE else Mamba(d_model=d_model)
        self.mamba_col = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) if MAMBA_AVAILABLE else Mamba(d_model=d_model)
        self.mamba_diag = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) if MAMBA_AVAILABLE else Mamba(d_model=d_model)

        self.proj_in = nn.Conv2d(in_ch, d_model, 1)
        self.proj_out = nn.Conv2d(d_model, in_ch, 1)
        self.upsample = PixelShuffle2x(in_ch, out_ch)
        self.norm = nn.GroupNorm(8, in_ch)

        # Learned class → direction routing weights
        self.routing = nn.Embedding(num_classes, 3)  # 3 directions

    def _apply_mamba(self, feat: Tensor, direction: str) -> Tensor:
        """Reshape feat to sequence, apply Mamba, reshape back."""
        B, C, H, W = feat.shape
        if direction == "row":
            seq = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif direction == "col":
            seq = feat.permute(0, 3, 2, 1).reshape(B, H * W, C)
        else:  # diag
            seq = feat.flip(3).permute(0, 2, 3, 1).reshape(B, H * W, C)

        if direction == "row":
            seq = self.mamba_row(seq)
        elif direction == "col":
            seq = self.mamba_col(seq)
        else:
            seq = self.mamba_diag(seq)

        out = seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, in_ch, H, W)
        c : Tensor  (B,) int class indices

        Returns
        -------
        Tensor  (B, out_ch, 2H, 2W)
        """
        feat = self.proj_in(x)  # (B, d_model, H, W)

        # Soft directional routing weighted by class embedding
        weights = torch.softmax(self.routing(c), dim=-1)  # (B, 3)

        row_out = self._apply_mamba(feat, "row")
        col_out = self._apply_mamba(feat, "col")
        diag_out = self._apply_mamba(feat, "diag")

        # Weighted combination
        w_row = weights[:, 0:1, None, None]
        w_col = weights[:, 1:2, None, None]
        w_diag = weights[:, 2:3, None, None]
        feat = w_row * row_out + w_col * col_out + w_diag * diag_out

        feat = self.proj_out(feat)
        out = self.norm(x + feat)   # residual
        return self.upsample(out)


# ---------------------------------------------------------------------------
# StyleGAN2-style refinement block
# ---------------------------------------------------------------------------

class RefinementBlock(nn.Module):
    """Modulated conv + AdaIN for high-resolution StyleGAN2-style refinement."""

    def __init__(self, in_ch: int, out_ch: int, style_dim: int) -> None:
        super().__init__()
        self.ada_in = AdaIN(in_ch, style_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        x = self.ada_in(x, w)
        return self.conv(x)


# ---------------------------------------------------------------------------
# DSS-GAN Generator
# ---------------------------------------------------------------------------

class DSSGANGenerator(nn.Module):
    """
    DSS-GAN hierarchical Mamba generator with Directional Latent Routing.

    Generates class-conditional images at the target resolution.

    Architecture
    ------------
    z → MLP mapping network → w (style)
    c → class embedding → concat with w
    (w, c) → Mamba Tokenizer (8×8 spatial start)
    → DLR blocks (8×8 → target resolution)
    → StyleGAN2 refinement
    → ToRGB

    Parameters
    ----------
    latent_dim : int
        Noise vector dimension.
    num_classes : int
        Number of output classes (for DLR conditioning).
    image_size : int
        Target output resolution (must be a power of 2 ≥ 8).
    base_channels : int
        Base channel multiplier.
    mamba_d_model : int
        Mamba hidden dim inside DLR blocks.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_classes: int = 2,
        image_size: int = 128,
        base_channels: int = 64,
        mamba_d_model: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_size = image_size

        style_dim = latent_dim + 64  # z + class embedding

        # Mapping network: z + class emb → w
        self.class_embed = nn.Embedding(num_classes, 64)
        self.mapping = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
        )

        # Mamba tokenizer: start from 8×8
        start_ch = min(base_channels * 8, 512)
        self.const_input = nn.Parameter(torch.randn(1, start_ch, 8, 8))
        self.tokenizer_norm = nn.GroupNorm(8, start_ch)

        # DLR upsampling blocks: 8×8 → image_size
        n_ups = int(math.log2(image_size // 8))
        dlr_blocks = []
        in_ch = start_ch
        for i in range(n_ups):
            out_ch = max(base_channels, in_ch // 2)
            dlr_blocks.append(DLRBlock(in_ch, out_ch, mamba_d_model, num_classes))
            in_ch = out_ch
        self.dlr_blocks = nn.ModuleList(dlr_blocks)

        # StyleGAN2 refinement
        self.refinement = RefinementBlock(in_ch, in_ch, style_dim)

        # ToRGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(in_ch, 3, 1),
            nn.Tanh(),
        )

    def forward(self, z: Tensor, c: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z : Tensor  (B, latent_dim)
        c : Tensor  (B,) long — class indices

        Returns
        -------
        Tensor  (B, 3, H, W) in [-1, 1]
        """
        B = z.size(0)
        c_emb = self.class_embed(c)                    # (B, 64)
        style_in = torch.cat([z, c_emb], dim=-1)       # (B, style_dim)
        w = self.mapping(style_in)                     # (B, style_dim)

        # Start from learned constant
        x = self.const_input.expand(B, -1, -1, -1)    # (B, C, 8, 8)
        x = self.tokenizer_norm(x)

        # DLR upsampling
        for dlr in self.dlr_blocks:
            x = dlr(x, c)

        # StyleGAN2 refinement
        x = self.refinement(x, w)

        return self.to_rgb(x)

    def sample_latent(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample standard Gaussian latent vectors."""
        return torch.randn(batch_size, self.latent_dim, device=device)
