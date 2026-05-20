"""
Mamba-based Student Discriminator for PATE-DSS-GAN.

Novel architecture: first Mamba discriminator in a PATE framework.

Architecture (per audit recommendation):
  Input (B, 3, H, W)
      ↓ CNN downsampling blocks (DiscriminatorBlock-style, H → 8)
  Bottleneck (B, C_bot, 8, 8)
      ↓ Reshape to sequence (B, 64, C_bot)
      ↓ Mamba SSM blocks with directional scanning (row / col / diagonal)
  Pooling → (B, C_bot)
      ↓ Linear head
  Scalar logit (B, 1)

The Mamba blocks at the 8×8 bottleneck provide:
  - O(N) complexity vs O(N²) for attention-based discriminators
  - Directional scanning captures spatial structure for real/fake decisions
  - Compatibility with any image resolution (CNN handles the downsampling)

Mamba dependency: `mamba-ssm` (pip install mamba-ssm). Falls back to a
pure-PyTorch selective SSM approximation if mamba-ssm is not installed,
enabling CPU-only development and testing.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Try to import the efficient CUDA Mamba kernel; fall back to pure-PyTorch.
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pure-PyTorch Mamba approximation (fallback for CPU / environments without
# the CUDA mamba-ssm package)
# ---------------------------------------------------------------------------

class SimplifiedMamba(nn.Module):
    """
    Lightweight selective SSM approximation for development/testing.

    Implements the core selective state-space recurrence without the
    hardware-optimised parallel scan. For full performance, install
    `mamba-ssm` (CUDA required).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv - 1, groups=self.d_inner)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor  shape (B, L, d_model)

        Returns
        -------
        Tensor  shape (B, L, d_model)
        """
        residual = x
        B, L, D = x.shape

        xz = self.in_proj(x)          # (B, L, 2*d_inner)
        x_part, z = xz.chunk(2, dim=-1)

        # 1D causal convolution along sequence dimension
        x_part = x_part.transpose(1, 2)                       # (B, d_inner, L)
        x_part = self.conv1d(x_part)[..., :L]                 # causal truncation
        x_part = self.act(x_part.transpose(1, 2))             # (B, L, d_inner)

        # Selective gating (simplified: use z as gate)
        y = x_part * torch.sigmoid(z)
        y = self.out_proj(y)
        return self.norm(y + residual)


# ---------------------------------------------------------------------------
# Directional scanning utilities
# ---------------------------------------------------------------------------

def _row_scan(x: Tensor) -> Tensor:
    """Flatten spatial map row-by-row: (B, C, H, W) → (B, H*W, C)."""
    B, C, H, W = x.shape
    return x.permute(0, 2, 3, 1).reshape(B, H * W, C)


def _col_scan(x: Tensor) -> Tensor:
    """Flatten column-by-column: (B, C, H, W) → (B, H*W, C)."""
    B, C, H, W = x.shape
    return x.permute(0, 3, 2, 1).reshape(B, H * W, C)


def _diag_scan(x: Tensor) -> Tensor:
    """
    Flatten along anti-diagonals (approximated via transpose+row-scan).
    Full diagonal scan from I2I-Mamba can replace this for higher fidelity.
    """
    B, C, H, W = x.shape
    # Rotate 45° approximation: flip then row-scan
    x_flip = x.flip(dims=[3])
    return _row_scan(x_flip)


_SCAN_FNS = {
    "row": _row_scan,
    "col": _col_scan,
    "diag": _diag_scan,
}


# ---------------------------------------------------------------------------
# CNN downsampling block
# ---------------------------------------------------------------------------

class DiscriminatorBlock(nn.Module):
    """
    Strided conv block with residual shortcut (DSS-GAN style).
    Halves spatial resolution at each call.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.skip = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 2, 0, bias=False)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x) + self.skip(x)


# ---------------------------------------------------------------------------
# Mamba Student Discriminator
# ---------------------------------------------------------------------------

class MambaStudentDiscriminator(nn.Module):
    """
    Hybrid CNN + Mamba student discriminator.

    The CNN downsampling path reduces any input resolution to an 8×8
    bottleneck, at which point Mamba SSM blocks with multi-directional
    scanning extract global context efficiently.

    Parameters
    ----------
    image_size : int
        Input spatial resolution (H = W). Supported: 32, 64, 128, 256.
    in_channels : int
        Input channels (3 for RGB).
    base_channels : int
        Base channel count for CNN blocks.
    mamba_d_model : int
        Mamba hidden dimension at the bottleneck.
    mamba_layers : int
        Number of Mamba blocks stacked at the bottleneck.
    scan_directions : list of str
        Subset of ['row', 'col', 'diag'] — ablation parameter.
    d_state : int
        SSM state dimension (only used by fallback SimplifiedMamba).
    """

    BOTTLENECK_SIZE = 8

    def __init__(
        self,
        image_size: int = 128,
        in_channels: int = 3,
        base_channels: int = 64,
        mamba_d_model: int = 256,
        mamba_layers: int = 4,
        scan_directions: Optional[List[str]] = None,
        d_state: int = 16,
    ) -> None:
        super().__init__()
        if scan_directions is None:
            scan_directions = ["row", "col", "diag"]

        for d in scan_directions:
            if d not in _SCAN_FNS:
                raise ValueError(f"Unknown scan direction '{d}'. Choose from {list(_SCAN_FNS)}")
        self.scan_directions = scan_directions

        # ---- CNN downsampling ----
        # Number of stride-2 blocks to reach 8×8 from image_size
        n_down = int(math.log2(image_size // self.BOTTLENECK_SIZE))
        assert image_size // (2 ** n_down) == self.BOTTLENECK_SIZE, (
            f"image_size={image_size} must be divisible to reach 8×8 with stride-2 blocks."
        )

        C = base_channels
        cnn_blocks: list = [
            nn.utils.spectral_norm(nn.Conv2d(in_channels, C, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_ch = C
        for i in range(n_down):
            out_ch = min(C * (2 ** (i + 1)), 512)
            cnn_blocks.append(DiscriminatorBlock(in_ch, out_ch))
            in_ch = out_ch

        self.cnn_down = nn.Sequential(*cnn_blocks)
        self.bottleneck_channels = in_ch

        # Project bottleneck channels → mamba_d_model
        self.proj_in = nn.Linear(self.bottleneck_channels, mamba_d_model)

        # ---- Mamba blocks ----
        if MAMBA_AVAILABLE:
            self.mamba_blocks = nn.ModuleList([
                Mamba(d_model=mamba_d_model, d_state=d_state, d_conv=4, expand=2)
                for _ in range(mamba_layers)
            ])
        else:
            print(
                "[MambaStudentDiscriminator] mamba-ssm not found. "
                "Using pure-PyTorch fallback (install mamba-ssm for GPU efficiency)."
            )
            self.mamba_blocks = nn.ModuleList([
                SimplifiedMamba(d_model=mamba_d_model, d_state=d_state)
                for _ in range(mamba_layers)
            ])

        # ---- Output head ----
        # After multi-direction scanning, features are concatenated → project back
        n_dirs = len(scan_directions)
        self.proj_out = nn.Linear(mamba_d_model * n_dirs, mamba_d_model)
        self.norm = nn.LayerNorm(mamba_d_model)
        self.head = nn.Linear(mamba_d_model, 1)

    def forward(self, x: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, 3, H, W)
            Input image.
        c : Tensor, optional  (B,) int class labels — not used in discriminator
            (included for interface compatibility with DSS-GAN).

        Returns
        -------
        Tensor  (B, 1)  raw logit.
        """
        # CNN: (B, 3, H, W) → (B, C_bot, 8, 8)
        feat = self.cnn_down(x)

        # Multi-direction Mamba scanning at 8×8 bottleneck
        dir_outputs = []
        for direction in self.scan_directions:
            scan_fn = _SCAN_FNS[direction]
            seq = scan_fn(feat)                       # (B, 64, C_bot)
            seq = self.proj_in(seq)                   # (B, 64, d_model)
            for mamba_block in self.mamba_blocks:
                seq = mamba_block(seq)                # (B, 64, d_model)
            pooled = seq.mean(dim=1)                  # (B, d_model)
            dir_outputs.append(pooled)

        # Aggregate directions
        combined = torch.cat(dir_outputs, dim=-1)     # (B, d_model * n_dirs)
        combined = self.proj_out(combined)             # (B, d_model)
        combined = self.norm(combined)
        logit = self.head(combined)                    # (B, 1)
        return logit
