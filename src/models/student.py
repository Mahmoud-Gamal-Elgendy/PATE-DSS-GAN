"""
DSS-GAN Student Discriminator — faithful port for PATE-DSS-GAN.

Architecture (matches DSS_GAN-main/src/model/discriminator.py):
  StyleGAN2-ADA residual CNN with AvgPool2d downsampling + MiniBatchStdDev.
  Class conditioning via the projection discriminator (Miyato & Koyama, 2018):
    score = fc(h) + (embed(y) · h) / sqrt(dim)

In the PATE framework this is the *student* discriminator:
  - Trained on noisy PATE labels (post-processing, no extra ε).
  - Class label y is available at training time (generated with known c).
  - When c is None (legacy interface), returns the unconditional fc score.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Mini-batch standard deviation (StyleGAN2 diversity statistic)
# ---------------------------------------------------------------------------

class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4, num_features: int = 1) -> None:
        super().__init__()
        self.group_size  = group_size
        self.num_features = num_features

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        G  = min(self.group_size, B) if self.group_size is not None else B
        F_ = self.num_features

        y = x.reshape(G, -1, F_, C // F_, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = (y ** 2).mean(dim=0)
        y = (y + 1e-8).sqrt()
        y = y.mean(dim=[2, 3, 4], keepdim=True).squeeze(2)
        y = y.repeat(G, 1, H, W)
        return torch.cat([x, y], dim=1)


# ---------------------------------------------------------------------------
# StyleGAN2-ADA residual discriminator block
# ---------------------------------------------------------------------------

class DiscriminatorBlock(nn.Module):
    """
    Residual block with optional AvgPool2d downsampling (StyleGAN2-ADA style).
    Skip connection uses a 1×1 conv aligned with the same downsampling.
    """

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True) -> None:
        super().__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True)

        if downsample or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.skip = None

        self.pool = nn.AvgPool2d(2) if downsample else None

    def forward(self, x: Tensor) -> Tensor:
        y = F.leaky_relu(self.conv1(x), 0.2)
        y = F.leaky_relu(self.conv2(y), 0.2)
        if self.pool is not None:
            y = self.pool(y)

        skip = self.skip(x) if self.skip is not None else x
        if self.pool is not None:
            skip = self.pool(skip)

        return (y + skip) * (1.0 / math.sqrt(2))


# ---------------------------------------------------------------------------
# Full DSS-GAN Discriminator (StyleGAN2-ADA + projection conditioning)
# ---------------------------------------------------------------------------

class MambaStudentDiscriminator(nn.Module):
    """
    StyleGAN2-ADA discriminator used as the PATE student.

    Named MambaStudentDiscriminator for drop-in compatibility with the
    PATE-DSS-GAN trainer (which imports this name). The architecture is a
    pure CNN — the Mamba name is kept only for interface stability.

    Class conditioning (projection discriminator):
        score = fc(h) + dot(embed(y), h) / sqrt(dim)
    When ``c`` is None the projection term is omitted (unconditional score).

    Parameters
    ----------
    image_size : int
        Input resolution. Must be 128, 256, or 512.
    in_channels : int
        Input image channels (3 for RGB).
    base_channels : int
        Base channel count; doubles every stage up to channel_max.
    num_classes : int
        Number of output classes for projection conditioning.
    channel_max : int
        Maximum channel count at any stage.
    mbstd_group_size : int
        Mini-batch std-dev group size.
    **kwargs
        Accepts (and silently ignores) mamba_d_model, mamba_layers,
        scan_directions — kept so the trainer can pass them without error.
    """

    def __init__(
        self,
        image_size: int = 128,
        in_channels: int = 3,
        base_channels: int = 64,
        num_classes: int = 2,
        channel_max: int = 512,
        mbstd_group_size: int = 8,
        **kwargs,                  # absorbs unused mamba_* / scan_directions args
    ) -> None:
        super().__init__()
        assert image_size in (128, 256, 512), (
            f"image_size must be 128, 256, or 512; got {image_size}"
        )
        self.resolution  = image_size
        self.num_classes = num_classes

        def nf(stage: int) -> int:
            return min(base_channels * (2 ** stage), channel_max)

        # Build per-resolution channel table
        if image_size == 128:
            channels = {
                '128': nf(0), '64': nf(1), '32': nf(2),
                '16':  nf(3), '8':  nf(3), '4':  nf(3),
            }
        elif image_size == 256:
            channels = {
                '256': nf(0), '128': nf(1), '64': nf(2),
                '32':  nf(3), '16':  nf(3), '8':  nf(3), '4': nf(3),
            }
        else:   # 512
            channels = {
                '512': nf(0), '256': nf(1), '128': nf(2),
                '64':  nf(3), '32':  nf(3), '16':  nf(3),
                '8':   nf(3), '4':   nf(3),
            }

        self.from_rgb = nn.Conv2d(in_channels, channels[str(image_size)], 1, bias=True)

        # Resolution-specific top blocks
        if image_size == 512:
            self.block_512 = DiscriminatorBlock(channels['512'], channels['256'])
            self.block_256 = DiscriminatorBlock(channels['256'], channels['128'])
            self.block_128 = DiscriminatorBlock(channels['128'], channels['64'])
        elif image_size == 256:
            self.block_256 = DiscriminatorBlock(channels['256'], channels['128'])
            self.block_128 = DiscriminatorBlock(channels['128'], channels['64'])
        else:   # 128
            self.block_128 = DiscriminatorBlock(channels['128'], channels['64'])

        # Shared bottom blocks
        self.block_64 = DiscriminatorBlock(channels['64'], channels['32'])
        self.block_32 = DiscriminatorBlock(channels['32'], channels['16'])
        self.block_16 = DiscriminatorBlock(channels['16'], channels['8'])
        self.block_8  = DiscriminatorBlock(channels['8'],  channels['4'])

        # 4×4 head
        self.mbstd    = MiniBatchStdDev(group_size=mbstd_group_size, num_features=1)
        self.block_4  = nn.Conv2d(channels['4'] + 1, channels['4'], 3, padding=1, bias=True)
        self.conv_out = nn.Conv2d(channels['4'], channels['4'], 4, padding=0, bias=True)

        feat_dim = channels['4']
        self.fc    = nn.Linear(feat_dim, 1, bias=True)
        self.embed = nn.Embedding(num_classes, feat_dim)   # projection conditioning

        print(
            f"\n{'='*60}\n"
            f"DSS-GAN DISCRIMINATOR (PATE Student) — {image_size}×{image_size}\n"
            f"{'='*60}\n"
            f"  Channels: {' | '.join(f'{k}={v}' for k, v in channels.items())}\n"
            f"  Projection conditioning: num_classes={num_classes}\n"
            f"  mbstd group_size={mbstd_group_size}\n"
            f"{'='*60}\n"
        )

    def forward(self, x: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, 3, H, W)
        c : Tensor, optional  (B,) — integer class labels.
            When provided, adds projection conditioning to the score.
            When None, returns the unconditional score (backward-compatible).

        Returns
        -------
        Tensor  (B,) — scalar real/fake score per image.
        """
        h = F.leaky_relu(self.from_rgb(x), 0.2)

        if self.resolution == 512:
            h = self.block_512(h)
            h = self.block_256(h)
            h = self.block_128(h)
        elif self.resolution == 256:
            h = self.block_256(h)
            h = self.block_128(h)
        else:
            h = self.block_128(h)

        h = self.block_64(h)
        h = self.block_32(h)
        h = self.block_16(h)
        h = self.block_8(h)

        h = self.mbstd(h)
        h = F.leaky_relu(self.block_4(h),  0.2)
        h = F.leaky_relu(self.conv_out(h), 0.2)
        h = h.flatten(1)                               # (B, feat_dim)

        out = self.fc(h).squeeze(1)                    # (B,)

        if c is not None:
            # Projection discriminator: dot product between class embedding and features
            cond = (self.embed(c) * h).sum(dim=1) / math.sqrt(h.shape[1])
            out  = out + cond

        return out
