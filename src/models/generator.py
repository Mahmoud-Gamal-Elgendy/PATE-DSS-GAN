"""
DSS-GAN Generator — faithful self-contained port for PATE-DSS-GAN.

Architecture (matches DSS_GAN-main/src/model/generator.py):
  z split into z_tok (token generation) + z_dir (directional modulation)
  y (class label) conditions every spatial stage via:
    - Class embedding added to the 8×8 token map
    - Per-direction class embeddings concat-ed into z_dir chunks → γ/β affine
      modulation of each Mamba sequence (Hybrid DLR)
    - FiLM layers at each spatial scale (optionally enabled)
    - StyleGAN2Block modulated conv at the final resolution

Spatial pipeline:
  8×8 → 16×16 → 32×32 → 64×64   (Mamba DLR blocks, z+class conditioned)
  → 128×128                       (StyleGAN2Block if res=128; Mamba if res≥256)
  → 256×256                       (StyleGAN2Block if res=256; Mamba if res=512)
  → 512×512                       (StyleGAN2Block if res=512)
  → refine → final_conv → to_RGB → tanh
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Mamba import with pure-PyTorch fallback
# ---------------------------------------------------------------------------

try:
    from mamba_ssm import Mamba as _MambaSSM
    _MAMBA_AVAILABLE = True
except ImportError:
    _MambaSSM = None
    _MAMBA_AVAILABLE = False


class _SimplifiedMamba(nn.Module):
    """
    Pure-PyTorch selective SSM approximation.
    Drop-in replacement for mamba_ssm.Mamba on CPU / environments without CUDA.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        d_inner = int(d_model * expand)
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.act      = nn.SiLU()
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)
        x_part = x_part.transpose(1, 2)
        x_part = self.conv1d(x_part)[..., :L]
        x_part = self.act(x_part.transpose(1, 2))
        y = x_part * torch.sigmoid(z)
        y = self.out_proj(y)
        return self.norm(y + residual)


def _make_mamba(d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> nn.Module:
    if _MAMBA_AVAILABLE:
        return _MambaSSM(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    return _SimplifiedMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


# ---------------------------------------------------------------------------
# Conditioning modules
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    """Feature-wise Linear Modulation via class-conditional γ, β embeddings."""

    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.gamma = nn.Embedding(num_classes, num_features)
        self.beta  = nn.Embedding(num_classes, num_features)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        gamma = self.gamma(y).view(-1, self.num_features, 1, 1)
        beta  = self.beta(y).view(-1, self.num_features, 1, 1)
        return gamma * x + beta


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_tokens: int, dim: int) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos


# ---------------------------------------------------------------------------
# Mamba block: LayerNorm pre-norm + residual
# ---------------------------------------------------------------------------

class RealVisionMambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.mamba = _make_mamba(dim, d_state, d_conv, expand)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.mamba.named_parameters():
            if 'dt_proj' in name or 'x_proj' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.5)
                else:
                    nn.init.normal_(param, mean=0, std=0.02)
            elif 'conv1d' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.8)
            elif 'in_proj' in name or 'out_proj' in name:
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param, gain=0.8)
                elif param.dim() == 1:
                    nn.init.zeros_(param)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mamba(self.norm(x))


# ---------------------------------------------------------------------------
# Directional weighting: z-modulated + class-biased softmax over scan dirs
# ---------------------------------------------------------------------------

class DirectionalWeighting(nn.Module):
    def __init__(
        self,
        num_directions: int,
        z_routing_dim: Optional[int] = None,
        temperature: float = 2.0,
    ) -> None:
        super().__init__()
        self.base_weights     = nn.Parameter(torch.ones(num_directions))
        self.temperature      = temperature
        self.use_z_modulation = z_routing_dim is not None
        if self.use_z_modulation:
            self.z_to_weights = nn.Linear(z_routing_dim, num_directions)
            nn.init.zeros_(self.z_to_weights.weight)
            nn.init.zeros_(self.z_to_weights.bias)

    def forward(
        self,
        z_dir: Optional[Tensor] = None,
        class_routing_bias: Optional[Tensor] = None,
    ) -> Tensor:
        weights = self.base_weights
        if self.use_z_modulation and z_dir is not None:
            weights = weights.unsqueeze(0) + self.z_to_weights(z_dir)
        else:
            weights = weights.unsqueeze(0)
        if class_routing_bias is not None:
            weights = weights + class_routing_bias
        return F.softmax(weights / self.temperature, dim=-1)


# ---------------------------------------------------------------------------
# Lightweight spatial self-attention (optional)
# ---------------------------------------------------------------------------

class LightweightAttention(nn.Module):
    def __init__(self, ch: int, spatial_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.qkv  = nn.Linear(ch, ch * 3)
        self.proj = nn.Linear(ch, ch)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        qkv = self.qkv(tokens)
        q, k, v = qkv.chunk(3, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
        attn = attn.softmax(dim=-1)
        out  = torch.matmul(attn, v)
        out  = self.proj(out)
        return out.transpose(1, 2).contiguous().view(B, C, H, W)


# ---------------------------------------------------------------------------
# Upsampling: nearest + conv (default) or blur-conv
# ---------------------------------------------------------------------------

class BlurUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: Optional[List[int]] = None, groups: int = 1) -> None:
        super().__init__()
        if kernel is None:
            kernel = [1, 3, 3, 1]
        k = torch.tensor(kernel, dtype=torch.float32)
        k = (k[:, None] * k[None, :])
        k = k / k.sum()
        self.register_buffer('blur_kernel', k)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv     = nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=groups)
        self.act      = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        B, C, H, W = x.shape
        kernel  = self.blur_kernel[None, None, :, :].repeat(C, 1, 1, 1)
        ks      = self.blur_kernel.size(0)
        pad     = ks - 1
        pad_l   = pad // 2
        x = F.pad(x, (pad_l, pad - pad_l, pad_l, pad - pad_l), mode='reflect')
        x = F.conv2d(x, kernel, padding=0, groups=C)
        return self.act(self.conv(x))


# ---------------------------------------------------------------------------
# Core DLR block: class + z conditioned directional Mamba scanning
# ---------------------------------------------------------------------------

class RealVisionMamba2x2OnMap(nn.Module):
    """
    Directional Latent Routing (DLR) block operating on a 2-D feature map.

    For each scan direction d:
      1. Flatten the spatial map into a sequence along direction d.
      2. Modulate each sequence element with γ, β derived from (z_dir_chunk ‖ class_emb).
      3. Run Mamba SSM along the sequence.
      4. Unscan back to 2-D.
    Direction contributions are combined with learned, class-biased softmax weights.
    """

    def __init__(
        self,
        ch: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        spatial_size: int = 8,
        scan_directions: Optional[List[str]] = None,
        use_z_routing: bool = False,
        z_routing_dim: Optional[int] = None,
        z_mlp_hidden: Optional[int] = None,
        random_corner: bool = False,
        temperature: float = 2.0,
        routing_clip: Optional[float] = None,
        residual_routing: bool = False,
        use_dense: bool = False,
        dense_compression: float = 0.5,
        num_classes: Optional[int] = None,
        class_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.ch              = ch
        self.spatial_size    = spatial_size
        self.random_corner   = random_corner
        self.routing_clip    = routing_clip
        self.residual_routing = residual_routing
        self.use_dense       = use_dense
        self.num_classes     = num_classes
        self.class_embed_dim = class_embed_dim

        if scan_directions is None:
            scan_directions = ['row_fwd', 'row_bwd', 'col_fwd', 'col_bwd']
        self.scan_directions = scan_directions
        self.num_directions  = len(scan_directions)

        self.use_z_routing = use_z_routing
        self.z_routing_dim = z_routing_dim

        if self.use_z_routing:
            assert z_routing_dim is not None, "z_routing_dim required when use_z_routing=True"
            assert z_routing_dim % self.num_directions == 0, (
                f"z_routing_dim={z_routing_dim} must be divisible by num_directions={self.num_directions}"
            )
            self.z_chunk_dim = z_routing_dim // self.num_directions
            if z_mlp_hidden is None:
                z_mlp_hidden = self.z_chunk_dim * 2

            if num_classes is not None:
                # Per-direction class embeddings for Hybrid DLR
                self.class_embeds = nn.ModuleDict({
                    d: nn.Embedding(num_classes, class_embed_dim)
                    for d in self.scan_directions
                })
                for emb in self.class_embeds.values():
                    nn.init.normal_(emb.weight, 0, 0.25)

                # Class → directional routing bias (small init → z_dir dominates early)
                self.class_routing_proj = nn.Linear(class_embed_dim, self.num_directions)
                nn.init.normal_(self.class_routing_proj.weight, std=0.001)
                nn.init.zeros_(self.class_routing_proj.bias)
                self.routing_alpha = nn.Parameter(torch.full((self.num_directions,), 0.01))

            # Per-direction MLP: (z_chunk ‖ class_emb) → (γ, β) for sequence modulation
            input_dim = self.z_chunk_dim + (class_embed_dim if num_classes else 0)
            self.z_projections = nn.ModuleDict({
                d: nn.Sequential(
                    nn.Linear(input_dim, z_mlp_hidden),
                    nn.GELU(),
                    nn.Linear(z_mlp_hidden, 2 * ch),
                )
                for d in self.scan_directions
            })

        self.mamba_blocks = nn.ModuleDict({
            d: RealVisionMambaBlock(ch, d_state, d_conv, expand)
            for d in self.scan_directions
        })

        if self.use_dense:
            self.dense_compress = nn.Conv2d(ch * 2, ch, 1)

        self.direction_weighting = DirectionalWeighting(
            num_directions=self.num_directions,
            z_routing_dim=z_routing_dim if use_z_routing else None,
            temperature=temperature,
        )

        # Precompute true anti-diagonal traversal indices
        if 'diag_left' in self.scan_directions or 'diag_right' in self.scan_directions:
            H = W = self.spatial_size
            diag_left_idx, diag_right_idx = [], []
            for k in range(H + W - 1):
                i_start = min(k, H - 1)
                i_end   = max(0, k - (W - 1))
                for i in range(i_start, i_end - 1, -1):
                    diag_left_idx.append(i * W + (k - i))
                for i in range(i_end, i_start + 1):
                    diag_right_idx.append(i * W + (k - i))
            self.register_buffer('diag_left_indices',  torch.tensor(diag_left_idx,  dtype=torch.long))
            self.register_buffer('diag_right_indices', torch.tensor(diag_right_idx, dtype=torch.long))

        self.last_direction_weights = None
        self.last_gamma_abs_mean    = None
        self.last_beta_abs_mean     = None

    # ------------------------------------------------------------------
    # Scan / unscan helpers
    # ------------------------------------------------------------------

    def scan_sequence(self, h: Tensor, direction: str) -> Tensor:
        B, C, H, W = h.shape
        if direction == 'row_fwd':
            return h.view(B, C, H * W).transpose(1, 2)
        elif direction == 'row_bwd':
            return h.flip([3]).contiguous().view(B, C, H * W).transpose(1, 2)
        elif direction == 'col_fwd':
            return h.permute(0, 1, 3, 2).contiguous().view(B, C, H * W).transpose(1, 2)
        elif direction == 'col_bwd':
            return h.flip([2]).permute(0, 1, 3, 2).contiguous().view(B, C, H * W).transpose(1, 2)
        elif direction == 'diag_left':
            return h.view(B, C, H * W).transpose(1, 2)[:, self.diag_left_indices, :]
        elif direction == 'diag_right':
            return h.view(B, C, H * W).transpose(1, 2)[:, self.diag_right_indices, :]
        else:
            raise ValueError(f"Unknown scan direction: {direction!r}")

    def unscan_sequence(self, seq: Tensor, direction: str, H: int, W: int) -> Tensor:
        B, N, C = seq.shape
        if direction == 'row_fwd':
            return seq.transpose(1, 2).view(B, C, H, W)
        elif direction == 'row_bwd':
            return seq.transpose(1, 2).view(B, C, H, W).flip([3])
        elif direction == 'col_fwd':
            return seq.transpose(1, 2).view(B, C, W, H).permute(0, 1, 3, 2)
        elif direction == 'col_bwd':
            return seq.transpose(1, 2).view(B, C, W, H).permute(0, 1, 3, 2).flip([2])
        elif direction == 'diag_left':
            flat = torch.zeros(B, H * W, C, device=seq.device, dtype=seq.dtype)
            flat[:, self.diag_left_indices, :] = seq
            return flat.transpose(1, 2).view(B, C, H, W)
        elif direction == 'diag_right':
            flat = torch.zeros(B, H * W, C, device=seq.device, dtype=seq.dtype)
            flat[:, self.diag_right_indices, :] = seq
            return flat.transpose(1, 2).view(B, C, H, W)
        else:
            raise ValueError(f"Unknown scan direction: {direction!r}")

    @staticmethod
    def _apply_rotation(x: Tensor, k) -> Tensor:
        if k is None or k == 0:
            return x
        return torch.rot90(x, k=k, dims=[2, 3])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        h: Tensor,
        z_dir: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        rot_k=None,
        prev_outputs: Optional[list] = None,
    ) -> Tensor:
        B, C, H, W = h.shape
        assert C == self.ch

        if self.use_dense and prev_outputs:
            h = self.dense_compress(torch.cat([prev_outputs[-1], h], dim=1))

        h_rot = self._apply_rotation(h, rot_k)

        # Compute directional weights (z-modulated + class-biased)
        if self.use_z_routing and z_dir is not None:
            class_routing_bias = None
            if self.num_classes is not None and y is not None and hasattr(self, 'class_routing_proj'):
                cls_emb_mean = sum(
                    self.class_embeds[d](y) for d in self.scan_directions
                ) / self.num_directions
                alpha             = torch.abs(self.routing_alpha).unsqueeze(0)
                class_routing_bias = alpha * self.class_routing_proj(cls_emb_mean)
            dir_weights = self.direction_weighting(z_dir, class_routing_bias)
        else:
            dir_weights = self.direction_weighting()

        self.last_direction_weights = (
            dir_weights.detach().cpu() if dir_weights.dim() == 1
            else dir_weights.mean(0).detach().cpu()
        )
        if hasattr(self, 'routing_alpha'):
            self.last_routing_alpha = self.routing_alpha.detach().cpu()

        outputs = []
        for idx, direction in enumerate(self.scan_directions):
            seq = self.scan_sequence(h_rot, direction)   # (B, L, C)

            # Hybrid DLR: z_dir chunk + class_emb → γ, β affine modulation
            if self.use_z_routing and z_dir is not None:
                z_chunk = z_dir[:, idx * self.z_chunk_dim:(idx + 1) * self.z_chunk_dim]

                if self.num_classes is not None and y is not None:
                    dlr_input = torch.cat([z_chunk, self.class_embeds[direction](y)], dim=1)
                else:
                    dlr_input = z_chunk

                gamma, beta = self.z_projections[direction](dlr_input).chunk(2, dim=1)

                if self.residual_routing:
                    if self.routing_clip is not None:
                        gamma = torch.tanh(gamma) * self.routing_clip + 1.0
                        beta  = torch.tanh(beta)  * self.routing_clip
                    else:
                        gamma = gamma + 1.0
                else:
                    if self.routing_clip is not None:
                        gamma = torch.clamp(gamma, -self.routing_clip, self.routing_clip)
                        beta  = torch.clamp(beta,  -self.routing_clip, self.routing_clip)

                seq = gamma.unsqueeze(1) * seq + beta.unsqueeze(1)

                if idx == 0:
                    self.last_gamma_abs_mean = gamma.abs().mean().item()
                    self.last_beta_abs_mean  = beta.abs().mean().item()

            seq = self.mamba_blocks[direction](seq)
            outputs.append(self.unscan_sequence(seq, direction, H, W))

        # Weighted combination of directional outputs
        if dir_weights.dim() == 1:
            result = sum(w * out for w, out in zip(dir_weights, outputs))
        else:
            result = torch.zeros_like(outputs[0])
            for idx, out in enumerate(outputs):
                result = result + dir_weights[:, idx:idx+1, None, None] * out

        return self._apply_rotation(result, -rot_k if rot_k is not None else None)


# ---------------------------------------------------------------------------
# Noise injection (class-conditioned stochastic variation)
# ---------------------------------------------------------------------------

class NoiseInjection(nn.Module):
    """Cascaded modulation: noise → class → feature."""

    def __init__(self, channels: int, num_classes: int, strength: float = 0.1) -> None:
        super().__init__()
        self.strength     = strength
        self.noise_gamma  = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.noise_beta   = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.class_emb    = nn.Embedding(num_classes, 64)
        self.class_linear = nn.Linear(64, 2)
        nn.init.zeros_(self.class_linear.weight)
        self.class_linear.bias.data[0] = 1.0

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if not self.training:
            return x
        B, C, H, W = x.shape
        noise    = torch.randn(B, 1, H, W, device=x.device, dtype=x.dtype)
        noise    = self.noise_gamma * noise + self.noise_beta
        gb       = self.class_linear(F.gelu(self.class_emb(y)))
        gamma    = gb[:, 0:1].view(-1, 1, 1, 1)
        beta     = gb[:, 1:2].view(-1, 1, 1, 1)
        return x + self.strength * (gamma * noise + beta)


# ---------------------------------------------------------------------------
# StyleGAN2-style modulated convolution block
# ---------------------------------------------------------------------------

class StyleGAN2Block(nn.Module):
    """
    Class-conditional weight-modulated conv (StyleGAN2).
    Style = class embedding; demodulated to unit variance; leaky-ReLU output.
    """

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.channels = channels
        self.style    = nn.Embedding(num_classes, channels)
        nn.init.zeros_(self.style.weight)   # style+1=1 at init → no signal collapse
        self.weight = nn.Parameter(
            torch.randn(channels, channels, 3, 3) * (2.0 / (channels * 9)) ** 0.5
        )
        self.bias           = nn.Parameter(torch.zeros(channels))
        self.noise_strength = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        B, C_in, H, W = x.shape
        C_out = self.channels

        style  = self.style(y).view(B, 1, C_in, 1, 1)
        weight = self.weight.unsqueeze(0) * (style + 1)              # modulate
        demod  = torch.rsqrt(weight.pow(2).sum([2, 3, 4], keepdim=True) + 1e-8)
        weight = (weight * demod).reshape(B * C_out, C_in, 3, 3)

        out = F.conv2d(x.reshape(1, B * C_in, H, W), weight, padding=1, groups=B)
        out = out.reshape(B, C_out, H, W) + self.bias.view(1, C_out, 1, 1)

        if self.training and self.noise_strength.item() > 0:
            out = out + self.noise_strength * torch.randn(B, 1, H, W, device=out.device)

        return F.leaky_relu(out, 0.2)


# ---------------------------------------------------------------------------
# Full DSS-GAN Generator (config-driven)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    DSS-GAN hierarchical Mamba generator.
    Accepts a config dict whose structure mirrors DSS_GAN-main/src/config/*.py.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.resolution  = config.get('resolution', 128)
        self.z_dim       = config['z_dim']
        self.num_classes = config['num_classes']

        self.use_vit_residual          = config.get('use_vit_residual', False)
        self.token_residual_weight     = config.get('token_residual_weight', 0.3)
        self.spatial_residual_weight   = config.get('spatial_residual_weight', 0.3)
        self.temperature               = config.get('temperature', 2.0)
        self.noise_strength            = config.get('noise_strength', 0.01)
        self.use_multiscale_skips      = config.get('use_multiscale_skips', False)
        self.skip_8_to_64              = config.get('skip_8_to_64', 0.0)
        self.skip_16_to_128            = config.get('skip_16_to_128', 0.0)
        self.use_blur_upsample         = config.get('use_blur_upsample', False)
        self.blur_kernel               = config.get('blur_kernel', [1, 3, 3, 1])
        self.dense_compression         = config.get('dense_compression', 0.5)
        self.use_lightweight_attention = config.get('use_lightweight_attention', False)
        self.attention_stages          = config.get('attention_stages', [])
        self.attention_residual_weight = config.get('attention_residual_weight', 0.1)
        self.use_grouped_conv          = config.get('use_grouped_conv', False)
        self.conv_groups               = config.get('conv_groups', 1)
        self.film_enabled              = config.get('film_enabled', {
            '8x8': False, '16x16': False, '32x32': False, '64x64': False
        })
        self.rotation_modes            = config.get('rotation_modes', ['none', 'rot180'])
        self._rot_to_k                 = {'none': None, 'rot90': 1, 'rot180': 2, 'rot270': 3}

        self.z_tok_dim = config.get('z_tok_dim', self.z_dim // 2)
        self.z_dir_dim = self.z_dim - self.z_tok_dim

        cfg_tok      = config['mamba_tokens']
        cfg_8        = config['mamba_8x8']
        cfg_16       = config['mamba_16x16']
        cfg_32       = config['mamba_32x32']
        cfg_64       = config['mamba_64x64']
        cfg_out      = config['output']
        z_mlp_hidden = config.get('z_mlp_hidden', {})

        self.ch_8x8    = config['ch_8x8']
        self.ch_16x16  = config['ch_16x16']
        self.ch_32x32  = config['ch_32x32']
        self.ch_64x64  = config['ch_64x64']
        self.ch_128x128 = config['ch_128x128']
        if self.resolution >= 256:
            self.ch_256x256 = config.get('ch_256x256', 196)
        if self.resolution >= 512:
            self.ch_512x512 = config.get('ch_512x512', 48)

        use_ps32 = config.get('use_pixel_shuffle_32', False)
        use_ps64 = config.get('use_pixel_shuffle_64', False)
        self.use_pixel_shuffle_32 = use_ps32
        self.use_pixel_shuffle_64 = use_ps64

        # ------------------------------------------------------------------
        # Helper: build one DLR stage (ModuleList of RealVisionMamba2x2OnMap)
        # ------------------------------------------------------------------
        def _dlr_stage(cfg: Dict, ch: int, spatial_size: int) -> nn.ModuleList:
            key = f'{spatial_size}x{spatial_size}'
            return nn.ModuleList([
                RealVisionMamba2x2OnMap(
                    ch=ch,
                    d_state=cfg['d_state'],
                    d_conv=cfg['d_conv'],
                    expand=cfg['expand'],
                    spatial_size=spatial_size,
                    scan_directions=cfg['scan_directions'],
                    use_z_routing=cfg['use_z_routing'],
                    z_routing_dim=self.z_dir_dim,
                    z_mlp_hidden=z_mlp_hidden.get(key),
                    random_corner=cfg['random_corner'],
                    temperature=self.temperature,
                    routing_clip=cfg.get('routing_clip'),
                    residual_routing=cfg.get('residual_routing', False),
                    use_dense=cfg.get('use_dense', False),
                    dense_compression=self.dense_compression,
                    num_classes=self.num_classes,
                    class_embed_dim=128,
                )
                for _ in range(cfg['depth'])
            ])

        # ------------------------------------------------------------------
        # Stage 0: z_tok → sequence → 8×8 map
        # ------------------------------------------------------------------
        self.z_to_tokens  = nn.Linear(self.z_tok_dim, 64 * self.ch_8x8)
        self.pos_enc64    = LearnedPositionalEncoding(64, self.ch_8x8)
        self.mamba_tokens = nn.ModuleList([
            RealVisionMambaBlock(self.ch_8x8, cfg_tok['d_state'], cfg_tok['d_conv'], cfg_tok['expand'])
            for _ in range(cfg_tok['depth'])
        ])
        self.class_embed = nn.Embedding(self.num_classes, self.ch_8x8)
        nn.init.normal_(self.class_embed.weight, std=0.02)

        # ------------------------------------------------------------------
        # DLR spatial stages
        # ------------------------------------------------------------------

        # 8×8
        self.mamba_8x8 = _dlr_stage(cfg_8, self.ch_8x8, 8)
        self.film_8x8  = FiLM(self.ch_8x8, self.num_classes)
        if cfg_8.get('noise', False):
            self.noise_8x8 = NoiseInjection(self.ch_8x8, self.num_classes, self.noise_strength)

        # 16×16
        self.up_8_to_16  = self._make_upsample(self.ch_8x8, self.ch_16x16)
        self.mamba_16x16 = _dlr_stage(cfg_16, self.ch_16x16, 16)
        self.film_16x16  = FiLM(self.ch_16x16, self.num_classes)
        if cfg_16.get('noise', False):
            self.noise_16x16 = NoiseInjection(self.ch_16x16, self.num_classes, self.noise_strength)

        # 32×32
        self.up_16_to_32  = self._make_upsample(self.ch_16x16, self.ch_32x32)
        if use_ps32:
            self.pixel_unshuffle_32 = nn.PixelUnshuffle(2)
            self.pixel_shuffle_32   = nn.PixelShuffle(2)
            ch32_eff = self.ch_32x32 * 4
            sp32     = 16
        else:
            ch32_eff = self.ch_32x32
            sp32     = 32
        self.mamba_32x32 = _dlr_stage(cfg_32, ch32_eff, sp32)
        if self.use_lightweight_attention and '32x32' in self.attention_stages:
            self.attn_32x32 = LightweightAttention(self.ch_32x32, 32)
        self.film_32x32 = FiLM(self.ch_32x32, self.num_classes)
        if cfg_32.get('noise', False):
            self.noise_32x32 = NoiseInjection(self.ch_32x32, self.num_classes, self.noise_strength)

        # 64×64
        self.up_32_to_64  = self._make_upsample(self.ch_32x32, self.ch_64x64)
        if use_ps64:
            self.pixel_unshuffle_64 = nn.PixelUnshuffle(2)
            self.pixel_shuffle_64   = nn.PixelShuffle(2)
            ch64_eff = self.ch_64x64 * 4
            sp64     = 32
        else:
            ch64_eff = self.ch_64x64
            sp64     = 64
        self.mamba_64x64 = _dlr_stage(cfg_64, ch64_eff, sp64)
        if self.use_lightweight_attention and '64x64' in self.attention_stages:
            self.attn_64x64 = LightweightAttention(self.ch_64x64, 64)
        self.film_64x64 = FiLM(self.ch_64x64, self.num_classes)
        if cfg_64.get('noise', False):
            self.noise_64x64 = NoiseInjection(self.ch_64x64, self.num_classes, self.noise_strength)

        # Multi-scale skips (optional)
        if self.use_multiscale_skips and self.skip_8_to_64 > 0:
            self.skip_proj_8_to_64 = nn.Sequential(
                nn.Upsample(scale_factor=8, mode='nearest'),
                nn.Conv2d(self.ch_8x8, self.ch_64x64, 1),
            )
        if self.use_multiscale_skips and self.skip_16_to_128 > 0:
            self.skip_proj_16_to_128 = nn.Sequential(
                nn.Upsample(scale_factor=8, mode='nearest'),
                nn.Conv2d(self.ch_16x16, self.ch_128x128, 1),
            )

        # 128×128 — StyleGAN2Block (res=128) or Mamba (res≥256)
        self.up_64_to_128 = self._make_upsample(self.ch_64x64, self.ch_128x128)
        if self.resolution == 128:
            self.style_refine_128 = StyleGAN2Block(self.ch_128x128, self.num_classes)
        else:
            cfg_128 = config['mamba_128x128']
            self.mamba_128x128 = _dlr_stage(cfg_128, self.ch_128x128, 128)
            self.film_128x128  = FiLM(self.ch_128x128, self.num_classes)

        # 256×256 — StyleGAN2Block (res=256) or Mamba (res=512)
        if self.resolution == 256:
            self.up_128_to_256 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(self.ch_128x128, self.ch_256x256, 3, padding=1),
                nn.GELU(),
            )
            self.style_refine_256 = StyleGAN2Block(self.ch_256x256, self.num_classes)
        elif self.resolution == 512:
            self.up_128_to_256 = self._make_upsample(self.ch_128x128, self.ch_256x256)
            cfg_256 = config['mamba_256x256']
            self.mamba_256x256 = _dlr_stage(cfg_256, self.ch_256x256, 256)
            self.film_256x256  = FiLM(self.ch_256x256, self.num_classes)

        # 512×512 — StyleGAN2Block
        if self.resolution == 512:
            self.up_256_to_512 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(self.ch_256x256, self.ch_512x512, 3, padding=1),
                nn.GELU(),
            )
            self.style_refine_512 = StyleGAN2Block(self.ch_512x512, self.num_classes)

        # ------------------------------------------------------------------
        # to-RGB head
        # ------------------------------------------------------------------
        if self.resolution == 128:
            final_ch_in = self.ch_128x128
        elif self.resolution == 256:
            final_ch_in = self.ch_256x256
        else:
            final_ch_in = self.ch_512x512

        refine_channels = cfg_out.get('refine_channels', [64])
        refine_layers: list = []
        in_ch = final_ch_in
        for out_ch_r in refine_channels:
            refine_layers += [nn.Conv2d(in_ch, out_ch_r, 3, padding=1), nn.GELU()]
            in_ch = out_ch_r
        self.refine     = nn.Sequential(*refine_layers) if refine_layers else nn.Identity()
        final_in_ch     = refine_channels[-1] if refine_channels else final_ch_in
        self.final_conv = nn.Conv2d(final_in_ch, cfg_out['final_ch'], 3, padding=1)
        self.to_rgb     = nn.Conv2d(cfg_out['final_ch'], 3, 1)

        # Logging buffers (populated during forward for ablation scripts)
        self.directions_w   = []
        self.dlr_gamma_abs  = []
        self.dlr_beta_abs   = []
        self.routing_alphas = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_upsample(self, in_ch: int, out_ch: int) -> nn.Module:
        if self.use_blur_upsample:
            groups = self.conv_groups if self.use_grouped_conv else 1
            return BlurUpsample(in_ch, out_ch, self.blur_kernel, groups)
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
        )

    def _run_mamba_stage(
        self,
        h: Tensor,
        blocks: nn.ModuleList,
        z_dir: Tensor,
        y: Tensor,
        rot_k,
    ) -> Tensor:
        prev = []
        h_before = h if self.use_vit_residual else None
        for blk in blocks:
            h = blk(h, z_dir=z_dir, y=y, rot_k=rot_k, prev_outputs=prev)
            prev.append(h.clone())
            if hasattr(blk, 'last_direction_weights') and blk.last_direction_weights is not None:
                self.directions_w.append(blk.last_direction_weights)
            if hasattr(blk, 'last_routing_alpha'):
                self.routing_alphas.append(blk.last_routing_alpha)
            if hasattr(blk, 'last_gamma_abs_mean') and blk.last_gamma_abs_mean is not None:
                self.dlr_gamma_abs.append(blk.last_gamma_abs_mean)
                self.dlr_beta_abs.append(blk.last_beta_abs_mean)
        if self.use_vit_residual:
            h = h + self.spatial_residual_weight * h_before
        return h

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z: Tensor, y: Tensor) -> Tensor:
        B = z.size(0)
        z_tok, z_dir = z.split([self.z_tok_dim, self.z_dir_dim], dim=1)

        self.directions_w   = []
        self.dlr_gamma_abs  = []
        self.dlr_beta_abs   = []
        self.routing_alphas = []

        if self.training:
            idx    = torch.randint(0, len(self.rotation_modes), (1,), device=z.device).item()
            rot_k  = self._rot_to_k.get(self.rotation_modes[idx], None)
        else:
            rot_k = None

        # ---- Stage 0: z_tok → token sequence → 8×8 map ----
        t = self.z_to_tokens(z_tok).view(B, 64, self.ch_8x8)
        t = self.pos_enc64(t)
        if self.use_vit_residual:
            t_skip = t
            for blk in self.mamba_tokens:
                t = blk(t)
            t = t + self.token_residual_weight * t_skip
        else:
            for blk in self.mamba_tokens:
                t = blk(t)
        t = t + self.class_embed(y).unsqueeze(1)
        h = t.transpose(1, 2).contiguous().view(B, self.ch_8x8, 8, 8)

        # ---- 8×8 ----
        h = self._run_mamba_stage(h, self.mamba_8x8, z_dir, y, rot_k)
        h_8x8 = h
        if self.film_enabled.get('8x8', False):
            h = self.film_8x8(h, y)
        if hasattr(self, 'noise_8x8'):
            h = self.noise_8x8(h, y)

        # ---- 16×16 ----
        h = self.up_8_to_16(h)
        h = self._run_mamba_stage(h, self.mamba_16x16, z_dir, y, rot_k)
        h_16x16 = h
        if self.film_enabled.get('16x16', False):
            h = self.film_16x16(h, y)
        if hasattr(self, 'noise_16x16'):
            h = self.noise_16x16(h, y)

        # ---- 32×32 ----
        h = self.up_16_to_32(h)
        if self.use_pixel_shuffle_32:
            h = self.pixel_unshuffle_32(h)
        h = self._run_mamba_stage(h, self.mamba_32x32, z_dir, y, rot_k)
        if self.use_pixel_shuffle_32:
            h = self.pixel_shuffle_32(h)
        if hasattr(self, 'attn_32x32'):
            h = h + self.attention_residual_weight * self.attn_32x32(h)
        if self.film_enabled.get('32x32', False):
            h = self.film_32x32(h, y)
        if hasattr(self, 'noise_32x32'):
            h = self.noise_32x32(h, y)

        # ---- 64×64 ----
        h = self.up_32_to_64(h)
        if self.use_multiscale_skips and self.skip_8_to_64 > 0:
            h = h + self.skip_8_to_64 * self.skip_proj_8_to_64(h_8x8)
        if self.use_pixel_shuffle_64:
            h = self.pixel_unshuffle_64(h)
        h = self._run_mamba_stage(h, self.mamba_64x64, z_dir, y, rot_k)
        if self.use_pixel_shuffle_64:
            h = self.pixel_shuffle_64(h)
        if hasattr(self, 'attn_64x64'):
            h = h + self.attention_residual_weight * self.attn_64x64(h)
        if self.film_enabled.get('64x64', False):
            h = self.film_64x64(h, y)
        if hasattr(self, 'noise_64x64'):
            h = self.noise_64x64(h, y)

        # ---- 128×128 ----
        h = self.up_64_to_128(h)
        if self.use_multiscale_skips and self.skip_16_to_128 > 0:
            h = h + self.skip_16_to_128 * self.skip_proj_16_to_128(h_16x16)
        if self.resolution == 128:
            h = self.style_refine_128(h, y)
        else:
            h = self._run_mamba_stage(h, self.mamba_128x128, z_dir, y, rot_k)
            if self.film_enabled.get('128x128', False):
                h = self.film_128x128(h, y)

        # ---- 256×256 ----
        if self.resolution == 256:
            h = self.up_128_to_256(h)
            h = self.style_refine_256(h, y)
        elif self.resolution == 512:
            h = self.up_128_to_256(h)
            h = self._run_mamba_stage(h, self.mamba_256x256, z_dir, y, rot_k)
            if self.film_enabled.get('256x256', False):
                h = self.film_256x256(h, y)

        # ---- 512×512 ----
        if self.resolution == 512:
            h = self.up_256_to_512(h)
            h = self.style_refine_512(h, y)

        # ---- to RGB ----
        h = self.refine(h)
        h = F.gelu(self.final_conv(h))
        return torch.tanh(self.to_rgb(h))

    def enable_weight_logging(self) -> None:
        pass  # logging is always on; kept for interface compatibility


# ---------------------------------------------------------------------------
# PATE-DSS-GAN entry point: simple keyword args → Generator config → Generator
# ---------------------------------------------------------------------------

class DSSGANGenerator(nn.Module):
    """
    Wraps the full DSS-GAN Generator for the PATE-DSS-GAN training interface.

    Parameters
    ----------
    latent_dim : int
        Total z dimension. Split internally: z_tok (≈ half) for token generation,
        z_dir (≈ half, divisible by num_scan_dirs) for DLR modulation.
    num_classes : int
    image_size : int
        Target resolution: 128, 256, or 512.
    base_channels : int
        Base channel multiplier for all spatial stages.
    mamba_d_model : int
        Accepted for interface compatibility; channels are governed by base_channels.
    """

    _SCAN_DIRS_DEFAULT = ['row_fwd', 'col_bwd', 'diag_left']

    def __init__(
        self,
        latent_dim: int = 152,
        num_classes: int = 3,
        image_size: int = 128,
        base_channels: int = 148,
        scan_directions: Optional[List[str]] = None,
        mamba_d_model: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim  = latent_dim
        self.num_classes = num_classes
        self.image_size  = image_size
        self.scan_directions = scan_directions or list(self._SCAN_DIRS_DEFAULT)

        config = self._build_config(
            latent_dim, num_classes, image_size, base_channels, self.scan_directions
        )
        self._gen = Generator(config)

        print(
            f"[DSSGANGenerator] resolution={image_size} | "
            f"z_dim={latent_dim} (tok={self._gen.z_tok_dim}, dir={self._gen.z_dir_dim}) | "
            f"ch_main={config['ch_8x8']} | num_classes={num_classes} | "
            f"scan_dirs={self.scan_directions}"
        )

    @classmethod
    def _build_config(
        cls,
        latent_dim: int,
        num_classes: int,
        image_size: int,
        base_channels: int,
        scan_dirs: Optional[List[str]] = None,
    ) -> Dict:
        if scan_dirs is None:
            scan_dirs = cls._SCAN_DIRS_DEFAULT
        num_dir   = len(scan_dirs)  # 3

        # Latent split matching DSS_GAN-main: z_tok_dim + z_dir_dim = z_dim
        # z_dir_dim must be divisible by num_dir.
        # Default layout (from original configs):
        #   128: tok=92, dir_per=20, z_dim=152
        #   256: tok=88, dir_per=28, z_dim=172
        #   512: tok=98, dir_per=36, z_dim=206
        if image_size <= 128:
            dir_per = 20
        elif image_size <= 256:
            dir_per = 28
        else:
            dir_per = 36
        z_dir_dim = dir_per * num_dir
        z_tok_dim = latent_dim - z_dir_dim

        ch = base_channels  # 148 for 128/256, tapered for 512

        # Routing clips matching original (NoFiLM variant)
        rc = {'8x8': 0.8, '16x16': 0.8, '32x32': 0.5, '64x64': 0.3, '128x128': 0.2}

        def _dlr_cfg(
            depth: int, d_state: int, d_conv: int, expand: float,
            rc_key: str, extra: Optional[Dict] = None,
        ) -> Dict:
            cfg: Dict = {
                'depth': depth, 'd_state': d_state, 'd_conv': d_conv, 'expand': expand,
                'scan_directions': scan_dirs,
                'use_z_routing': True, 'random_corner': False, 'noise': False,
                'routing_clip': rc[rc_key], 'residual_routing': True,
            }
            if extra:
                cfg.update(extra)
            return cfg

        # Channel schedule per resolution (matches DSS_GAN-main configs)
        if image_size <= 128:
            ch_8 = ch_16 = ch_32 = ch_64 = ch
            ch_128 = ch + 20  # 168 with base=148
        elif image_size <= 256:
            ch_8 = ch_16 = ch_32 = ch_64 = ch
            ch_128 = ch + 20
        else:  # 512
            ch_8 = base_channels
            ch_16 = max(base_channels - 20, 64)
            ch_32 = max(base_channels - 40, 64)
            ch_64 = max(base_channels - 50, 64)
            ch_128 = max(base_channels - 68, 48)

        # Mamba expand schedule — DSS-GAN paper Table 13
        if image_size <= 128:
            expand_8_32, expand_64 = 2.0, 1.5
        elif image_size <= 256:
            expand_8_32, expand_64 = 1.5, 1.0
        else:
            expand_8_32, expand_64 = 1.0, 1.0

        config: Dict = {
            'resolution':  image_size,
            'z_dim':       latent_dim,
            'z_tok_dim':   z_tok_dim,
            'num_classes': num_classes,
            'temperature': 1.0,
            'noise_strength': 0.01,
            'use_vit_residual': True,
            'token_residual_weight':   0.3,
            'spatial_residual_weight': 0.3,
            'rotation_modes':  ['none', 'rot180'],
            'film_enabled': {'8x8': False, '16x16': False, '32x32': False, '64x64': False},
            'use_multiscale_skips': False, 'skip_8_to_64': 0.0, 'skip_16_to_128': 0.0,
            'use_blur_upsample': False, 'blur_kernel': [1, 3, 3, 1],
            'use_dense_connections': False, 'use_lightweight_attention': False,
            'use_grouped_conv': True,
            # Channels
            'ch_8x8': ch_8, 'ch_16x16': ch_16, 'ch_32x32': ch_32, 'ch_64x64': ch_64,
            'ch_128x128': ch_128,
            'use_pixel_shuffle_32': False, 'use_pixel_shuffle_64': False,
            # z_mlp_hidden (matches original config.py)
            'z_mlp_hidden': {'8x8': 64, '16x16': 128, '32x32': 256, '64x64': 256},
            # Mamba stage configs — Table 13 (128: expand 2.0 / 1.5; 256: 1.5 / 1.0)
            'mamba_tokens': {'depth': 2, 'd_state': 64, 'd_conv': 4, 'expand': 2},
            'mamba_8x8':   _dlr_cfg(2, 64, 4, expand_8_32, '8x8'),
            'mamba_16x16': _dlr_cfg(1, 64, 4, expand_8_32, '16x16'),
            'mamba_32x32': _dlr_cfg(1, 64, 4, expand_8_32, '32x32'),
            'mamba_64x64': _dlr_cfg(1, 64, 3, expand_64, '64x64'),
            # to-RGB
            'output': {'final_ch': 64, 'refine_channels': [128]},
        }

        if image_size >= 256:
            config['ch_256x256'] = 196
            config['mamba_128x128'] = _dlr_cfg(
                1, 48, 3, 1.0, '128x128',
                {'scan_directions': scan_dirs},
            )
            config['output'] = {'final_ch': 64, 'refine_channels': [128]}

        if image_size >= 512:
            config['ch_256x256'] = 64
            config['ch_512x512'] = 48
            config['mamba_128x128'] = _dlr_cfg(
                1, 48, 3, 1, '128x128',
                {'scan_directions': scan_dirs},
            )
            config['mamba_256x256'] = _dlr_cfg(
                1, 16, 3, 1, '64x64',
                {'scan_directions': scan_dirs[:2], 'routing_clip': 0.15},
            )
            config['output'] = {'final_ch': 16, 'refine_channels': [32]}

        return config

    def forward(self, z: Tensor, c: Tensor) -> Tensor:
        return self._gen(z, c)

    def sample_latent(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device)
