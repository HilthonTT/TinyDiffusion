import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm using the largest group count <= max_groups that divides channels.

    A plain min(32, channels) is not enough: 48 channels with 32 groups raises,
    since 48 % 32 != 0.
    """
    groups = next(
        g for g in range(min(max_groups, channels), 0, -1) if channels % g == 0
    )
    return nn.GroupNorm(groups, channels)

def zero_module(module: nn.Module) -> nn.Module:
    """Zero out a module's parameters so it stars as the zero map."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class ResBlock(nn.Module):
    """Pre-activation residual block with FiLM time conditioning."""
 
    def __init__(
        self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm1 = norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
 
        self.t_proj = nn.Linear(t_dim, 2 * out_ch)
 
        self.norm2 = norm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1))
 
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
 
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
 
        scale, shift = self.t_proj(F.silu(t_emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
 
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)

class SelfAttention(nn.Module):
    """Multi-head self-attention over the spatial dimensions."""

    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()

        if channels % n_heads != 0:
            raise ValueError(f"{channels} channels not divisible by {n_heads} heads")

        self.n_heads = n_heads
        self.norm = norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b, 3, self.n_heads, c // self.n_heads, h * w)
        q, k, v = (t.transpose(-2, -1) for t in qkv.unbind(1))  # b, heads, hw, d
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Strided conv downsample (learned, unlike MaxPool)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

class Upsample(nn.Module):
    """Nearest-neighbour upsample + conv, avoiding transposed-conv checkerboarding."""
 
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))

class TimestepSequential(nn.Sequential):
    """Sequential that forwards the time embedding to ResBlocks only."""

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = layer(x, t_emb) if isinstance(layer, ResBlock) else layer(x)
        return x