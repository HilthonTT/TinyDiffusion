"""Building blocks shared by the UNet: normalisation, ResBlocks, attention, resampling."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm using the largest group count <= max_groups that divides channels.

    A plain min(32, channels) is not enough: 48 channels with 32 groups raises,
    since 48 % 32 != 0.

    Args:
        channels: number of channels to normalise.
        max_groups: upper bound on the group count.

    Returns:
        A GroupNorm whose group count divides channels.
    """
    groups = next(g for g in range(min(max_groups, channels), 0, -1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out a module's parameters so it starts as the zero map.

    Args:
        module: module to zero in place.

    Returns:
        The same module, for use inline.
    """
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(nn.Module):
    """Pre-activation residual block with FiLM time conditioning."""

    def __init__(
        self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_dim, 2 * out_channels)

        self.norm2 = group_norm(out_channels)
        self.drop = nn.Dropout(dropout)
        self.conv2 = zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: (B, in_channels, H, W) feature map.
            time_emb: (B, time_dim) time embedding.

        Returns:
            (B, out_channels, H, W) feature map.
        """
        h = self.conv1(F.silu(self.norm1(x)))

        scale, shift = self.time_proj(F.silu(time_emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over the spatial dimensions."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(f"{channels} channels not divisible by {num_heads} heads")

        self.num_heads = num_heads
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over H*W positions and add the result back to the input.

        Args:
            x: (B, C, H, W) feature map.

        Returns:
            (B, C, H, W) feature map.
        """
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
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
        """Halve the spatial resolution of `x`."""
        return self.op(x)


class Upsample(nn.Module):
    """Nearest-neighbour upsample + conv, avoiding transposed-conv checkerboarding."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Double the spatial resolution of `x`."""
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class TimestepSequential(nn.Sequential):
    """Sequential that forwards the time embedding to ResBlocks only."""

    # Deliberately widens Sequential.forward: children that need conditioning get it.
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Run each child layer, passing `time_emb` to the ones that accept it."""
        for layer in self:
            x = layer(x, time_emb) if isinstance(layer, ResBlock) else layer(x)
        return x
