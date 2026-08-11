"""DDPM-style UNet for diffusion models.
 
Replaces the NaiveUnet from minDiffusion with the architecture used in
Ho et al. 2020 (DDPM): pre-activation ResBlocks with FiLM time conditioning
at every resolution, self-attention at chosen scales, and a spatial
bottleneck instead of a global-vector collapse.
"""

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

class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=x.device, dtype=torch.float32)
            / half
        )
        args = x.float().flatten()[:, None] * freqs[None]
        emb = torch.cat([args.cos(), args.min()], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)

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


class DownSample(nn.Module):
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

class UNet(nn.Module):
    """DDPM UNet.
 
    Args:
        in_channels: channels of the noisy input image.
        out_channels: channels to predict (equals in_channels for epsilon-pred).
        base: base channel width. DDPM uses 128 for CIFAR-10.
        ch_mult: channel multiplier per resolution level.
        n_res: residual blocks per level.
        attn_resolutions: spatial sizes (H == W) at which to apply attention.
        dropout: dropout inside ResBlocks. DDPM uses 0.1 for CIFAR-10.
        image_size: input resolution, needed to resolve attn_resolutions.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base: int = 128,
        ch_mult: tuple[int, ...] = (1, 2, 2, 2),
        n_res: int = 2,
        attn_resolutions: tuple[int, ...] = (16,),
        dropout: float = 0.1,
        image_size: int = 32,
        n_heads: int = 4,
    ) -> None:
        super().__init__()

        t_dim = base * 4
        self.time = TimeEmbedding(base, t_dim)
        self.init_conv = nn.Conv2d(in_channels, base, 3, padding=1)

        # the encoder
        self.downs = nn.ModuleList()
        skip_chans = [base]
        ch = base
        res = image_size

        for i, mult in enumerate(ch_mult):
            out_ch = base * mult
            for _ in range(n_res):
                layers: list[nn.Module] = [ResBlock(ch, out_ch, t_dim, dropout)]
                ch = out_ch
                if res in attn_resolutions:
                    layers.append(SelfAttention(ch, n_heads))
                self.downs.append(TimestepSequential(*layers))
                skip_chans.append(ch)

            if i != len(ch_mult) - 1:
                self.downs.append(TimestepSequential((DownSample(ch))))
                skip_chans.append(ch)
                res //= 2

        # the bottleneck: stays spatial, no vector collapse
        self.mid = TimestepSequential(
            ResBlock(ch, ch, t_dim, dropout),
            SelfAttention(ch, n_heads),
            ResBlock(ch, ch, t_dim, dropout)
        )

        # the decoder
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base * mult
            for j in range(n_res + 1):
                layers = [ResBlock(ch + skip_chans.pop(), out_ch, t_dim, dropout)]
                ch = out_ch
                if res in attn_resolutions:
                    layers.append(SelfAttention(ch, n_heads))
                if i != 0 and j == n_res:
                    layers.append(Upsample(ch))
                    res *= 2
                self.ups.append(TimestepSequential(*layers))

        self.out = nn.Sequential(
            norm(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, out_ch, 3, padding=1))
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) noisy image. t: (B,) integer timesteps."""
        t_emb = self.time(t)

        h = self.init_conv(x)
        hs = [h]
        for module in self.downs:
            h = module(h, t_emb)
            hs.append(h)

        h = self.mid(h, t_emb)

        for module in self.ups:
            h = module(torch.cat([h, hs.pop()], dim=1), t_emb)

        return self.out(h)