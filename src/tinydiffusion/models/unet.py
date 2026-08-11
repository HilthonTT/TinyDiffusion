"""DDPM-style UNet for diffusion models.
 
Replaces the NaiveUnet from minDiffusion with the architecture used in
Ho et al. 2020 (DDPM): pre-activation ResBlocks with FiLM time conditioning
at every resolution, self-attention at chosen scales, and a spatial
bottleneck instead of a global-vector collapse.
"""

import torch.nn as nn
import torch

from blocks import (
    ResBlock,
    SelfAttention,
    TimestepSequential,
    Downsample,
    Upsample,
    norm,
    zero_module,
)
from embeddings import TimeEmbedding

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
                self.downs.append(TimestepSequential((Downsample(ch))))
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