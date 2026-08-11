"""Timestep embeddings feeding the UNet's FiLM conditioning."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP.

    Args:
        dim: width of the raw sinusoidal embedding.
        out_dim: width of the projected embedding handed to the ResBlocks.
    """

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
        """Embed integer timesteps.

        Args:
            x: (B,) integer timesteps.

        Returns:
            (B, out_dim) embedding.
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / half
        )
        args = x.float().flatten()[:, None] * freqs[None]
        emb = torch.cat([args.cos(), args.min()], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)
