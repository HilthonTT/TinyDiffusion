"""Timestep and class embeddings feeding the UNet's FiLM conditioning."""

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
        self.out_dim = out_dim
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
        emb = torch.cat([args.cos(), args.sin()], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class LabelEmbedding(nn.Module):
    """Class-label embedding with a reserved null token.

    The table holds ``num_classes + 1`` rows: one per class, plus a final row
    at index ``num_classes`` standing for "no class given". That row is what
    classifier-free guidance needs — Ho & Salimans 2022
    (https://arxiv.org/abs/2207.12598) train one network to do both jobs by
    replacing the label with the null token for a fraction of training
    examples, then extrapolate away from the null prediction at sample time.
    Without a null token there is no unconditional prediction to extrapolate
    from, so the reserved row is the whole mechanism.

    The embedding is summed into the timestep embedding rather than
    concatenated, so every ResBlock's existing FiLM conditioning carries the
    class for free.

    Args:
        num_classes: number of real classes. The null token is added on top.
        out_dim: width of the timestep embedding it is summed into.

    Raises:
        ValueError: if `num_classes` is not positive.
    """

    def __init__(self, num_classes: int, out_dim: int) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        self.num_classes = num_classes
        self.embed = nn.Embedding(num_classes + 1, out_dim)

    @property
    def null_class(self) -> int:
        """The index standing for "unconditional".

        Returns:
            ``num_classes``, the reserved final row.
        """
        return self.num_classes

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Embed class labels.

        Args:
            y: (B,) integer labels in ``[0, num_classes]``, where
                ``num_classes`` itself is the null token.

        Returns:
            (B, out_dim) embedding.
        """
        return self.embed(y)
