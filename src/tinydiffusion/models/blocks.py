"""Building blocks shared by the UNet: normalisation, ResBlocks, attention, resampling."""

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _checkpointed(
    body: Callable[..., torch.Tensor], use_checkpoint: bool, *args: torch.Tensor
) -> torch.Tensor:
    """Run a block's body under gradient checkpointing where it pays off.

    Checkpointing drops a block's intermediate activations and recomputes them
    during the backward pass: roughly a third more compute for a large cut in
    memory, which is what lets a wider model or a bigger batch fit at all.

    There is nothing to save when no backward pass will follow, so a block
    falls through to the plain call under :func:`torch.no_grad` — which is
    every sampler, the validation scoring and the FID pass. Recomputing there
    would be pure overhead, and ``torch.utils.checkpoint`` warns about it.

    ``use_reentrant=False`` picks the non-reentrant implementation. Besides
    handling inputs that do not require grad, it restores the RNG state before
    recomputing, so the backward pass sees the same dropout mask the forward
    drew. The reentrant version does not, and silently differentiates a
    different network than the one that produced the output.

    Args:
        body: the block's real forward pass.
        use_checkpoint: whether this block was built to be checkpointed.
        *args: the tensors to pass to `body`.

    Returns:
        Whatever `body` returns.
    """
    if use_checkpoint and torch.is_grad_enabled():
        return cast(torch.Tensor, checkpoint(body, *args, use_reentrant=False))
    return body(*args)


class Float32GroupNorm(nn.GroupNorm):
    """GroupNorm that normalises in float32 whatever precision it is handed.

    Normalisation is where reduced precision does the most damage — the
    variance is a sum over a whole group of channels, and float16 runs out of
    mantissa long before that sum is done — so the weights here are left in
    float32 by :func:`~tinydiffusion.utils.fp16.convert_module_to_f16` and the
    input is promoted to meet them.

    The promotion is not merely for accuracy: ``F.group_norm`` refuses a
    float16 input against a float32 weight outright on CUDA, so a network with
    half-precision convolutions and full-precision norms does not run at all
    without it.

    Both casts are free in the two cases that are not
    :attr:`~tinydiffusion.training.config.TrainConfig.full_fp16`. Under plain
    float32 they are no-ops, and under autocast GroupNorm is on the float32
    list already, so this only moves a cast that was going to happen anyway.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise `x` in float32 and hand it back in its own dtype.

        Args:
            x: (B, C, ...) feature map.

        Returns:
            The normalised feature map, in `x`'s dtype.
        """
        return super().forward(x.float()).to(x.dtype)


def group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm using the largest group count <= max_groups that divides channels.

    A plain min(32, channels) is not enough: 48 channels with 32 groups raises,
    since 48 % 32 != 0.

    Args:
        channels: number of channels to normalise.
        max_groups: upper bound on the group count.

    Returns:
        A :class:`Float32GroupNorm` whose group count divides channels.
    """
    groups = next(g for g in range(min(max_groups, channels), 0, -1) if channels % g == 0)
    return Float32GroupNorm(groups, channels)


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
    """Pre-activation residual block with FiLM time conditioning.

    Args:
        in_channels: channels of the incoming feature map.
        out_channels: channels to produce.
        time_dim: width of the time embedding the FiLM projection reads.
        dropout: dropout applied before the second convolution.
        use_checkpoint: recompute this block's activations in the backward
            pass rather than holding them. See :func:`_checkpointed`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint

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
        return _checkpointed(self._forward, self.use_checkpoint, x, time_emb)

    def _forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """The block's body, split out so it can be recomputed on demand."""
        h = self.conv1(F.silu(self.norm1(x)))

        projection = self.time_proj(F.silu(time_emb)).to(h.dtype)
        scale, shift = projection[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over the spatial dimensions.

    Args:
        channels: width of the feature map to attend over.
        num_heads: attention heads. Must divide `channels`.
        use_checkpoint: recompute this block's activations in the backward
            pass rather than holding them. See :func:`_checkpointed`.

    Raises:
        ValueError: if `channels` is not divisible by `num_heads`.
    """

    def __init__(self, channels: int, num_heads: int = 4, use_checkpoint: bool = False) -> None:
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(f"{channels} channels not divisible by {num_heads} heads")

        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
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
        return _checkpointed(self._forward, self.use_checkpoint, x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """The block's body, split out so it can be recomputed on demand."""
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = (t.transpose(-2, -1) for t in qkv.unbind(1))
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

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Run each child layer, passing `time_emb` to the ones that accept it."""
        for layer in self:
            x = layer(x, time_emb) if isinstance(layer, ResBlock) else layer(x)
        return x
