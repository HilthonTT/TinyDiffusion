"""DDPM-style UNet for diffusion models.

Replaces the NaiveUnet from minDiffusion with the architecture used in
Ho et al. 2020 (DDPM): pre-activation ResBlocks with FiLM time conditioning
at every resolution, self-attention at chosen scales, and a spatial
bottleneck instead of a global-vector collapse.
"""

import torch
import torch.nn as nn

from tinydiffusion.models.blocks import (
    Downsample,
    ResBlock,
    SelfAttention,
    TimestepSequential,
    Upsample,
    group_norm,
    zero_module,
)
from tinydiffusion.models.embeddings import LabelEmbedding, TimeEmbedding

# DDPM projects the sinusoidal embedding to 4x the base width before conditioning.
TIME_EMBED_MULT = 4


class UNet(nn.Module):
    """DDPM UNet.

    Args:
        in_channels: channels of the noisy input image.
        out_channels: channels to predict (equals in_channels for epsilon-pred).
        base_channels: base channel width. DDPM uses 128 for CIFAR-10.
        channel_mult: channel multiplier per resolution level.
        num_res_blocks: residual blocks per level.
        attn_resolutions: spatial sizes (H == W) at which to apply attention.
        dropout: dropout inside ResBlocks. DDPM uses 0.1 for CIFAR-10.
        image_size: input resolution, needed to resolve attn_resolutions.
        num_heads: attention heads used by every SelfAttention layer.
        num_classes: number of classes to condition on, or None for an
            unconditional model. When set, `forward` takes a label per image.

    Raises:
        ValueError: if `image_size` cannot be halved once per level beyond the
            first and still leave a bottleneck of at least 2x2, or if
            `num_classes` is set but not positive.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (16,),
        dropout: float = 0.1,
        image_size: int = 32,
        num_heads: int = 4,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()

        # The decoder concatenates each skip with a nearest-neighbour upsample,
        # which can only undo an exact halving. An image_size that does not
        # divide evenly would otherwise fail deep inside forward() with a
        # torch.cat size mismatch, and a 1x1 bottleneck fails in GroupNorm.
        divisor = 2 ** (len(channel_mult) - 1)
        if image_size % divisor or image_size // divisor < 2:
            raise ValueError(
                f"image_size={image_size} does not fit {len(channel_mult)} levels: it must be a "
                f"multiple of {divisor} and leave a bottleneck of at least 2x2 "
                f"(smallest valid size is {2 * divisor})"
            )

        time_dim = base_channels * TIME_EMBED_MULT
        self.time_embed = TimeEmbedding(base_channels, time_dim)
        self.num_classes = num_classes
        # Summed into the time embedding, so the class rides the FiLM path the
        # ResBlocks already have and the rest of the architecture is untouched.
        self.label_embed = (
            LabelEmbedding(num_classes, time_dim) if num_classes is not None else None
        )
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # the encoder
        self.downs = nn.ModuleList()
        skip_channels = [base_channels]
        channels = base_channels
        resolution = image_size

        for level, mult in enumerate(channel_mult):
            level_channels = base_channels * mult
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [ResBlock(channels, level_channels, time_dim, dropout)]
                channels = level_channels
                if resolution in attn_resolutions:
                    layers.append(SelfAttention(channels, num_heads))
                self.downs.append(TimestepSequential(*layers))
                skip_channels.append(channels)

            is_last_level = level == len(channel_mult) - 1
            if not is_last_level:
                self.downs.append(TimestepSequential(Downsample(channels)))
                skip_channels.append(channels)
                resolution //= 2

        # the bottleneck: stays spatial, no vector collapse
        self.mid = TimestepSequential(
            ResBlock(channels, channels, time_dim, dropout),
            SelfAttention(channels, num_heads),
            ResBlock(channels, channels, time_dim, dropout),
        )

        # the decoder
        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            level_channels = base_channels * mult
            # One extra block per level consumes the skip left by the Downsample.
            for block in range(num_res_blocks + 1):
                layers = [
                    ResBlock(channels + skip_channels.pop(), level_channels, time_dim, dropout)
                ]
                channels = level_channels
                if resolution in attn_resolutions:
                    layers.append(SelfAttention(channels, num_heads))
                if level != 0 and block == num_res_blocks:
                    layers.append(Upsample(channels))
                    resolution *= 2
                self.ups.append(TimestepSequential(*layers))

        self.out = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(channels, out_channels, 3, padding=1)),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the noise in `x`.

        Args:
            x: (B, C, H, W) noisy image.
            t: (B,) integer timesteps.
            y: (B,) integer class labels, where `num_classes` is the null
                token. Omitted on an unconditional model; omitting it on a
                conditional one predicts unconditionally, as if every label
                were null.

        Returns:
            (B, out_channels, H, W) noise prediction.

        Raises:
            ValueError: if labels are passed to an unconditional model.
        """
        time_emb = self.time_embed(t)

        if self.label_embed is not None:
            if y is None:
                y = torch.full(
                    (x.shape[0],), self.label_embed.null_class, device=x.device, dtype=torch.long
                )
            time_emb = time_emb + self.label_embed(y)
        elif y is not None:
            raise ValueError("this UNet was built without num_classes, so it takes no labels")

        h = self.init_conv(x)
        skips = [h]
        for module in self.downs:
            h = module(h, time_emb)
            skips.append(h)

        h = self.mid(h, time_emb)

        for module in self.ups:
            h = module(torch.cat([h, skips.pop()], dim=1), time_emb)

        return self.out(h)
