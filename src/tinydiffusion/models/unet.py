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
from tinydiffusion.utils.fp16 import convert_module_to_f16, convert_module_to_f32

# DDPM projects the sinusoidal embedding to 4x the base width before conditioning.
TIME_EMBED_MULT = 4


class UNet(nn.Module):
    """DDPM UNet.

    The network is built in float32 and stays there unless
    :meth:`convert_to_fp16` is called; see it for what a half-precision network
    keeps in full precision, and why.

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
        use_checkpoint: trade compute for memory by recomputing each ResBlock
            and attention layer during the backward pass instead of holding
            its activations. See
            :func:`~tinydiffusion.models.blocks._checkpointed`.

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
        use_checkpoint: bool = False,
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

        self.use_checkpoint = use_checkpoint
        # Flipped by convert_to_fp16(). forward() reads it to meet the
        # convolutions in whatever precision they are currently holding.
        self.dtype = torch.float32

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
                layers: list[nn.Module] = [
                    ResBlock(channels, level_channels, time_dim, dropout, use_checkpoint)
                ]
                channels = level_channels
                if resolution in attn_resolutions:
                    layers.append(SelfAttention(channels, num_heads, use_checkpoint))
                self.downs.append(TimestepSequential(*layers))
                skip_channels.append(channels)

            is_last_level = level == len(channel_mult) - 1
            if not is_last_level:
                self.downs.append(TimestepSequential(Downsample(channels)))
                skip_channels.append(channels)
                resolution //= 2

        # the bottleneck: stays spatial, no vector collapse
        self.mid = TimestepSequential(
            ResBlock(channels, channels, time_dim, dropout, use_checkpoint),
            SelfAttention(channels, num_heads, use_checkpoint),
            ResBlock(channels, channels, time_dim, dropout, use_checkpoint),
        )

        # the decoder
        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            level_channels = base_channels * mult
            # One extra block per level consumes the skip left by the Downsample.
            for block in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        channels + skip_channels.pop(),
                        level_channels,
                        time_dim,
                        dropout,
                        use_checkpoint,
                    )
                ]
                channels = level_channels
                if resolution in attn_resolutions:
                    layers.append(SelfAttention(channels, num_heads, use_checkpoint))
                if level != 0 and block == num_res_blocks:
                    layers.append(Upsample(channels))
                    resolution *= 2
                self.ups.append(TimestepSequential(*layers))

        self.out = nn.Sequential(
            group_norm(channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(channels, out_channels, 3, padding=1)),
        )

    def _convolutional_stack(self) -> tuple[nn.Module, ...]:
        """The parts of the network whose precision `convert_to_fp16` changes.

        Everything else — the timestep and label embeddings, the FiLM
        projections inside each ResBlock, every GroupNorm, and the output head
        — is deliberately left in float32. Between them they are a rounding
        error in both parameter count and FLOPs, and they are exactly the
        places where half precision costs accuracy: sums over a whole channel
        group, a table lookup that is added to everything downstream, and the
        final projection whose output the loss is taken on.

        Returns:
            The modules to hand to
            :func:`~tinydiffusion.utils.fp16.convert_module_to_f16`.
        """
        return (self.init_conv, self.downs, self.mid, self.ups)

    def convert_to_fp16(self) -> None:
        """Put the convolutional stack into float16, in place.

        Half the weight memory and half the activation memory, with the
        convolutions running as float16 kernels end to end rather than being
        cast per operation the way autocast does it. What makes it trainable is
        that the optimiser keeps stepping a float32 copy of these weights; see
        :mod:`tinydiffusion.utils.fp16`, and do not call this without that copy.
        """
        for module in self._convolutional_stack():
            module.apply(convert_module_to_f16)
        self.dtype = torch.float16

    def convert_to_fp32(self) -> None:
        """Put the convolutional stack back into float32, undoing `convert_to_fp16`.

        The dtype round trip does not restore the mantissa bits float16 threw
        away, so this is for handing the network back to code that expects an
        ordinary float32 model — the samplers, the EMA copy at the end of a run
        — rather than for recovering precision.
        """
        for module in self._convolutional_stack():
            module.apply(convert_module_to_f32)
        self.dtype = torch.float32

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

        # A no-op unless convert_to_fp16 has run. The time embedding stays in
        # float32 either way: each ResBlock casts it down as it applies it.
        h = self.init_conv(x.to(self.dtype))
        skips = [h]
        for module in self.downs:
            h = module(h, time_emb)
            skips.append(h)

        h = self.mid(h, time_emb)

        for module in self.ups:
            h = module(torch.cat([h, skips.pop()], dim=1), time_emb)

        if self.dtype is not torch.float32:
            # The output head kept its float32 weights, so the cast back
            # happens here. Guarded rather than unconditional: under autocast
            # `h` is already half and forcing it up would only make autocast
            # cast it down again for the final convolution.
            h = h.float()
        return self.out(h)
