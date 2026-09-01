"""Compute precision for the sampling path.

Training already has two half-precision strategies — autocast, and the float16
weights with a float32 master copy in :mod:`tinydiffusion.utils.fp16` — and
neither of them reaches the commands that only ever run a forward pass. Those
are where the arithmetic actually is: a ``fid`` over 10,000 images at 50 steps
with guidance is a million network evaluations, and until now every one of them
ran in float32.

Nothing here changes by default. Precision is a measurement setting as much as
a speed one — two checkpoints compared at different precisions are not
comparable — so ``fp32`` remains what a command does when it is not asked
otherwise, and it is bit-for-bit what the same command did before this module
existed. The ladder above it trades accuracy for throughput in steps:

``tf32``
    float32 storage, reduced-mantissa matmul and convolution kernels on
    Ampere-and-later hardware. The cheapest rung, and the one with nothing to
    go wrong: accumulation is still float32.
``fp16``
    Half precision through autocast. The largest speedup on any GPU with
    tensor cores.
``bf16``
    Half precision with float32's exponent range, so it cannot overflow the
    way float16 can. Needs Ampere or later to run in hardware at all.

Half precision also brings the memory format with it. Tensor cores read
NHWC, and cuDNN handed an NCHW half-precision tensor transposes it per
convolution rather than declining to use them — which is most of the speedup,
spent. Measured on a Turing card (RTX 2070) at the ``configs/mnist.toml``
geometry, batch 256:

===============  =========  =========  =======
layout           fp32       fp16       speedup
===============  =========  =========  =======
NCHW             215 ms     193 ms     1.12x
channels_last    360 ms     153 ms     2.36x
===============  =========  =========  =======

So the two are not independent settings to be offered separately: NCHW throws
away half of what half precision is for, and channels_last on its own makes
float32 markedly *worse*. :func:`apply_precision` therefore switches the layout
exactly when it switches the dtype, and never otherwise.

The wrapper goes round the *network* rather than round the sampler. Two
reasons: the sampler protocol in
:mod:`tinydiffusion.diffusion.samplers` is fixed and shared, so a sampler that
had to know about precision would be a change to every implementation of it;
and the network is the only part of a step whose cost is worth reducing. The
schedule arithmetic, the DDIM update and — the one that matters —
classifier-free guidance's extrapolation and rescaling all stay in float32,
where a standard deviation taken over a whole image is not being asked of a
format with ten mantissa bits.
"""

import torch
import torch.nn as nn

from tinydiffusion.utils.device import bf16_supported, enable_tf32

__all__ = [
    "DEFAULT_PRECISION",
    "PRECISIONS",
    "Autocast",
    "apply_precision",
    "resolve_precision",
]

PRECISIONS = ("fp32", "tf32", "fp16", "bf16")
"""The precisions a sampling command accepts, cheapest last to fastest first."""

DEFAULT_PRECISION = "fp32"
"""What every command does unless asked otherwise: exactly what it did before."""

_AUTOCAST_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
"""The two rungs that are an autocast dtype. ``tf32`` is a backend flag instead."""


def resolve_precision(precision: str, device: torch.device | str, *, verbose: bool = True) -> str:
    """Reduce a requested precision to one this device can actually run.

    Every fallback is announced rather than silent. A precision that quietly
    became something else would show up as a score that moved for no stated
    reason, which is the one thing a measurement setting must not do.

    Args:
        precision: one of :data:`PRECISIONS`.
        device: the device the network will run on.
        verbose: print a line when the request is downgraded.

    Returns:
        A member of :data:`PRECISIONS` that `device` supports.

    Raises:
        ValueError: if `precision` does not name one of :data:`PRECISIONS`.
    """
    if precision not in PRECISIONS:
        raise ValueError(
            f"unknown precision {precision!r}, expected one of {', '.join(PRECISIONS)}"
        )
    if precision == DEFAULT_PRECISION:
        return precision

    if torch.device(device).type != "cuda":
        if verbose:
            print(f"{precision} needs a CUDA device; sampling in float32 instead")
        return DEFAULT_PRECISION

    if precision == "bf16" and not bf16_supported():
        if verbose:
            print("this GPU emulates bfloat16 rather than running it, falling back to fp16")
        return "fp16"

    return precision


class Autocast(nn.Module):
    """Run a network under autocast in NHWC, handing its prediction back in float32.

    The cast back is not incidental. Everything downstream of the network — the
    guidance extrapolation, the schedule coefficients, the solver's own update
    — is written for float32 and is a rounding error in cost next to the
    forward pass, so returning half precision would spread the format through
    the parts of a step that gain nothing from it. The layout is dropped on the
    way out for the same reason, leaving the solver the ordinary contiguous
    tensor it had before.

    Args:
        net: the network to wrap. Called with whatever arguments the wrapper
            is called with, so this sits underneath the conditioning wrappers
            in :mod:`tinydiffusion.diffusion.guidance` rather than over them —
            which is what keeps their arithmetic in float32. Its weights are
            converted to channels_last in place; see the module docstring for
            why that travels with the dtype. Values are untouched — a memory
            format is a stride pattern — but a network handed here and then
            sampled in float32 elsewhere would be the slower for it.
        device_type: ``"cuda"`` or ``"cpu"``, for the autocast context.
        dtype: the half-precision dtype to run in.
    """

    def __init__(self, net: nn.Module, device_type: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.net = net.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
        self.device_type = device_type
        self.dtype = dtype
        self.train(net.training)

    def forward(self, x: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
        """Evaluate the network in half precision.

        Args:
            x: ``(B, C, H, W)`` latents, converted to match the weights' layout.
                One transpose of a single tensor per step, against a whole
                network's worth of convolutions that would otherwise each do
                their own.
            *rest: the network's remaining arguments — ``t``, and ``y`` for a
                conditional network. Neither is a feature map, so neither has a
                layout to convert.

        Returns:
            The network's prediction, in float32 and contiguous.
        """
        with torch.autocast(self.device_type, dtype=self.dtype):
            out = self.net(x.contiguous(memory_format=torch.channels_last), *rest)
        return out.float().contiguous()


def apply_precision(net: nn.Module, precision: str, device: torch.device | str) -> nn.Module:
    """Prepare a network to be sampled at `precision`.

    Args:
        net: the network to sample from, typically the EMA weights.
        precision: an already-resolved member of :data:`PRECISIONS`; see
            :func:`resolve_precision`, which is what makes it one this device
            can run.
        device: the device the network will run on.

    Returns:
        The network to hand to a sampler: wrapped in :class:`Autocast` for the
        two half-precision rungs — which also converts `net` to channels_last
        in place — and `net` itself otherwise. ``tf32`` is a backend flag
        rather than a wrapper, so it returns `net` unchanged having set it —
        which is process-wide and stays set, matching what
        :func:`~tinydiffusion.utils.device.enable_tf32` does for a training run.

    Raises:
        ValueError: if `precision` does not name one of :data:`PRECISIONS`.
    """
    if precision not in PRECISIONS:
        raise ValueError(
            f"unknown precision {precision!r}, expected one of {', '.join(PRECISIONS)}"
        )
    if precision == "tf32":
        enable_tf32()
        return net
    dtype = _AUTOCAST_DTYPES.get(precision)
    if dtype is None:
        return net
    return Autocast(net, torch.device(device).type, dtype)
