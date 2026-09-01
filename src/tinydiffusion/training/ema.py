"""Exponential moving average of model weights."""

import copy
from collections.abc import Sequence

import torch
import torch.nn as nn


class EMA:
    """Exponential moving average of model weights.

    DDPM's published sample quality depends on this. Sample from `ema.module`,
    not from the live model.

    Args:
        model: the live model whose weights are averaged.
        decay: target decay rate applied once past `warmup`.
        warmup: number of steps over which the decay is ramped in.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup: int = 0) -> None:
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self._params: list[torch.Tensor] = list(self.module.parameters())
        self._buffers: list[torch.Tensor] = list(self.module.buffers())

    @property
    def current_decay(self) -> float:
        """The decay in force at the current step.

        Ramped in over `warmup` so early averages are not dominated by the
        random initialisation. Worth logging: an EMA still deep in its warmup
        explains sample grids that lag the training loss.

        Returns:
            The decay that the next :meth:`update` will apply.
        """
        step = self.step + 1
        if step <= self.warmup:
            return min(self.decay, (1 + step) / (10 + step))
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module, params: Sequence[torch.Tensor] | None = None) -> None:
        """Fold one optimiser step of `model` into the averaged weights.

        The parameter average goes through ``torch._foreach_lerp_``, which is
        the same arithmetic as a ``lerp_`` per parameter but launches a handful
        of fused kernels instead of one per tensor. At a few hundred parameter
        tensors and one call per optimiser step, the launch overhead is
        otherwise a measurable slice of a small model's step time.

        Args:
            model: the live model, after its parameter update. Its buffers are
                copied across either way.
            params: the weights to average in, in
                :meth:`~torch.nn.Module.parameters` order, when `model`'s own
                are not the ones to read. That is the
                :attr:`~tinydiffusion.training.config.TrainConfig.full_fp16`
                case: the model holds float16 weights there, and averaging at a
                decay of 0.9999 means folding in a change four orders of
                magnitude smaller than the value it lands on, which float16
                cannot represent at all — the average would simply stop moving.
                Defaults to `model`'s parameters.

        Raises:
            ValueError: if the weights do not match the module the average was
                built from, in count or in dtype.
        """
        decay = self.current_decay
        self.step += 1

        live: list[torch.Tensor] = list(model.parameters() if params is None else params)
        if len(live) != len(self._params):
            raise ValueError(
                f"model has {len(live)} parameter tensors, but this EMA was built from "
                f"{len(self._params)}"
            )
        pairs = enumerate(zip(self._params, live, strict=True))
        mismatched = next((i for i, (held, new) in pairs if held.dtype != new.dtype), None)
        if mismatched is not None:
            raise ValueError(
                f"parameter {mismatched} is {live[mismatched].dtype} but this EMA holds "
                f"{self._params[mismatched].dtype}; pass full-precision weights as `params` "
                f"when the model itself is a reduced-precision copy"
            )
        torch._foreach_lerp_(self._params, live, 1.0 - decay)

        for ema_b, b in zip(self._buffers, model.buffers(), strict=True):
            ema_b.copy_(b)
