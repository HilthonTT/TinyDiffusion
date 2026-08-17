"""Learning-rate schedule for the training loop.

One function, :func:`lr_factor`, shaped for ``LambdaLR``: it returns a
multiplier on ``cfg.lr`` rather than a rate, and is stepped once per *applied*
optimiser step so the ramp counts real updates rather than batches AMP threw
away.

Not to be confused with :mod:`tinydiffusion.diffusion.schedules`, which is the
noise schedule the forward process uses.
"""

import math

__all__ = ["lr_factor"]


def _warmup_lr(step: int, warmup: int) -> float:
    """Linear LR ramp-in factor, for LambdaLR.

    Stepped per optimiser step, not per epoch.

    Args:
        step: optimiser steps completed.
        warmup: steps to ramp over. 0 disables the ramp.

    Returns:
        A multiplier on ``cfg.lr`` in [0, 1].
    """
    if warmup <= 0:
        return 1.0
    return min(step, warmup) / warmup


def _cosine_lr(step: int, warmup: int, total: int) -> float:
    """Cosine decay factor over the steps that follow the warmup ramp.

    Args:
        step: optimiser steps completed.
        warmup: steps the ramp covers, which the decay starts after.
        total: optimiser steps the whole run will take.

    Returns:
        A multiplier in [0, 1], 1 at the end of the ramp and 0 at the last step.
    """
    decaying = total - warmup
    if decaying <= 0:
        # A run shorter than its own warmup never reaches the decay; the ramp
        # is the whole schedule and this must not divide by zero.
        return 1.0
    progress = min(max(step - warmup, 0) / decaying, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def lr_factor(step: int, *, warmup: int, total: int, schedule: str) -> float:
    """The multiplier on ``cfg.lr`` at a given optimiser step, for LambdaLR.

    Args:
        step: optimiser steps completed.
        warmup: steps to ramp over. 0 disables the ramp.
        total: optimiser steps the whole run will take, which the cosine decay
            is measured against.
        schedule: ``"constant"`` to hold ``lr`` after the ramp, ``"cosine"`` to
            decay it to zero over what remains.

    Returns:
        A multiplier on ``cfg.lr`` in [0, 1].
    """
    factor = _warmup_lr(step, warmup)
    if schedule == "cosine":
        # Multiplied rather than branched: during the ramp the cosine term is
        # still 1, so the two compose without a discontinuity where they meet.
        factor *= _cosine_lr(step, warmup, total)
    return factor
