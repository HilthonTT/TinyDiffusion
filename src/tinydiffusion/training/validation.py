"""Held-out scoring, shared by the training loop and the ``eval`` command.

Training draws a random timestep and fresh noise per image, so a single loss
value moves for reasons that have nothing to do with the weights. Everything
here pins both: a fixed grid of timesteps, and a fresh draw of noise per
(batch, timestep) taken from a dedicated generator seeded the same way every
call. What is left varies only with the model, which is what makes two epochs
— or two checkpoints — comparable.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn

from tinydiffusion.diffusion.gaussian_diffusion import Diffusion
from tinydiffusion.diffusion.guidance import Conditioned
from tinydiffusion.utils.modules import eval_mode

__all__ = ["DEFAULT_VAL_STEPS", "eval_timesteps", "validation_loss"]

DEFAULT_VAL_STEPS = 10
"""Timesteps to score at. Enough to cover the schedule without being slow."""


def eval_timesteps(num_timesteps: int, num_steps: int) -> torch.Tensor:
    """Evenly spaced timesteps to score at, ascending.

    Args:
        num_timesteps: length of the model's schedule.
        num_steps: how many timesteps to score.

    Returns:
        Long tensor of length ``num_steps``.

    Raises:
        ValueError: if ``num_steps`` cannot index the schedule.
    """
    if not 1 <= num_steps <= num_timesteps:
        raise ValueError(f"num_steps must lie in [1, {num_timesteps}], got {num_steps}")
    return torch.linspace(0, num_timesteps - 1, num_steps).round().long()


@torch.no_grad()
def validation_loss(
    diffusion: Diffusion,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    model: nn.Module,
    num_classes: int | None = None,
    device: torch.device | str = "cpu",
    num_steps: int = DEFAULT_VAL_STEPS,
    seed: int = 0,
) -> float:
    """Score `model` on held-out batches at a fixed grid of timesteps.

    The noise comes from a private CPU generator rather than the global RNG.
    Reseeding the global stream mid-run — which is what the ``eval`` command
    can afford to do — would rewind the training loop's own randomness and
    make every epoch after the first draw the same batches of noise.

    Args:
        diffusion: the process being trained, used for its objective.
        batches: held-out ``(images, labels)`` pairs. Images are in [-1, 1] and
            may live on any device; they are moved to `device` per batch.
        model: network to score. Pass ``ema.module`` for the EMA weights.
        num_classes: the run's class count, or None when unconditional. A
            conditional model is scored on the true labels and never with
            guidance: the conditional prediction is what the run optimised.
        device: device to score on.
        num_steps: how many timesteps to score at.
        seed: seed for the noise generator. Fixed across calls, so the same
            sequence of draws is replayed every time and the only thing that
            changes between epochs is the weights.

    Returns:
        The mean loss over every batch and timestep, weighted by image count.

    Raises:
        ValueError: if `batches` is empty, or `num_steps` cannot index the
            schedule.
    """
    if not batches:
        raise ValueError("validation needs at least one batch")

    steps = eval_timesteps(diffusion.num_timesteps, num_steps).to(device)
    generator = torch.Generator().manual_seed(seed)
    total = 0.0
    num_images = 0

    with eval_mode(model):
        for x, y in batches:
            x = x.to(device)
            scored = Conditioned(model, y.to(device)) if num_classes is not None else model
            for step in steps:
                t = step.expand(x.shape[0])
                noise = torch.randn(x.shape, generator=generator).to(device)
                total += float(diffusion.loss_at(x, t, noise=noise, model=scored)) * x.shape[0]
            num_images += x.shape[0]

    return total / (num_images * len(steps))
