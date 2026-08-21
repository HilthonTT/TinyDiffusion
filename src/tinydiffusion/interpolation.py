"""Walk between two latents and sample every point on the way.

A grid of samples says what a model can draw. A walk between two of them says
something the grid cannot: whether the space in between is *populated*. Every
sampler here is a deterministic function of its starting latent when ``eta`` is
0, so a smooth path through that latent space produces a smooth path through
image space — if the model has learned one. Where it has not, the walk snaps
from one digit to another partway across, and that discontinuity is the model
telling you it memorised two modes and nothing between them.

The path is a *spherical* interpolation, not a straight line. Latents are drawn
from an isotropic Gaussian, where almost all the mass sits in a thin shell at
radius ``sqrt(d)`` — 55.4 for a 1x32x32 image. The midpoint of a straight line
between two such vectors has an expected norm of about ``0.71 * sqrt(d)``, well
inside the shell and far away from anything the model was trained on, so a
linear walk washes out in the middle and comes back. Slerp travels along the
shell and every point on it is as plausible a draw as the two ends.
"""

from collections.abc import Sequence
from pathlib import Path

import torch
from torchvision.utils import save_image

from tinydiffusion.data.datasets import denormalize
from tinydiffusion.diffusion.guidance import conditioned
from tinydiffusion.diffusion.samplers import get_sampler
from tinydiffusion.sampling import load_for_sampling
from tinydiffusion.utils.seed import seed_everything

__all__ = ["interpolate_from_checkpoint", "latent_walk", "slerp"]


def slerp(start: torch.Tensor, end: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two latents.

    Args:
        start: the latent at ``weight = 0``, of any shape.
        end: the latent at ``weight = 1``, the same shape as `start`.
        weight: ``(steps,)`` positions along the path. Values outside [0, 1]
            extrapolate, which is meaningful on the sphere but leaves the
            region the model was trained on.

    Returns:
        ``(steps, *start.shape)``, the path from `start` to `end`.

    Raises:
        ValueError: if the two latents disagree on shape, or `weight` is not
            one-dimensional.
    """
    if start.shape != end.shape:
        raise ValueError(f"latents differ in shape: {tuple(start.shape)} and {tuple(end.shape)}")
    if weight.ndim != 1:
        raise ValueError(f"weight must be 1-D, got shape {tuple(weight.shape)}")

    flat_start, flat_end = start.flatten().double(), end.flatten().double()
    unit_start = flat_start / flat_start.norm().clamp_min(1e-12)
    unit_end = flat_end / flat_end.norm().clamp_min(1e-12)
    # Clamped because a dot product of two unit vectors can leave [-1, 1] by an
    # ulp, and acos answers NaN rather than 0 when it does.
    omega = (unit_start * unit_end).sum().clamp(-1.0, 1.0).acos()

    steps = weight.double().to(flat_start.device)
    sin_omega = omega.sin()
    if sin_omega.abs() < 1e-7:
        # Parallel or antiparallel: the great circle through them is not
        # unique, and the spherical formula is 0/0. A straight line is the
        # limit of slerp as the angle closes, so it is the right answer here
        # rather than a fallback.
        path = torch.outer(1.0 - steps, flat_start) + torch.outer(steps, flat_end)
    else:
        path = torch.outer(((1.0 - steps) * omega).sin() / sin_omega, flat_start) + torch.outer(
            (steps * omega).sin() / sin_omega, flat_end
        )
    return path.reshape(-1, *start.shape).to(start.dtype)


def latent_walk(
    size: tuple[int, ...],
    steps: int,
    *,
    device: torch.device | str,
    seed_start: int,
    seed_end: int,
) -> torch.Tensor:
    """Build the path between the latents two seeds draw.

    Seeds rather than tensors, because a seed is what a caller can write down
    and get back: the same pair reproduces the same walk on any machine.

    Args:
        size: shape of one latent, e.g. ``(1, 32, 32)``.
        steps: how many points on the path, counting both ends. Two gives just
            the endpoints.
        device: device to build on.
        seed_start: seed for the latent at the start of the walk.
        seed_end: seed for the latent at the end of it.

    Returns:
        ``(steps, *size)`` latents, the first from `seed_start` and the last
        from `seed_end`.

    Raises:
        ValueError: if `steps` is below 2.
    """
    if steps < 2:
        raise ValueError(f"a walk needs at least its two ends, got steps={steps}")

    # Drawn on the CPU whichever device they will be used on, so a seed names
    # the same latent on a GPU machine and a CPU one.
    ends = [
        torch.randn(size, generator=torch.Generator().manual_seed(seed))
        for seed in (seed_start, seed_end)
    ]
    path = slerp(ends[0], ends[1], torch.linspace(0, 1, steps))
    return path.to(device)


@torch.no_grad()
def interpolate_from_checkpoint(
    checkpoint: Path,
    out: Path,
    *,
    steps: int = 8,
    num_steps: int | None = None,
    sampler: str | None = None,
    spacing: str | None = None,
    labels: Sequence[int] | None = None,
    guidance: float | None = None,
    guidance_rescale: float | None = None,
    seed_start: int = 0,
    seed_end: int = 1,
    device: str | None = None,
) -> Path:
    """Sample a walk between two latents and write it as a strip.

    Sampling is deterministic here — ``eta`` is fixed at 0 — because a walk
    whose points each drew their own noise would vary for two reasons at once
    and show neither. Every difference along the strip is then the latent's.

    Args:
        checkpoint: file to sample from.
        out: image path to write. One row, oldest to newest, left to right.
        steps: points along the walk, counting both ends.
        num_steps: denoising steps per image. Defaults to the checkpoint's.
        sampler: which sampler to draw with, or None for the checkpoint's own.
        spacing: which timestep spacing to visit, or None for the checkpoint's.
        labels: the class to hold fixed across the walk. A single label is the
            useful case — the walk then shows what varies *within* a class.
            More than one is cycled over the strip, which mixes the two
            variables and is rarely what you want. None on a conditional
            checkpoint holds class 0.
        guidance: classifier-free guidance scale, or None for the checkpoint's.
        guidance_rescale: guidance rescale factor, or None for the checkpoint's.
        seed_start: seed for the latent the walk starts at.
        seed_end: seed for the latent it ends at.
        device: device to sample on. Defaults to CUDA when available.

    Returns:
        The path that was written.

    Raises:
        ValueError: if `steps` is below 2, guidance is asked of an
            unconditional checkpoint, or no sampler or spacing goes by the
            name given.
    """
    if steps < 2:
        raise ValueError(f"a walk needs at least its two ends, got steps={steps}")

    diffusion, ema, cfg = load_for_sampling(checkpoint, device)
    if guidance is not None and cfg.num_classes is None and guidance != 1.0:
        raise ValueError("this checkpoint is unconditional, so guidance does not apply")
    if labels is not None and cfg.num_classes is None:
        raise ValueError("this checkpoint is unconditional, so it cannot be given labels")

    size = (cfg.dataset_spec().channels, cfg.image_size, cfg.image_size)
    noise = latent_walk(size, steps, device=cfg.device, seed_start=seed_start, seed_end=seed_end)

    y = None
    if cfg.num_classes is not None:
        # One label per point, holding the class fixed so the strip has exactly
        # one thing moving along it.
        asked = torch.tensor(labels or [0], dtype=torch.long, device=cfg.device)
        out_of_range = sorted({int(v) for v in asked if not 0 <= v < cfg.num_classes})
        if out_of_range:
            raise ValueError(
                f"label(s) {', '.join(map(str, out_of_range))} outside "
                f"[0, {cfg.num_classes - 1}] for this checkpoint"
            )
        y = asked[torch.arange(steps, device=cfg.device) % asked.shape[0]]

    scale = cfg.guidance if guidance is None else guidance
    rescale = cfg.guidance_rescale if guidance_rescale is None else guidance_rescale

    # The global RNG still decides anything the sampler draws for itself; at
    # eta=0 nothing does, but a sampler asked for eta > 0 elsewhere would.
    seed_everything(seed_start)
    images = get_sampler(cfg.sampler if sampler is None else sampler)(
        diffusion,
        steps,
        size,
        cfg.device,
        num_steps=num_steps if num_steps is not None else cfg.sample_steps,
        eta=0.0,
        model=conditioned(ema.module, y, num_classes=cfg.num_classes, scale=scale, rescale=rescale),
        noise=noise,
        spacing=cfg.sample_spacing if spacing is None else spacing,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    # One row: the walk is a sequence, and wrapping it would put the two ends
    # of a jump in different corners of the image.
    save_image(denormalize(images), out, nrow=steps)
    return out
