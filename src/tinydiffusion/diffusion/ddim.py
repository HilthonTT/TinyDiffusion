"""DDIM sampling (Song et al. 2020, https://arxiv.org/abs/2010.02502).

DDIM is a *sampler*, not a different training objective. A model trained with
the DDPM loss can be sampled with either method, so this lives as a function
over a trained process rather than as a subclass that overrides `sample`.
"""

from typing import Protocol

import torch
import torch.nn as nn

from tinydiffusion.diffusion.gaussian_diffusion import Diffusion
from tinydiffusion.diffusion.latents import initial_latent
from tinydiffusion.diffusion.prediction import predict_xstart_eps
from tinydiffusion.utils.modules import eval_mode

__all__ = [
    "DEFAULT_SPACING",
    "SPACINGS",
    "TimestepSpacing",
    "ddim_sample",
    "get_spacing",
    "quadratic_timesteps",
    "spacing_names",
    "uniform_timesteps",
]


def _check_num_steps(num_timesteps: int, num_steps: int) -> None:
    """Raise if `num_steps` cannot index into a `num_timesteps`-long schedule."""
    if not 1 <= num_steps <= num_timesteps:
        raise ValueError(f"num_steps must lie in [1, {num_timesteps}], got {num_steps}")


def uniform_timesteps(num_timesteps: int, num_steps: int) -> torch.Tensor:
    """Evenly spaced subsequence of [0, num_timesteps-1], descending.

    Args:
        num_timesteps: number of steps the model was trained with.
        num_steps: number of sampling steps to actually take.

    Returns:
        Long tensor of length num_steps, strictly decreasing, starting at
        num_timesteps - 1. Ends at 0 whenever more than one step is taken.
    """
    _check_num_steps(num_timesteps, num_steps)
    # Built descending rather than ascending-then-flipped: a one-step schedule
    # has to be [T-1], the timestep matching the pure noise the chain starts
    # from. Ascending would collapse it to [0] and denoise as if x_T were
    # already a clean image. The two agree for every longer schedule.
    return torch.linspace(num_timesteps - 1, 0, num_steps).round().long()


def quadratic_timesteps(num_timesteps: int, num_steps: int) -> torch.Tensor:
    """Quadratically spaced subsequence, denser near t=0.

    The DDIM paper found this better than uniform on CIFAR-10 at low step counts.

    Args:
        num_timesteps: number of steps the model was trained with.
        num_steps: number of sampling steps to actually take.

    Returns:
        Long tensor of descending, de-duplicated timesteps.
    """
    _check_num_steps(num_timesteps, num_steps)
    # Descending for the same reason as `uniform_timesteps`; squaring a
    # descending ramp gives the same set, still denser near t=0.
    steps = (torch.linspace((num_timesteps - 1) ** 0.5, 0, num_steps) ** 2).round().long()
    return torch.unique(steps).flip(0)


class TimestepSpacing(Protocol):
    """How a sampler picks which timesteps of the training schedule to visit."""

    def __call__(self, num_timesteps: int, num_steps: int) -> torch.Tensor:
        """Return a descending subsequence of ``[0, num_timesteps - 1]``."""
        ...


SPACINGS: dict[str, TimestepSpacing] = {
    "uniform": uniform_timesteps,
    "quadratic": quadratic_timesteps,
}
"""Name to spacing. ``uniform`` is the safe default; ``quadratic`` is denser near t=0."""

DEFAULT_SPACING = "uniform"
"""What a config that says nothing gets, and what every sampler used before there was a choice."""


def spacing_names() -> tuple[str, ...]:
    """The registered spacing names.

    Returns:
        The keys of :data:`SPACINGS`, sorted, for error messages and CLI choices.
    """
    return tuple(sorted(SPACINGS))


def get_spacing(name: str) -> TimestepSpacing:
    """Resolve a spacing name to the function that builds the subsequence.

    Args:
        name: a key of :data:`SPACINGS`.

    Returns:
        The spacing function.

    Raises:
        ValueError: if no spacing is registered under that name.
    """
    try:
        return SPACINGS[name]
    except KeyError:
        raise ValueError(
            f"unknown timestep spacing {name!r}, expected one of {', '.join(spacing_names())}"
        ) from None


@torch.no_grad()
def ddim_sample(
    diffusion: Diffusion,
    num_samples: int,
    size: tuple[int, ...],
    device: torch.device | str,
    num_steps: int = 50,
    eta: float = 0.0,
    model: nn.Module | None = None,
    timesteps: torch.Tensor | None = None,
    clip_denoised: bool = True,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    spacing: str = DEFAULT_SPACING,
) -> torch.Tensor:
    """Sample by running the DDIM reverse chain over a timestep subsequence.

    Args:
        diffusion: a trained process, used for its schedule buffers and network.
        num_samples: batch size to generate.
        size: shape of one sample, e.g. (1, 28, 28).
        device: device to generate on.
        num_steps: how many denoising steps to take. Ignored if `timesteps` given.
        eta: 0.0 is deterministic DDIM; 1.0 reproduces DDPM ancestral sampling.
        model: network to sample from. Pass `ema.module` to use EMA weights.
        timesteps: explicit descending subsequence, overriding `num_steps`.
        clip_denoised: clamp the predicted x_0 to [-1, 1] at each step. Matters
            much more at low step counts than it does for full-chain DDPM.
        noise: the starting x_T, of shape ``(num_samples, *size)``. None draws a
            fresh one. Passing it in is what makes a series of grids comparable:
            reusing one latent across epochs shows the same images sharpening,
            where a fresh draw each time shows a different sample of the model.
        generator: RNG for the starting latent and, when `eta` is positive, the
            per-step noise. None uses the global RNG. Passing one is how a
            caller gets a reproducible sample without reseeding the process —
            which matters for anything serving concurrent requests.
        spacing: which subsequence of the training schedule to visit; a key of
            :data:`SPACINGS`. Ignored when `timesteps` is given explicitly.

    Returns:
        Tensor of shape (num_samples, *size).

    Raises:
        ValueError: if `eta` falls outside [0, 1], no spacing goes by that
            name, `noise` is not shaped ``(num_samples, *size)``, or
            `generator` is on another device.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta}")

    net = model if model is not None else diffusion.net

    if timesteps is None:
        timesteps = get_spacing(spacing)(diffusion.num_timesteps, num_steps)
    ts = timesteps.to(device)
    # Pair each t with its predecessor; the last step lands on the t=-1 sentinel,
    # for which alphabar is defined as 1 (a noise-free x_0).
    ts_prev = torch.cat([ts[1:], ts.new_tensor([-1])])

    alphabar = diffusion.alphabar_t
    one = alphabar.new_ones(())

    with eval_mode(net):
        x = initial_latent(num_samples, size, device, noise=noise, generator=generator)

        for t_cur, t_prev in zip(ts, ts_prev, strict=True):
            ab_t = alphabar[t_cur]
            ab_prev = one if t_prev < 0 else alphabar[t_prev]

            t_batch = t_cur.repeat(num_samples)

            # DDIM's own variance is used whatever the process says: a learned
            # reverse variance describes the full-chain step, not this strided
            # one.
            x0, eps = predict_xstart_eps(
                diffusion, x, t_batch, model=net, clip_denoised=clip_denoised
            )

            # sigma_t from DDIM Eq. 16.
            sigma = eta * (((1 - ab_prev) / (1 - ab_t)) * (1 - ab_t / ab_prev)).sqrt()
            direction = (1 - ab_prev - sigma**2).clamp(min=0.0).sqrt()

            x = ab_prev.sqrt() * x0 + direction * eps
            if eta > 0 and t_prev >= 0:
                # randn_like takes no generator, so the shape is spelled out.
                step_noise = torch.randn(
                    x.shape, device=x.device, dtype=x.dtype, generator=generator
                )
                x = x + sigma * step_noise

    return x
