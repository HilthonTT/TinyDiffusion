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
    "KARRAS_RHO",
    "SPACINGS",
    "TimestepSpacing",
    "ddim_sample",
    "get_spacing",
    "karras_timesteps",
    "quadratic_timesteps",
    "schedule_sigmas",
    "spacing_names",
    "uniform_timesteps",
]

KARRAS_RHO = 7.0
"""Curvature of the Karras ramp. 7 is the value the EDM paper settled on."""


def _check_num_steps(num_timesteps: int, num_steps: int) -> None:
    """Raise if `num_steps` cannot index into a `num_timesteps`-long schedule."""
    if not 1 <= num_steps <= num_timesteps:
        raise ValueError(f"num_steps must lie in [1, {num_timesteps}], got {num_steps}")


def uniform_timesteps(
    num_timesteps: int, num_steps: int, *, alphabar: torch.Tensor | None = None
) -> torch.Tensor:
    """Evenly spaced subsequence of [0, num_timesteps-1], descending.

    Args:
        num_timesteps: number of steps the model was trained with.
        num_steps: number of sampling steps to actually take.
        alphabar: unused. This spacing is defined on the index rather than on
            the noise level, so it is the same subsequence whatever schedule
            produced it; the argument is part of the shared
            :class:`TimestepSpacing` signature.

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


def quadratic_timesteps(
    num_timesteps: int, num_steps: int, *, alphabar: torch.Tensor | None = None
) -> torch.Tensor:
    """Quadratically spaced subsequence, denser near t=0.

    The DDIM paper found this better than uniform on CIFAR-10 at low step counts.

    Args:
        num_timesteps: number of steps the model was trained with.
        num_steps: number of sampling steps to actually take.
        alphabar: unused; see :func:`uniform_timesteps`.

    Returns:
        Long tensor of descending, de-duplicated timesteps.
    """
    _check_num_steps(num_timesteps, num_steps)
    # Descending for the same reason as `uniform_timesteps`; squaring a
    # descending ramp gives the same set, still denser near t=0.
    steps = (torch.linspace((num_timesteps - 1) ** 0.5, 0, num_steps) ** 2).round().long()
    return torch.unique(steps).flip(0)


def schedule_sigmas(alphabar: torch.Tensor) -> torch.Tensor:
    """Noise level of each timestep, as a variance-exploding sigma.

    ``x_t = sqrt(abar) * x_0 + sqrt(1 - abar) * eps`` divided through by its own
    signal coefficient is ``x_0 + sigma * eps`` with
    ``sigma = sqrt((1 - abar) / abar)``. That is the quantity the Karras
    schedule is written in, and it is what makes "space the steps evenly in
    noise" a different instruction from "space them evenly in t".

    Args:
        alphabar: the schedule's cumulative alpha product, length
            ``num_timesteps``, ascending in t from nearly 1 to nearly 0.

    Returns:
        Sigma per timestep, ascending. The last entry is ``inf`` on a schedule
        whose terminal SNR is exactly zero.
    """
    return ((1.0 - alphabar) / alphabar).sqrt()


def karras_timesteps(
    num_timesteps: int,
    num_steps: int,
    *,
    alphabar: torch.Tensor | None = None,
    rho: float = KARRAS_RHO,
) -> torch.Tensor:
    """Subsequence spaced evenly in noise level rather than in t (Karras et al., 2022).

    ``uniform`` and ``quadratic`` both space the steps by index. This one takes
    the EDM ramp,
    ``sigma_i = (sigma_max^(1/rho) + i/(n-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho``,
    and maps each sigma back to the timestep that carries it, interpolating in
    log-sigma. The steps then land where the *noise* changes evenly, which is
    the thing the denoiser actually sees.

    .. warning::
        **It does not honour `num_steps` on a cosine schedule.** Cosine ends at
        ``abar ~ 2e-9``, a sigma of about 20,000 against the 80 the EDM ramp
        was designed around, so a large part of the ramp lands in the handful
        of timesteps above ``t = 936`` and rounding back to integers collapses
        them: 20 requested steps come back as 12, and 40 as 22. Ask for roughly
        double what you want, and read the count off the timesteps rather than
        assuming it. On ``schedule = "linear"``, whose terminal sigma is about
        157, 20 requested steps are 20.

    On the shipped MNIST model (cosine, 1,000 images, KID with its spread over
    50 subsets of 500) it sits between the two index spacings — better than
    ``uniform`` at the same number of network evaluations, worse than
    ``quadratic``:

    ==== ================= ================= =================
    NFEs uniform           quadratic         karras
    ==== ================= ================= =================
    12   0.01360 +- 0.0012 0.00452 +- 0.0005 0.00963 +- 0.0008
    ~20  0.00720 +- 0.0008 0.00273 +- 0.0004 0.00471 +- 0.0006
    ==== ================= ================= =================

    Every gap there is several times its own error bar, so the ordering is
    real for this model. It is one model on one dataset: ``quadratic`` winning
    on MNIST is not a claim about anything else, and the point of ``fid --kid``
    is that you can check rather than believe.

    Args:
        num_timesteps: number of steps the model was trained with.
        num_steps: number of sampling steps to actually take.
        alphabar: the schedule's cumulative alpha product. Required: unlike the
            index-based spacings this one is a function of the noise levels, so
            there is nothing sensible to fall back on.
        rho: curvature of the ramp. Higher concentrates more of it near
            ``sigma_min``.

    Returns:
        Long tensor of descending, de-duplicated timesteps, starting at
        ``num_timesteps - 1``. Shorter than `num_steps` whenever two of the
        ramp's sigmas round to the same timestep.

    Raises:
        ValueError: if `alphabar` is not given, does not match `num_timesteps`,
            or `num_steps` is out of range.
    """
    _check_num_steps(num_timesteps, num_steps)
    if alphabar is None:
        raise ValueError(
            "karras spacing is defined on the schedule's noise levels, so it needs "
            "alphabar; the index-based spacings are 'uniform' and 'quadratic'"
        )
    if alphabar.numel() != num_timesteps:
        raise ValueError(
            f"alphabar has {alphabar.numel()} entries, expected num_timesteps={num_timesteps}"
        )

    # On the host, whatever device the schedule lives on: the other spacings
    # are pure index arithmetic and hand back a CPU tensor, and a subsequence
    # of 20 integers is not worth a device round trip to disagree about.
    sigmas = schedule_sigmas(alphabar.detach().cpu()).double()
    # A zero terminal SNR puts an infinity at the top of the schedule. The ramp
    # is anchored to the largest sigma that is a number, and the first timestep
    # is restored below, so the chain still starts from the pure-noise step.
    finite = sigmas.isfinite()
    sigma_min = sigmas[finite].min().clamp_min(torch.finfo(torch.float64).tiny)
    sigma_max = sigmas[finite].max()

    ramp = torch.linspace(0, 1, num_steps, dtype=torch.float64)
    low, high = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    targets = (high + ramp * (low - high)) ** rho

    # Interpolated in log-sigma, where the schedule is close to linear in t and
    # so the inverse is well conditioned across four decades of noise.
    log_sigmas = sigmas[finite].log()
    index = torch.arange(num_timesteps, dtype=torch.float64)[finite]
    slot = torch.searchsorted(log_sigmas, targets.log()).clamp(1, log_sigmas.numel() - 1)
    before, after = slot - 1, slot
    span = (log_sigmas[after] - log_sigmas[before]).clamp_min(1e-12)
    weight = ((targets.log() - log_sigmas[before]) / span).clamp(0, 1)
    steps = (index[before] + weight * (index[after] - index[before])).round().long()

    # The chain starts from pure noise whatever the ramp's top sigma rounded
    # to, which is the invariant the other two spacings hold by construction.
    steps[0] = num_timesteps - 1
    return torch.unique(steps.clamp(0, num_timesteps - 1)).flip(0)


class TimestepSpacing(Protocol):
    """How a sampler picks which timesteps of the training schedule to visit."""

    def __call__(
        self, num_timesteps: int, num_steps: int, *, alphabar: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return a descending subsequence of ``[0, num_timesteps - 1]``.

        Args:
            num_timesteps: length of the training schedule.
            num_steps: how many steps to take.
            alphabar: the schedule's cumulative alpha product, for the spacings
                defined on noise level rather than on index. Those defined on
                index accept and ignore it.
        """
        ...


SPACINGS: dict[str, TimestepSpacing] = {
    "uniform": uniform_timesteps,
    "quadratic": quadratic_timesteps,
    "karras": karras_timesteps,
}
"""Name to spacing.

``uniform`` is the safe default and ``quadratic`` is denser near t=0; both are
defined on the index. ``karras`` is defined on the noise level instead, and is
the one that needs the schedule handed to it — see :func:`karras_timesteps` for
where it does and does not work.
"""

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
        timesteps = get_spacing(spacing)(
            diffusion.num_timesteps, num_steps, alphabar=diffusion.alphabar_t
        )
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
