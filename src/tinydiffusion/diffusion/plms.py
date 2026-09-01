"""PLMS sampling (Liu et al. 2022, https://arxiv.org/abs/2202.09778).

DDIM takes a first-order step along a direction it measures once.
:mod:`~tinydiffusion.diffusion.heun` buys a second order by measuring the
direction twice. PLMS buys it — and two more — by *remembering*: the noise
estimates from the last few steps are already paid for, and a linear multistep
formula fits a cubic through them to extrapolate where the next step should go.
Fourth-order accuracy at one network evaluation per step, which is the cheapest
order on offer here.

What it costs instead is history. The first three steps have nothing to
extrapolate from, so the order ramps 1, 2, 3, 4 as the buffer fills; those early
steps are where a short chain is least accurate, and they are exactly the ones
this cannot help. Below about 15 steps that ramp eats most of the benefit, and
:mod:`~tinydiffusion.diffusion.dpm_solver` — second order from its second step —
is the better bet. Above it PLMS pulls ahead.

.. note::
    The Adams-Bashforth coefficients assume a *uniform* step size, which is
    what ``spacing = "uniform"`` gives on the log-SNR-agnostic index the
    formula is written over. ``quadratic`` and ``karras`` still work and still
    beat first order, but the coefficients are then only approximately right,
    and the two effects — better-placed steps, slightly mis-weighted history —
    do not compose predictably. Measure with ``fid --kid`` rather than assume.

Like DDIM this is a sampler, not a training objective: any checkpoint this
project produces can be drawn from with it.
"""

import torch
import torch.nn as nn

from tinydiffusion.diffusion.ddim import DEFAULT_SPACING, get_spacing
from tinydiffusion.diffusion.gaussian_diffusion import Diffusion
from tinydiffusion.diffusion.latents import initial_latent
from tinydiffusion.diffusion.prediction import predict_xstart_eps
from tinydiffusion.utils.modules import eval_mode

__all__ = ["PLMS_COEFFICIENTS", "plms_sample"]

PLMS_COEFFICIENTS: tuple[tuple[float, ...], ...] = (
    (1.0,),
    (3 / 2, -1 / 2),
    (23 / 12, -16 / 12, 5 / 12),
    (55 / 24, -59 / 24, 37 / 24, -9 / 24),
)
"""Adams-Bashforth weights, indexed by how much history is available.

Entry ``k`` weights the current noise estimate and the ``k`` before it, so the
solver runs at order ``k + 1``. The first entry is a plain DDIM step, and the
last is the fourth-order formula PLMS is named for; the two in between are the
ramp that fills the buffer.
"""

_MAX_HISTORY = len(PLMS_COEFFICIENTS) - 1
"""Past noise estimates the highest-order formula reads."""


@torch.inference_mode()
def plms_sample(
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
    """Sample with the pseudo linear multistep solver.

    Signature-compatible with
    :func:`~tinydiffusion.diffusion.ddim.ddim_sample`, so the solvers are
    interchangeable through
    :data:`~tinydiffusion.diffusion.samplers.SAMPLERS`.

    Args:
        diffusion: a trained process, used for its schedule buffers and network.
        num_samples: batch size to generate.
        size: shape of one sample, e.g. ``(1, 28, 28)``.
        device: device to generate on.
        num_steps: how many denoising steps to take, one network evaluation
            each. Ignored if `timesteps` is given. The order ramp needs four
            steps to reach full order, so this is worth little below about 15.
        eta: accepted only as 0. The multistep formula extrapolates a
            deterministic trajectory; injecting noise between steps would make
            the history it reads describe a path the chain did not take.
        model: network to sample from. Pass ``ema.module`` to use EMA weights.
        timesteps: explicit descending subsequence, overriding `num_steps`.
        clip_denoised: clamp the predicted x_0 to [-1, 1] at each evaluation.
            The clamp is applied to what goes *into* the history, and the noise
            estimate stored is the one consistent with the clamped image; the
            extrapolated estimate the step is taken along is not clamped again,
            since a linear combination of four in-range predictions has no
            single x_0 to clamp.
        noise: the starting x_T, of shape ``(num_samples, *size)``. None draws
            a fresh one.
        generator: RNG for the starting latent. None uses the global RNG. The
            chain itself is deterministic, so this is the only draw there is.
        spacing: which subsequence of the training schedule to visit; a key of
            :data:`~tinydiffusion.diffusion.ddim.SPACINGS`. See the module
            docstring for why a non-uniform one is only approximately right.
            Ignored when `timesteps` is given.

    Returns:
        Tensor of shape ``(num_samples, *size)``.

    Raises:
        ValueError: if `eta` is not 0, no spacing goes by that name, `noise` is
            not shaped ``(num_samples, *size)``, or `generator` is on another
            device.
    """
    if eta != 0.0:
        raise ValueError(f"plms is a deterministic solver, so eta must be 0, got {eta}")

    net = model if model is not None else diffusion.net

    if timesteps is None:
        timesteps = get_spacing(spacing)(
            diffusion.num_timesteps, num_steps, alphabar=diffusion.alphabar_t
        )
    ts = timesteps.to(device)
    ts_prev = torch.cat([ts[1:], ts.new_tensor([-1])])

    alphabar = diffusion.alphabar_t
    one = alphabar.new_ones(())

    with eval_mode(net):
        x = initial_latent(num_samples, size, device, noise=noise, generator=generator)
        history: list[torch.Tensor] = []

        for t_cur, t_prev in zip(ts, ts_prev, strict=True):
            ab_t = alphabar[t_cur]
            ab_prev = one if t_prev < 0 else alphabar[t_prev]

            _, eps = predict_xstart_eps(
                diffusion, x, t_cur.repeat(num_samples), model=net, clip_denoised=clip_denoised
            )

            weights = PLMS_COEFFICIENTS[len(history)]
            eps_hat = weights[0] * eps
            for weight, past in zip(weights[1:], history, strict=True):
                eps_hat = eps_hat + weight * past

            x0 = (x - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps_hat

            history.insert(0, eps)
            del history[_MAX_HISTORY:]

    return x
