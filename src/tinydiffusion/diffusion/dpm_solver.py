"""DPM-Solver++(2M) sampling (Lu et al. 2022, https://arxiv.org/abs/2211.01095).

DDIM treats the reverse process as a first-order step along the probability-flow
ODE, which is why it needs 50-odd steps to look good. DPM-Solver++ notices that
the ODE has an exactly solvable linear part, integrates that in closed form, and
approximates only what is left — the x_0 prediction — with a multistep formula
that reuses the *previous* step's evaluation rather than paying for a second
network call. So a step costs what a DDIM step costs, and 10 to 20 of them land
around where 50 DDIM steps do.

Like DDIM this is a sampler, not a training objective: any checkpoint this
project produces can be drawn from with either.
"""

import torch
import torch.nn as nn

from tinydiffusion.diffusion.ddim import DEFAULT_SPACING, get_spacing
from tinydiffusion.diffusion.gaussian_diffusion import Diffusion
from tinydiffusion.diffusion.latents import initial_latent
from tinydiffusion.diffusion.prediction import predict_xstart_eps
from tinydiffusion.utils.modules import eval_mode

__all__ = ["dpmpp_sample"]


@torch.inference_mode()
def dpmpp_sample(
    diffusion: Diffusion,
    num_samples: int,
    size: tuple[int, ...],
    device: torch.device | str,
    num_steps: int = 20,
    eta: float = 0.0,
    model: nn.Module | None = None,
    timesteps: torch.Tensor | None = None,
    clip_denoised: bool = True,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    spacing: str = DEFAULT_SPACING,
) -> torch.Tensor:
    """Sample with the second-order multistep DPM-Solver++.

    Signature-compatible with
    :func:`~tinydiffusion.diffusion.ddim.ddim_sample`, so the two are
    interchangeable through
    :data:`~tinydiffusion.diffusion.samplers.SAMPLERS`.

    Args:
        diffusion: a trained process, used for its schedule buffers and network.
        num_samples: batch size to generate.
        size: shape of one sample, e.g. ``(1, 28, 28)``.
        device: device to generate on.
        num_steps: how many denoising steps to take. Ignored if `timesteps` is
            given. 15 to 20 is the useful range; below about 10 the second-order
            correction starts to hurt more than it helps.
        eta: accepted only as 0. The solver integrates the probability-flow
            ODE, which has no noise term to scale — anything else would be a
            different sampler wearing this one's name.
        model: network to sample from. Pass ``ema.module`` to use EMA weights.
        timesteps: explicit descending subsequence, overriding `num_steps`.
        clip_denoised: clamp the predicted x_0 to [-1, 1] at each step.
        noise: the starting x_T, of shape ``(num_samples, *size)``. None draws
            a fresh one.
        generator: RNG for the starting latent. None uses the global RNG. The
            chain itself is deterministic, so this is the only draw there is.
        spacing: which subsequence of the training schedule to visit; a key of
            :data:`~tinydiffusion.diffusion.ddim.SPACINGS`. The step ratio the
            second-order term is built from is computed per step, so a
            non-uniform grid stays correct. Ignored when `timesteps` is given.

    Returns:
        Tensor of shape ``(num_samples, *size)``.

    Raises:
        ValueError: if `eta` is not 0, no spacing goes by that name, `noise` is
            not shaped ``(num_samples, *size)``, or `generator` is on another
            device.
    """
    if eta != 0.0:
        raise ValueError(f"dpmpp is a deterministic solver, so eta must be 0, got {eta}")

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
        x0_prev: torch.Tensor | None = None
        h_prev: torch.Tensor | None = None

        for t_cur, t_prev in zip(ts, ts_prev, strict=True):
            ab_t = alphabar[t_cur]
            ab_prev = one if t_prev < 0 else alphabar[t_prev]

            x0, _ = predict_xstart_eps(
                diffusion, x, t_cur.repeat(num_samples), model=net, clip_denoised=clip_denoised
            )

            if t_prev < 0:
                return x0

            alpha_t, sigma_t = ab_t.sqrt(), (1 - ab_t).sqrt()
            alpha_prev, sigma_prev = ab_prev.sqrt(), (1 - ab_prev).sqrt()
            lambda_t = alpha_t.log() - sigma_t.log()
            lambda_prev = alpha_prev.log() - sigma_prev.log()
            h = lambda_prev - lambda_t

            if x0_prev is None or h_prev is None:
                d = x0
            else:
                r = h_prev / h
                d = (1.0 + 1.0 / (2.0 * r)) * x0 - (1.0 / (2.0 * r)) * x0_prev

            x = (sigma_prev / sigma_t) * x - alpha_prev * torch.expm1(-h) * d
            x0_prev, h_prev = x0, h

    return x
