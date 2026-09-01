"""Turning one network evaluation into x_0 and epsilon, whatever the process.

Every sampler needs the same thing out of the network — the implied clean
image, and the noise consistent with it after clipping — and getting it wrong
is the classic subtle sampling bug: clamp x_0 but keep the epsilon that implied
the unclamped one, and the direction term pulls against the clamp at every
step. It lives here so :mod:`~tinydiffusion.diffusion.ddim` and
:mod:`~tinydiffusion.diffusion.dpm_solver` cannot drift apart on it.
"""

import torch
import torch.nn as nn

from tinydiffusion.diffusion.gaussian_diffusion import Diffusion, GaussianDiffusion

__all__ = ["predict_xstart_eps"]


def predict_xstart_eps(
    diffusion: Diffusion,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    model: nn.Module | None = None,
    clip_denoised: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the network at `t` and return the implied ``(x_0, eps)``.

    Args:
        diffusion: the trained process, for its schedule and its
            parameterisation.
        x: ``(B, C, H, W)`` latents at timestep `t`.
        t: ``(B,)`` integer timesteps.
        model: network to evaluate. Defaults to the process's own; pass
            ``ema.module`` for EMA weights, or a
            :class:`~tinydiffusion.diffusion.guidance.Conditioned` wrapper for
            class conditioning.
        clip_denoised: clamp the predicted x_0 to [-1, 1], and re-derive
            epsilon from the clamped value.

    Returns:
        Tuple of ``(pred_xstart, eps)``, both shaped like `x`.
    """
    net = model if model is not None else diffusion.net

    if isinstance(diffusion, GaussianDiffusion):
        *_, x0 = diffusion.p_mean_variance(x, t, model=net, clip_denoised=clip_denoised)
        return x0, diffusion.predict_eps_from_xstart(x, t, x0)

    ab_t = diffusion.alphabar_t.gather(0, t).reshape(-1, *([1] * (x.dim() - 1)))
    eps = net(x, t)
    x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
    if clip_denoised:
        x0 = x0.clamp(-1.0, 1.0)
        eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
    return x0, eps
