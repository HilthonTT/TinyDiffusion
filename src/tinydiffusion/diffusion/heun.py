"""Heun sampling: the second-order corrector DDIM does not take.

DDIM is Euler's method on the probability-flow ODE — it evaluates the network
once, believes that direction for the whole step, and pays for it with the
truncation error that makes 50 steps the usual budget. Heun's method takes the
same Euler step provisionally, evaluates the network *again* at where it landed,
and re-takes the step along the average of the two directions. The error per
step falls from O(h^2) to O(h^3), which is the same trade Karras et al. 2022
(https://arxiv.org/abs/2206.00364) make in EDM, where Heun is the default
solver.

It is not free: a step costs two network evaluations rather than one, so 20
Heun steps cost what 40 DDIM steps do. The question is only ever whether the
second-order step is worth more than a shorter first-order one at the same
budget, and it usually is once the step size is large enough for truncation
error to dominate. :mod:`~tinydiffusion.diffusion.dpm_solver` gets its second
order from the *previous* step's evaluation instead, so it reaches second order
at one evaluation per step; Heun's is the version that needs no history and so
is correct from its very first step, which is where a short chain does its
worst damage.

The step itself is taken in the same exponentially-integrated form
:mod:`~tinydiffusion.diffusion.dpm_solver` uses — the linear part of the ODE
solved in closed form, the x_0 prediction the only thing approximated — so
"Heun" here means the trapezoidal average of the x_0 predictions at the two
ends of the step, not Heun applied to the raw ODE.
"""

import torch
import torch.nn as nn

from tinydiffusion.diffusion.ddim import DEFAULT_SPACING, get_spacing
from tinydiffusion.diffusion.gaussian_diffusion import Diffusion
from tinydiffusion.diffusion.latents import initial_latent
from tinydiffusion.diffusion.prediction import predict_xstart_eps
from tinydiffusion.utils.modules import eval_mode

__all__ = ["heun_sample"]


@torch.inference_mode()
def heun_sample(
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
    """Sample with the second-order Heun (trapezoidal) solver.

    Signature-compatible with
    :func:`~tinydiffusion.diffusion.ddim.ddim_sample`, so the three solvers are
    interchangeable through
    :data:`~tinydiffusion.diffusion.samplers.SAMPLERS`.

    .. warning::
        `num_steps` counts *steps*, not network evaluations, and this solver
        spends two per step. Compare it against DDIM and DPM-Solver++ at twice
        their step count, not at the same one — ``heun`` at 20 and ``dpmpp`` at
        40 are the comparison that holds the cost fixed.

    The final step is the exception: it lands on the ``t = -1`` sentinel, where
    ``alphabar`` is 1 and there is no noise left to correct for, so every solver
    collapses to its own x_0 prediction and there is nothing for a second
    evaluation to improve. That step costs one evaluation, giving
    ``2 * num_steps - 1`` in total.

    Args:
        diffusion: a trained process, used for its schedule buffers and network.
        num_samples: batch size to generate.
        size: shape of one sample, e.g. ``(1, 28, 28)``.
        device: device to generate on.
        num_steps: how many denoising steps to take, each costing two network
            evaluations. Ignored if `timesteps` is given.
        eta: accepted only as 0. Like DPM-Solver++ this integrates the
            probability-flow ODE, which has no noise term to scale.
        model: network to sample from. Pass ``ema.module`` to use EMA weights.
        timesteps: explicit descending subsequence, overriding `num_steps`.
        clip_denoised: clamp the predicted x_0 to [-1, 1] at each evaluation,
            both the predictor's and the corrector's.
        noise: the starting x_T, of shape ``(num_samples, *size)``. None draws
            a fresh one.
        generator: RNG for the starting latent. None uses the global RNG. The
            chain itself is deterministic, so this is the only draw there is.
        spacing: which subsequence of the training schedule to visit; a key of
            :data:`~tinydiffusion.diffusion.ddim.SPACINGS`. The step is
            computed from the two endpoints it actually spans, so a non-uniform
            grid stays correct. Ignored when `timesteps` is given.

    Returns:
        Tensor of shape ``(num_samples, *size)``.

    Raises:
        ValueError: if `eta` is not 0, no spacing goes by that name, `noise` is
            not shaped ``(num_samples, *size)``, or `generator` is on another
            device.
    """
    if eta != 0.0:
        raise ValueError(f"heun is a deterministic solver, so eta must be 0, got {eta}")

    net = model if model is not None else diffusion.net

    if timesteps is None:
        timesteps = get_spacing(spacing)(
            diffusion.num_timesteps, num_steps, alphabar=diffusion.alphabar_t
        )
    ts = timesteps.to(device)
    # As in DDIM, the last step lands on a t=-1 sentinel whose alphabar is 1.
    ts_prev = torch.cat([ts[1:], ts.new_tensor([-1])])

    alphabar = diffusion.alphabar_t
    one = alphabar.new_ones(())

    with eval_mode(net):
        x = initial_latent(num_samples, size, device, noise=noise, generator=generator)

        for t_cur, t_prev in zip(ts, ts_prev, strict=True):
            ab_t = alphabar[t_cur]
            ab_prev = one if t_prev < 0 else alphabar[t_prev]

            x0, _ = predict_xstart_eps(
                diffusion, x, t_cur.repeat(num_samples), model=net, clip_denoised=clip_denoised
            )

            if t_prev < 0:
                # Denoising to alphabar = 1 leaves the x_0 prediction itself,
                # and a corrector would have to evaluate the network at a
                # timestep that does not exist.
                return x0

            alpha_t, sigma_t = ab_t.sqrt(), (1 - ab_t).sqrt()
            alpha_prev, sigma_prev = ab_prev.sqrt(), (1 - ab_prev).sqrt()
            # lambda = log-SNR / 2, the variable the linear part of the ODE is
            # exactly integrable in; h is the gap this step spans.
            h = (alpha_prev.log() - sigma_prev.log()) - (alpha_t.log() - sigma_t.log())
            # Common to both the provisional step and the corrected one, so the
            # two differ in nothing but which x_0 estimate they are given.
            decay, step = sigma_prev / sigma_t, -alpha_prev * torch.expm1(-h)

            # Euler: the whole step along the direction measured at t_cur.
            x_euler = decay * x + step * x0
            # And the direction at where that landed. Evaluated at t_prev, the
            # timestep x_euler is an estimate of, which is what makes the
            # average a trapezoidal rule over the step rather than two samples
            # of the same end of it.
            x0_next, _ = predict_xstart_eps(
                diffusion,
                x_euler,
                t_prev.repeat(num_samples),
                model=net,
                clip_denoised=clip_denoised,
            )
            x = decay * x + step * (0.5 * (x0 + x0_next))

    return x
