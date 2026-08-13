"""DDIM sampling (Song et al. 2020, https://arxiv.org/abs/2010.02502).

DDIM is a *sampler*, not a different training objective. A model trained with
the DDPM loss can be sampled with either method, so this lives as a function
over a trained process rather than as a subclass that overrides `sample`.
"""

import torch
import torch.nn as nn

from tinydiffusion.diffusion.gaussian_diffusion import Diffusion, GaussianDiffusion
from tinydiffusion.utils.modules import eval_mode


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

    Returns:
        Tensor of shape (num_samples, *size).

    Raises:
        ValueError: if `eta` falls outside [0, 1].
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta}")

    net = model if model is not None else diffusion.net

    if timesteps is None:
        timesteps = uniform_timesteps(diffusion.num_timesteps, num_steps)
    ts = timesteps.to(device)
    # Pair each t with its predecessor; the last step lands on the t=-1 sentinel,
    # for which alphabar is defined as 1 (a noise-free x_0).
    ts_prev = torch.cat([ts[1:], ts.new_tensor([-1])])

    alphabar = diffusion.alphabar_t
    one = alphabar.new_ones(())

    with eval_mode(net):
        x = torch.randn(num_samples, *size, device=device)

        for t_cur, t_prev in zip(ts, ts_prev, strict=True):
            ab_t = alphabar[t_cur]
            ab_prev = one if t_prev < 0 else alphabar[t_prev]

            t_batch = t_cur.repeat(num_samples)

            if isinstance(diffusion, GaussianDiffusion):
                # The network may emit 2C channels, and may not be predicting
                # epsilon at all, so the implied x_0 has to come from the
                # process itself. DDIM's own variance is used either way: a
                # learned reverse variance describes the full-chain step, not
                # this strided one.
                *_, x0 = diffusion.p_mean_variance(
                    x, t_batch, model=net, clip_denoised=clip_denoised
                )
                eps = diffusion.predict_eps_from_xstart(x, t_batch, x0)
            else:
                eps = net(x, t_batch)
                x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
                if clip_denoised:
                    x0 = x0.clamp(-1.0, 1.0)
                    # Re-derive eps so the direction term stays consistent with
                    # the clamped x_0. Skipping this is a common, subtle bug.
                    eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()

            # sigma_t from DDIM Eq. 16.
            sigma = eta * (((1 - ab_prev) / (1 - ab_t)) * (1 - ab_t / ab_prev)).sqrt()
            direction = (1 - ab_prev - sigma**2).clamp(min=0.0).sqrt()

            x = ab_prev.sqrt() * x0 + direction * eps
            if eta > 0 and t_prev >= 0:
                x = x + sigma * torch.randn_like(x)

    return x
