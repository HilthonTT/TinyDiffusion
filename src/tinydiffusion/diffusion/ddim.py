"""DDIM sampling (Song et al. 2020, https://arxiv.org/abs/2010.02502).
 
DDIM is a *sampler*, not a different training objective. A model trained with
the DDPM loss can be sampled with either method, so this lives as a function
over a trained DDPM rather than as a subclass that overrides `sample`.
"""

import torch
import torch.nn as nn

from ddpm import DDPM

def uniform_timesteps(n_T: int, n_steps: int) -> torch.Tensor:
    """Evenly spaced subsequence of [0, n_T-1], descending.
 
    Args:
        n_T: number of steps the model was trained with.
        n_steps: number of sampling steps to actually take.
 
    Returns:
        Long tensor of length n_steps, strictly decreasing, ending at 0.
    """
    if not 1 <= n_steps <= n_T:
        raise ValueError(f"n_steps must lie in [1, {n_T}], got {n_steps}")
    steps = torch.linspace(0, n_T - 1, n_steps).round().long()
    return steps.flip(0)

def quadratic_timesteps(n_T: int, n_steps: int) -> torch.Tensor:
    """Quadratically spaced subsequence, denser near t=0.
 
    The DDIM paper found this better than uniform on CIFAR-10 at low step counts.
    """
    if not 1 <= n_steps <= n_T:
        raise ValueError(f"n_steps must lie in [1, {n_T}], got {n_steps}")
    steps = (torch.linspace(0, (n_T - 1) ** 0.5, n_steps) ** 2).round().long()
    return torch.unique(steps).flip(0)

@torch.no_grad()
def ddim_sample(
    diffusion: DDPM,
    n_sample: int,
    size: tuple[int, ...],
    device: torch.device | str,
    n_steps: int = 50,
    eta: float = 0.0,
    model: nn.Module | None = None,
    timesteps: torch.Tensor | None = None,
    clip_denoised: bool = True,
) -> torch.Tensor:
    """Sample by running the DDIM reverse chain over a timestep subsequence.
 
    Args:
        diffusion: a trained DDPM, used for its schedule buffers and eps_model.
        n_sample: batch size to generate.
        size: shape of one sample, e.g. (1, 28, 28).
        device: device to generate on.
        n_steps: how many denoising steps to take. Ignored if `timesteps` given.
        eta: 0.0 is deterministic DDIM; 1.0 reproduces DDPM ancestral sampling.
        model: network to sample from. Pass `ema.module` to use EMA weights.
        timesteps: explicit descending subsequence, overriding `n_steps`.
        clip_denoised: clamp the predicted x_0 to [-1, 1] at each step. Matters
            much more at low step counts than it does for full-chain DDPM.
 
    Returns:
        Tensor of shape (n_sample, *size).
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta}")
 
    net = model if model is not None else diffusion.eps_model
    was_training = net.training
    net.eval()
 
    ts = uniform_timesteps(diffusion.n_T, n_steps) if timesteps is None else timesteps
    ts = ts.to(device)
    # Pair each t with its predecessor; the last step lands on the t=-1 sentinel,
    # for which alphabar is defined as 1 (a noise-free x_0).
    ts_prev = torch.cat([ts[1:], ts.new_tensor([-1])])
 
    alphabar = diffusion.alphabar_t
    one = alphabar.new_ones(())
 
    try:
        x = torch.randn(n_sample, *size, device=device)
 
        for t_cur, t_prev in zip(ts, ts_prev, strict=True):
            ab_t = alphabar[t_cur]
            ab_prev = one if t_prev < 0 else alphabar[t_prev]
 
            t_batch = t_cur.expand(n_sample)
            eps = net(x, t_batch)
 
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if clip_denoised:
                x0 = x0.clamp(-1.0, 1.0)
                # Re-derive eps so the direction term stays consistent with the
                # clamped x_0. Skipping this is a common and subtle bug.
                eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
 
            # sigma_t from DDIM Eq. 16.
            sigma = eta * (((1 - ab_prev) / (1 - ab_t)) * (1 - ab_t / ab_prev)).sqrt()
            direction = (1 - ab_prev - sigma**2).clamp(min=0.0).sqrt()
 
            x = ab_prev.sqrt() * x0 + direction * eps
            if eta > 0 and t_prev >= 0:
                x = x + sigma * torch.randn_like(x)
    finally:
        net.train(was_training)
 
    return x