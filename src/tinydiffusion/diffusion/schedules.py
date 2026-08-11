import math
from __future__ import annotations
from typing import Dict

import torch

def linear_beta_schedule(beta1: float, beta2: float, T: int) -> torch.Tensor:
    """DDPM's original linear schedule. Tuned for T=1000."""
    return torch.linspace(beta1, beta2, T, dtype=torch.float32)

def cosine_beta_scheduler(T: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal 2021. Noticeably better than linear at low resolution."""
    steps = torch.arange(T + 1, dtype=torch.float32) / T
    alphabar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphabar = alphabar / alphabar[0]
    betas = 1 - alphabar[1:] / alphabar[:-1]
    return betas.clamp(max=0.999)

def ddpm_schedules(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Pre-compute every coefficient needed for training and sampling.
 
    All tensors have length T and are indexed by t in [0, T-1], where t=0 is
    the first (least noisy) step. This differs from the minDiffusion
    convention of T+1 entries indexed from 1.
    """
    if not (betas > 0).all() or not (betas < 1).all():
        raise ValueError("all betas must lie in (0, 1)")
 
    alpha_t = 1.0 - betas
    alphabar_t = torch.cumprod(alpha_t, dim=0)
    # alphabar_{t-1}, with alphabar_{-1} := 1 so that t=0 is noise-free.
    alphabar_prev = torch.cat([torch.ones(1), alphabar_t[:-1]])
 
    # Posterior variance beta_tilde_t = beta_t * (1 - abar_{t-1}) / (1 - abar_t).
    # It is 0 at t=0, so clamp before taking a log or a sqrt.
    posterior_var = betas * (1.0 - alphabar_prev) / (1.0 - alphabar_t)
 
    return {
        "betas": betas,
        "alpha_t": alpha_t,
        "alphabar_t": alphabar_t,
        "alphabar_prev": alphabar_prev,
        "sqrtab": alphabar_t.sqrt(),
        "sqrtmab": (1.0 - alphabar_t).sqrt(),
        "oneover_sqrta": alpha_t.rsqrt(),
        "mab_over_sqrtmab": betas / (1.0 - alphabar_t).sqrt(),
        "sqrt_recip_ab": alphabar_t.reciprocal().sqrt(),
        "sqrt_recipm1_ab": (1.0 / alphabar_t - 1.0).sqrt(),
        "posterior_var": posterior_var,
        "posterior_logvar": posterior_var.clamp(min=1e-20).log(),
        "posterior_mean_c0": betas * alphabar_prev.sqrt() / (1.0 - alphabar_t),
        "posterior_mean_ct": (1.0 - alphabar_prev) * alpha_t.sqrt() / (1.0 - alphabar_t),
    }
