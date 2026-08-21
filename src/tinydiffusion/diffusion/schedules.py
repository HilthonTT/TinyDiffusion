"""Beta schedules and the derived coefficients used by DDPM/DDIM."""

import math

import torch


def linear_beta_schedule(beta_start: float, beta_end: float, num_timesteps: int) -> torch.Tensor:
    """DDPM's original linear schedule. Tuned for num_timesteps=1000.

    Args:
        beta_start: beta at t=0.
        beta_end: beta at the final step.
        num_timesteps: length of the schedule.

    Returns:
        1-D tensor of betas, length num_timesteps.
    """
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal 2021. Noticeably better than linear at low resolution.

    Args:
        num_timesteps: length of the schedule.
        s: small offset preventing beta from being too small near t=0.

    Returns:
        1-D tensor of betas, length num_timesteps.
    """
    steps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
    alphabar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphabar = alphabar / alphabar[0]
    betas = 1 - alphabar[1:] / alphabar[:-1]
    return betas.clamp(max=0.999)


def ddpm_schedules(betas: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pre-compute every coefficient needed for training and sampling.

    All tensors have length T and are indexed by t in [0, T-1], where t=0 is
    the first (least noisy) step. This differs from the minDiffusion
    convention of T+1 entries indexed from 1.

    Args:
        betas: 1-D tensor of betas, every entry in (0, 1).

    Returns:
        Mapping from buffer name to coefficient tensor.

    Raises:
        ValueError: if any beta falls outside (0, 1).
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
        # Signal-to-noise ratio of x_t. Only the loss weighting reads it, but it
        # belongs to the schedule rather than to any one objective.
        "snr": alphabar_t / (1.0 - alphabar_t),
    }


def enforce_zero_terminal_snr(betas: torch.Tensor, floor: float = 1e-4) -> torch.Tensor:
    """Rescale a schedule so the last step carries (almost) no signal.

    A schedule that does not reach zero signal-to-noise at its last step
    leaves x_T holding a trace of the image's mean brightness. Training never
    notices — it always starts from a real image — but sampling starts from
    pure noise, whose mean is 0, and the model spends the chain restoring a
    brightness the noise never had. The symptom is samples that are never
    fully black or fully white, and it gets worse the fewer steps you take.

    Lin et al. 2024 (https://arxiv.org/abs/2305.08891) fix it by rescaling
    sqrt(alphabar) linearly so its last entry is zero, holding the first entry
    fixed, and recovering the betas from the result.

    How much this is worth depends on the schedule.
    :func:`linear_beta_schedule` leaves sqrt(alphabar_T) at about 0.006 over
    1000 steps and 0.6 over 100, which is a real leak;
    :func:`cosine_beta_schedule` clamps its betas at 0.999 and already lands
    within 5e-5 of zero, so the rescale mostly just makes that exact. Either
    way it costs nothing at training time, and it is the schedule the short
    sampling chains were tuned against.

    Args:
        betas: 1-D tensor of betas, every entry in (0, 1).
        floor: smallest sqrt(alphabar) the last steps may take. Exactly zero is
            what the paper asks for, but it puts an infinity in every
            coefficient that divides by sqrt(alphabar) — the epsilon
            parameterisation's x_0 recovery, chiefly — and those coefficients
            are built for the whole schedule whether or not this run uses
            them. A floor of 1e-4 leaves an SNR of 1e-8, which is zero for
            every practical purpose, and keeps the buffers finite.

    Returns:
        1-D tensor of rescaled betas, the same length as the input.

    Raises:
        ValueError: if any beta falls outside (0, 1), `floor` is not in
            (0, 1), or `floor` is not below the schedule's own first
            sqrt(alphabar), which leaves nothing to rescale.
    """
    if not (betas > 0).all() or not (betas < 1).all():
        raise ValueError("all betas must lie in (0, 1)")
    if not 0.0 < floor < 1.0:
        raise ValueError(f"floor must lie in (0, 1), got {floor}")

    sqrt_ab = torch.cumprod(1.0 - betas, dim=0).sqrt()
    first, last = sqrt_ab[0].clone(), sqrt_ab[-1].clone()
    if first <= floor:
        raise ValueError(
            f"floor={floor} is not below the schedule's own first sqrt(alphabar) of "
            f"{first.item():g}, so there is nothing left to rescale"
        )
    # Shift the tail onto the floor and stretch so the head lands back where it
    # was. Mapped affinely rather than rescaled to zero and then clamped: a
    # clamp flattens however many of the last steps fall below the floor onto
    # one value, and two equal alphabars make alpha exactly 1 — a beta of 0,
    # which `ddpm_schedules` rejects outright. The linear schedule at 1,000
    # steps lands two entries under a floor of 1e-4 and so could not be built
    # at all. An affine map keeps the sequence strictly decreasing, which is
    # what keeps every beta inside (0, 1).
    sqrt_ab = floor + (sqrt_ab - last) * ((first - floor) / (first - last))

    alphabar = sqrt_ab.square()
    # alpha_t = abar_t / abar_{t-1}, with abar_{-1} = 1.
    alphas = alphabar / torch.cat([torch.ones(1, dtype=alphabar.dtype), alphabar[:-1]])
    return 1.0 - alphas
