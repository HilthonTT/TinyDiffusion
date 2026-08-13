"""Likelihood helpers for the variational lower bound.

Ported from Ho et al.'s original codebase via openai/improved-diffusion, and
rewritten to be torch-native and strictly typed.
"""

import math

import torch

__all__ = [
    "approx_standard_normal_cdf",
    "discretized_gaussian_log_likelihood",
    "mean_flat",
    "normal_kl",
]


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Average over every dimension except the batch.

    Args:
        x: tensor of shape ``(B, ...)``.

    Returns:
        Shape ``(B,)`` tensor of per-sample means.
    """
    return x.flatten(start_dim=1).mean(dim=1)


def normal_kl(
    mean1: torch.Tensor,
    logvar1: torch.Tensor,
    mean2: torch.Tensor,
    logvar2: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between two diagonal Gaussians, elementwise.

    Computed in terms of log-variance rather than variance so the expression
    stays finite when a variance underflows, which it does at t=0 where the
    true posterior is a point mass.

    Args:
        mean1: mean of the first distribution (the one being measured).
        logvar1: log-variance of the first distribution.
        mean2: mean of the second distribution.
        logvar2: log-variance of the second distribution.

    Returns:
        Elementwise KL, broadcast to the common shape, in nats.
    """
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Fast tanh approximation of the standard normal CDF.

    Args:
        x: points at which to evaluate the CDF.

    Returns:
        Approximate ``Phi(x)``, same shape as `x`.
    """
    coefficient = math.sqrt(2.0 / math.pi)
    return 0.5 * (1.0 + torch.tanh(coefficient * (x + 0.044715 * x.pow(3))))


def discretized_gaussian_log_likelihood(
    x: torch.Tensor,
    *,
    means: torch.Tensor,
    log_scales: torch.Tensor,
) -> torch.Tensor:
    """Log-likelihood of a Gaussian discretized onto 8-bit image values.

    The final reverse step emits a continuous density, but the data are
    integers rescaled to [-1, 1]. Integrating the density over each value's
    1/255-wide bin is what makes the bound comparable to a real bits-per-dim
    number rather than an arbitrary density.

    Args:
        x: target images in [-1, 1], assumed to have come from uint8 values.
        means: predicted Gaussian means, same shape as `x`.
        log_scales: predicted log standard deviations, same shape as `x`.

    Returns:
        Elementwise log probabilities in nats, same shape as `x`.

    Raises:
        ValueError: if the three tensors do not share a shape.
    """
    if not x.shape == means.shape == log_scales.shape:
        raise ValueError(
            f"shape mismatch: x {tuple(x.shape)}, means {tuple(means.shape)}, "
            f"log_scales {tuple(log_scales.shape)}"
        )

    centered = x - means
    inv_stdv = torch.exp(-log_scales)
    cdf_plus = approx_standard_normal_cdf(inv_stdv * (centered + 1.0 / 255.0))
    cdf_min = approx_standard_normal_cdf(inv_stdv * (centered - 1.0 / 255.0))

    log_cdf_plus = cdf_plus.clamp(min=1e-12).log()
    log_one_minus_cdf_min = (1.0 - cdf_min).clamp(min=1e-12).log()
    log_cdf_delta = (cdf_plus - cdf_min).clamp(min=1e-12).log()

    # The extreme bins are half-open: everything below -0.999 collapses into the
    # lowest bucket and everything above 0.999 into the highest, so those use a
    # one-sided tail rather than a difference of two nearly equal CDFs.
    return torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, log_cdf_delta),
    )
