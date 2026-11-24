import math

import pytest
import torch

from tinydiffusion.diffusion.schedules import (
    cosine_beta_schedule,
    ddpm_schedules,
    enforce_zero_terminal_snr,
    linear_beta_schedule,
)

T = 100


def _alphabar(betas):
    return torch.cumprod(1.0 - betas, dim=0)


@pytest.mark.parametrize(
    "betas",
    [linear_beta_schedule(1e-4, 0.02, T), cosine_beta_schedule(T)],
    ids=["linear", "cosine"],
)
def test_the_terminal_signal_is_removed(betas):
    before = _alphabar(betas)[-1].sqrt()
    after = _alphabar(enforce_zero_terminal_snr(betas))[-1].sqrt()

    assert after < before
    assert after == pytest.approx(1e-4, rel=1e-2)


@pytest.mark.parametrize(
    "betas",
    [linear_beta_schedule(1e-4, 0.02, T), cosine_beta_schedule(T)],
    ids=["linear", "cosine"],
)
def test_the_first_step_is_left_where_it_was(betas):
    """The rescale pins the head: only the noisy end is meant to move."""
    rescaled = enforce_zero_terminal_snr(betas)
    assert rescaled[0] == pytest.approx(betas[0].item(), rel=1e-3)


def test_the_linear_schedule_is_the_one_that_leaks():
    """Where the rescale earns its keep, at the length anyone trains at."""
    betas = linear_beta_schedule(1e-4, 0.02, 1000)
    assert _alphabar(betas)[-1].sqrt() > 1e-3
    assert _alphabar(enforce_zero_terminal_snr(betas))[-1].sqrt() < 1e-3

    assert _alphabar(cosine_beta_schedule(1000))[-1].sqrt() < 1e-3


def test_the_schedule_stays_monotone_and_usable():
    betas = enforce_zero_terminal_snr(cosine_beta_schedule(T))
    alphabar = _alphabar(betas)

    assert torch.all(betas > 0) and torch.all(betas < 1)
    assert torch.all(alphabar[:-1] >= alphabar[1:])
    for name, buffer in ddpm_schedules(betas).items():
        assert torch.isfinite(buffer).all(), name


def test_a_schedule_outside_the_open_interval_is_rejected():
    with pytest.raises(ValueError, match="all betas must lie"):
        enforce_zero_terminal_snr(torch.tensor([0.0, 0.5]))


def test_the_floor_has_to_be_a_probability():
    with pytest.raises(ValueError, match="floor must lie"):
        enforce_zero_terminal_snr(cosine_beta_schedule(T), floor=0.0)


@pytest.mark.parametrize("length", [10, 100, 1000, 4000])
@pytest.mark.parametrize(
    "schedule",
    [lambda n: linear_beta_schedule(1e-4, 0.02, n), cosine_beta_schedule],
    ids=["linear", "cosine"],
)
def test_the_rescale_stays_buildable_at_every_length(schedule, length):
    """The floor must not flatten the tail into betas of zero.

    Clamping sqrt(alphabar) at the floor collapses however many of the last
    steps fall below it onto one value, and two equal alphabars mean an alpha
    of exactly 1 — a beta of 0, which `ddpm_schedules` refuses. The linear
    schedule at the length everyone trains at is where that bites: it left two
    entries under the floor, so `zero_snr = true` could not build a model at
    all.
    """
    betas = enforce_zero_terminal_snr(schedule(length))

    assert torch.all(betas > 0) and torch.all(betas < 1)
    alphabar = _alphabar(betas)
    assert torch.all(alphabar[:-1] > alphabar[1:])
    for name, buffer in ddpm_schedules(betas).items():
        assert torch.isfinite(buffer).all(), name


def test_the_terminal_signal_lands_on_the_floor_it_was_given():
    betas = enforce_zero_terminal_snr(linear_beta_schedule(1e-4, 0.02, 1000), floor=1e-3)
    assert _alphabar(betas)[-1].sqrt().item() == pytest.approx(1e-3, rel=1e-2)


def test_a_floor_above_the_schedule_leaves_nothing_to_rescale():
    with pytest.raises(ValueError, match="nothing left to rescale"):
        enforce_zero_terminal_snr(cosine_beta_schedule(T), floor=0.999999)


def _float64_schedule(betas):
    """The coefficients ddpm_schedules owes, derived independently in float64."""
    b = betas.double()
    ab = torch.cumprod(1.0 - b, dim=0)
    ab_prev = torch.cat([ab.new_ones(1), ab[:-1]])
    return {
        "sqrtmab": (1.0 - ab).sqrt(),
        "sqrt_recipm1_ab": (1.0 / ab - 1.0).sqrt(),
        "posterior_var": b * (1.0 - ab_prev) / (1.0 - ab),
        "snr": ab / (1.0 - ab),
    }


@pytest.mark.parametrize(
    "betas",
    [linear_beta_schedule(1e-4, 0.02, 1000), cosine_beta_schedule(1000)],
    ids=["linear", "cosine"],
)
def test_the_coefficients_are_accurate_to_float32_rounding(betas):
    """Deriving them in float32 left 1 - abar near t=0 with three or four digits."""
    schedule = ddpm_schedules(betas)
    for name, reference in _float64_schedule(betas).items():
        assert schedule[name].dtype == torch.float32
        torch.testing.assert_close(schedule[name].double(), reference, rtol=1e-6, atol=0)


def test_a_beta_below_float32_epsilon_still_gives_finite_coefficients():
    """In float32, 1 - 1e-8 is exactly 1, which made abar_0 = 1 and t=0 a 0/0."""
    betas = torch.cat([torch.tensor([1e-8]), linear_beta_schedule(1e-4, 0.02, T - 1)])
    schedule = ddpm_schedules(betas)
    for name, buffer in schedule.items():
        assert buffer.isfinite().all(), name
    assert schedule["sqrtmab"][0] > 0


def test_the_cosine_betas_match_a_float64_construction():
    steps = torch.arange(1001, dtype=torch.float64) / 1000
    alphabar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
    reference = (1 - alphabar[1:] / alphabar[:-1]).clamp(max=0.999)

    betas = cosine_beta_schedule(1000)
    assert betas.dtype == torch.float32
    torch.testing.assert_close(betas.double(), reference, rtol=1e-6, atol=0)


@pytest.mark.parametrize(
    "betas",
    [linear_beta_schedule(1e-4, 0.02, 1000), cosine_beta_schedule(1000)],
    ids=["linear", "cosine"],
)
def test_the_rescale_holds_the_first_beta_to_float32_rounding(betas):
    """In float32 the 'fixed' first beta drifted by 0.1% over a 1000-step schedule."""
    rescaled = enforce_zero_terminal_snr(betas)
    assert rescaled.dtype == betas.dtype
    assert rescaled[0].item() == pytest.approx(betas[0].item(), rel=1e-6)
    torch.testing.assert_close(
        rescaled.double(), enforce_zero_terminal_snr(betas.double()), rtol=1e-6, atol=0
    )
