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
    # Down to the floor, give or take the float error in rebuilding the betas
    # from it and taking the product back out again.
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

    # The cosine schedule clamps its betas at 0.999 and gets there by itself,
    # so the rescale has little left to do; this pins that it stays true.
    assert _alphabar(cosine_beta_schedule(1000))[-1].sqrt() < 1e-3


def test_the_schedule_stays_monotone_and_usable():
    betas = enforce_zero_terminal_snr(cosine_beta_schedule(T))
    alphabar = _alphabar(betas)

    assert torch.all(betas > 0) and torch.all(betas < 1)
    assert torch.all(alphabar[:-1] >= alphabar[1:])
    # Every derived coefficient has to stay finite, since they are built for
    # the whole schedule whether or not this run's parameterisation reads them.
    for name, buffer in ddpm_schedules(betas).items():
        assert torch.isfinite(buffer).all(), name


def test_a_schedule_outside_the_open_interval_is_rejected():
    with pytest.raises(ValueError, match="all betas must lie"):
        enforce_zero_terminal_snr(torch.tensor([0.0, 0.5]))


def test_the_floor_has_to_be_a_probability():
    with pytest.raises(ValueError, match="floor must lie"):
        enforce_zero_terminal_snr(cosine_beta_schedule(T), floor=0.0)
