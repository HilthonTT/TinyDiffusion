import pytest
import torch
import torch.nn as nn

from tinydiffusion.diffusion.ddim import (
    ddim_sample,
    quadratic_timesteps,
    uniform_timesteps,
)
from tinydiffusion.diffusion.ddpm import DDPM
from tinydiffusion.diffusion.schedules import linear_beta_schedule

T = 100


@pytest.mark.parametrize("subsequence", [uniform_timesteps, quadratic_timesteps])
def test_timesteps_are_descending_and_span_the_schedule(subsequence):
    steps = subsequence(T, 10)
    assert steps.dtype == torch.long
    assert steps[0] == T - 1
    assert steps[-1] == 0
    assert torch.all(steps[:-1] > steps[1:])


@pytest.mark.parametrize("subsequence", [uniform_timesteps, quadratic_timesteps])
def test_a_single_step_starts_from_the_noisiest_timestep(subsequence):
    # [0] would denoise pure noise as if it were an almost-clean image.
    assert subsequence(T, 1).tolist() == [T - 1]


@pytest.mark.parametrize("subsequence", [uniform_timesteps, quadratic_timesteps])
def test_impossible_step_counts_are_rejected(subsequence):
    with pytest.raises(ValueError, match="num_steps"):
        subsequence(T, 0)
    with pytest.raises(ValueError, match="num_steps"):
        subsequence(T, T + 1)


def test_uniform_matches_the_full_schedule_reversed():
    assert uniform_timesteps(T, T).tolist() == list(reversed(range(T)))


class _Oracle(nn.Module):
    """Predicts the exact epsilon that maps `target` to x_t."""

    def __init__(self, alphabar: torch.Tensor, target: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("alphabar", alphabar)
        self.register_buffer("target", target)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ab = self.alphabar.gather(0, t).reshape(-1, 1, 1, 1)
        return (x - ab.sqrt() * self.target) / (1 - ab).sqrt()


@pytest.fixture
def oracle_diffusion():
    """A DDPM whose network is exact, so eta=0 sampling must return `target`."""
    target = torch.full((1, 1, 4, 4), 0.5)
    target[..., 2:] = -0.5
    schedule = DDPM(nn.Identity(), betas=linear_beta_schedule(1e-4, 0.02, T), num_timesteps=T)
    return DDPM(_Oracle(schedule.alphabar_t, target), betas=schedule.betas, num_timesteps=T), target


@pytest.mark.parametrize("num_steps", [1, 2, 10, T])
def test_deterministic_ddim_inverts_an_exact_noise_model(oracle_diffusion, num_steps):
    diffusion, target = oracle_diffusion
    out = ddim_sample(
        diffusion, 3, (1, 4, 4), "cpu", num_steps=num_steps, eta=0.0, clip_denoised=False
    )
    assert torch.allclose(out, target.expand_as(out), atol=1e-5)


def test_eta_outside_the_unit_interval_is_rejected(oracle_diffusion):
    diffusion, _ = oracle_diffusion
    with pytest.raises(ValueError, match="eta"):
        ddim_sample(diffusion, 1, (1, 4, 4), "cpu", num_steps=4, eta=1.5)


def test_sampling_stays_finite_for_every_eta(oracle_diffusion):
    diffusion, _ = oracle_diffusion
    for eta in (0.0, 0.5, 1.0):
        out = ddim_sample(diffusion, 2, (1, 4, 4), "cpu", num_steps=8, eta=eta)
        assert torch.isfinite(out).all()
