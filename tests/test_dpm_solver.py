import pytest
import torch
import torch.nn as nn

from tinydiffusion.diffusion.ddim import ddim_sample, uniform_timesteps
from tinydiffusion.diffusion.ddpm import DDPM
from tinydiffusion.diffusion.dpm_solver import dpmpp_sample
from tinydiffusion.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType
from tinydiffusion.diffusion.samplers import SAMPLERS, get_sampler, sampler_names
from tinydiffusion.diffusion.schedules import linear_beta_schedule

T = 100
SIZE = (1, 4, 4)


class _Echo(nn.Module):
    """A network whose output depends on its input, cheaply and smoothly."""

    def forward(self, x, t):
        return x * 0.5


@pytest.fixture
def diffusion():
    return DDPM(_Echo(), betas=linear_beta_schedule(1e-4, 0.02, T), num_timesteps=T)


@pytest.fixture
def noise():
    return torch.randn(2, *SIZE)


def test_it_produces_the_shape_it_was_asked_for(diffusion):
    assert dpmpp_sample(diffusion, 3, SIZE, "cpu", num_steps=5).shape == (3, *SIZE)


def test_the_chain_is_deterministic_given_its_latent(diffusion, noise):
    kwargs = {"num_steps": 6, "noise": noise}
    first = dpmpp_sample(diffusion, 2, SIZE, "cpu", **kwargs)
    second = dpmpp_sample(diffusion, 2, SIZE, "cpu", **kwargs)
    assert torch.equal(first, second)


def test_two_steps_are_still_just_ddim(diffusion, noise):
    """The first step is first-order, and the last returns x_0 either way.

    So a two-step chain has no second-order correction to apply, and the two
    samplers have to agree exactly. Anything else means the shared first-order
    step has drifted.
    """
    a = dpmpp_sample(diffusion, 2, SIZE, "cpu", num_steps=2, noise=noise)
    b = ddim_sample(diffusion, 2, SIZE, "cpu", num_steps=2, eta=0.0, noise=noise)
    assert torch.allclose(a, b, atol=1e-6)


def test_the_correction_starts_at_the_third_step(diffusion, noise):
    """With three steps there is a previous x_0 to extrapolate from."""
    a = dpmpp_sample(diffusion, 2, SIZE, "cpu", num_steps=3, noise=noise)
    b = ddim_sample(diffusion, 2, SIZE, "cpu", num_steps=3, eta=0.0, noise=noise)
    assert not torch.allclose(a, b, atol=1e-6)


def test_a_short_chain_lands_near_a_long_one(noise):
    """The point of the solver: second-order convergence on the same ODE.

    Both samplers integrate the same trajectory, so a 500-step DDIM run stands
    in for the exact answer. The solver has to get an order of magnitude closer
    to it at the same step count — that gap is the whole reason to have it.

    Run unclamped and over a full-length schedule: clipping x_0 puts a kink in
    the trajectory, and extrapolating across a kink is not what a second-order
    method is good at.
    """
    long_schedule = DDPM(_Echo(), betas=linear_beta_schedule(1e-4, 0.02, 1000), num_timesteps=1000)
    kwargs = {"noise": noise, "clip_denoised": False}

    reference = ddim_sample(long_schedule, 2, SIZE, "cpu", num_steps=500, eta=0.0, **kwargs)
    solver = dpmpp_sample(long_schedule, 2, SIZE, "cpu", num_steps=20, **kwargs)
    baseline = ddim_sample(long_schedule, 2, SIZE, "cpu", num_steps=20, eta=0.0, **kwargs)

    solver_error = (solver - reference).abs().mean()
    baseline_error = (baseline - reference).abs().mean()
    assert solver_error < baseline_error / 10


def test_a_stochastic_eta_is_refused_rather_than_ignored(diffusion):
    with pytest.raises(ValueError, match="deterministic solver"):
        dpmpp_sample(diffusion, 1, SIZE, "cpu", num_steps=4, eta=0.5)


def test_it_honours_an_explicit_timestep_grid(diffusion, noise):
    steps = uniform_timesteps(T, 4)
    a = dpmpp_sample(diffusion, 2, SIZE, "cpu", timesteps=steps, noise=noise)
    b = dpmpp_sample(diffusion, 2, SIZE, "cpu", num_steps=4, noise=noise)
    assert torch.equal(a, b)


def test_a_mis_shaped_latent_is_caught_before_the_chain(diffusion):
    with pytest.raises(ValueError, match="noise must be shaped"):
        dpmpp_sample(diffusion, 3, SIZE, "cpu", num_steps=4, noise=torch.randn(2, *SIZE))


def test_it_routes_a_non_epsilon_process_through_p_mean_variance():
    """A v-prediction model has to be read as one, not as raw epsilon."""
    process = GaussianDiffusion(
        _Echo(),
        betas=linear_beta_schedule(1e-4, 0.02, T),
        num_timesteps=T,
        model_mean_type=ModelMeanType.V,
    )
    samples = dpmpp_sample(process, 2, SIZE, "cpu", num_steps=5)

    assert samples.shape == (2, *SIZE)
    assert torch.isfinite(samples).all()


def test_the_registry_holds_both_samplers():
    assert sampler_names() == ("ddim", "dpmpp")
    assert get_sampler("dpmpp") is dpmpp_sample
    assert set(SAMPLERS) == set(sampler_names())


def test_an_unregistered_sampler_says_what_is_available():
    with pytest.raises(ValueError, match=r"unknown sampler 'euler'.*ddim, dpmpp"):
        get_sampler("euler")
