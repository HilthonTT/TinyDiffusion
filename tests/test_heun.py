import pytest
import torch
import torch.nn as nn

from tinydiffusion.diffusion.ddim import ddim_sample, uniform_timesteps
from tinydiffusion.diffusion.ddpm import DDPM
from tinydiffusion.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType
from tinydiffusion.diffusion.heun import heun_sample
from tinydiffusion.diffusion.samplers import get_sampler
from tinydiffusion.diffusion.schedules import linear_beta_schedule

T = 100
SIZE = (1, 4, 4)


class _Echo(nn.Module):
    """A network whose output depends on its input, cheaply and smoothly."""

    def forward(self, x, t):
        return x * 0.5


class _Counting(nn.Module):
    """_Echo, but it says how many times it was called."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, t):
        self.calls += 1
        return x * 0.5


@pytest.fixture
def diffusion():
    return DDPM(_Echo(), betas=linear_beta_schedule(1e-4, 0.02, T), num_timesteps=T)


@pytest.fixture
def noise():
    return torch.randn(2, *SIZE)


def test_it_produces_the_shape_it_was_asked_for(diffusion):
    assert heun_sample(diffusion, 3, SIZE, "cpu", num_steps=5).shape == (3, *SIZE)


def test_the_chain_is_deterministic_given_its_latent(diffusion, noise):
    kwargs = {"num_steps": 6, "noise": noise}
    first = heun_sample(diffusion, 2, SIZE, "cpu", **kwargs)
    second = heun_sample(diffusion, 2, SIZE, "cpu", **kwargs)
    assert torch.equal(first, second)


def test_one_step_is_still_just_ddim(diffusion, noise):
    """A single step lands on the t=-1 sentinel, where there is nothing to correct.

    So it has to be exactly the DDIM step. Anything else means the shared
    first-order arithmetic has drifted.
    """
    a = heun_sample(diffusion, 2, SIZE, "cpu", num_steps=1, noise=noise)
    b = ddim_sample(diffusion, 2, SIZE, "cpu", num_steps=1, eta=0.0, noise=noise)
    assert torch.allclose(a, b, atol=1e-6)


def test_the_correction_shows_up_before_dpm_solver_s_would(diffusion, noise):
    """Second order from history needs three steps; from a corrector, two.

    ``dpmpp`` has no previous x_0 to extrapolate from until its third step, so
    a three-step chain is exactly where the two second-order solvers part
    company. Unclamped, because the clamp is what they would otherwise agree
    on: a correction hidden behind a clip is one this test cannot see.
    """
    kwargs = {"num_steps": 3, "noise": noise, "clip_denoised": False}
    a = heun_sample(diffusion, 2, SIZE, "cpu", **kwargs)
    b = ddim_sample(diffusion, 2, SIZE, "cpu", eta=0.0, **kwargs)
    assert not torch.allclose(a, b, atol=1e-6)


def test_it_spends_two_evaluations_a_step_bar_the_last(noise):
    """The cost that has to be known to compare it fairly against the others.

    Every step but the final one evaluates a predictor and a corrector; the
    final step denoises to alphabar = 1, where the x_0 prediction *is* the
    answer and a corrector would have nowhere to stand.
    """
    net = _Counting()
    process = DDPM(net, betas=linear_beta_schedule(1e-4, 0.02, T), num_timesteps=T)
    heun_sample(process, 2, SIZE, "cpu", num_steps=5, noise=noise)
    assert net.calls == 2 * 5 - 1


def test_a_short_chain_lands_near_a_long_one(noise):
    """The point of the solver: second-order convergence on the same ODE.

    Both samplers integrate the same trajectory, so a full 1,000-step DDIM run
    stands in for the exact answer. Run unclamped, since clipping x_0 puts a
    kink in the trajectory and a corrector evaluated across a kink is not
    measuring the same curve the predictor was.
    """
    long_schedule = DDPM(_Echo(), betas=linear_beta_schedule(1e-4, 0.02, 1000), num_timesteps=1000)
    kwargs = {"noise": noise, "clip_denoised": False}

    reference = ddim_sample(long_schedule, 2, SIZE, "cpu", num_steps=1000, eta=0.0, **kwargs)
    solver = heun_sample(long_schedule, 2, SIZE, "cpu", num_steps=20, **kwargs)
    baseline = ddim_sample(long_schedule, 2, SIZE, "cpu", num_steps=20, eta=0.0, **kwargs)

    solver_error = (solver - reference).abs().mean()
    baseline_error = (baseline - reference).abs().mean()
    assert solver_error < baseline_error / 5


def test_a_stochastic_eta_is_refused_rather_than_ignored(diffusion):
    with pytest.raises(ValueError, match="deterministic solver"):
        heun_sample(diffusion, 1, SIZE, "cpu", num_steps=4, eta=0.5)


def test_it_honours_an_explicit_timestep_grid(diffusion, noise):
    steps = uniform_timesteps(T, 4)
    a = heun_sample(diffusion, 2, SIZE, "cpu", timesteps=steps, noise=noise)
    b = heun_sample(diffusion, 2, SIZE, "cpu", num_steps=4, noise=noise)
    assert torch.equal(a, b)


def test_a_mis_shaped_latent_is_caught_before_the_chain(diffusion):
    with pytest.raises(ValueError, match="noise must be shaped"):
        heun_sample(diffusion, 3, SIZE, "cpu", num_steps=4, noise=torch.randn(2, *SIZE))


def test_it_routes_a_non_epsilon_process_through_p_mean_variance():
    """A v-prediction model has to be read as one, not as raw epsilon."""
    process = GaussianDiffusion(
        _Echo(),
        betas=linear_beta_schedule(1e-4, 0.02, T),
        num_timesteps=T,
        model_mean_type=ModelMeanType.V,
    )
    samples = heun_sample(process, 2, SIZE, "cpu", num_steps=5)

    assert samples.shape == (2, *SIZE)
    assert torch.isfinite(samples).all()


def test_it_is_reachable_by_name():
    assert get_sampler("heun") is heun_sample
