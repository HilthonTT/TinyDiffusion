import pytest
import torch
import torch.nn as nn

from tinydiffusion.diffusion.ddim import ddim_sample, uniform_timesteps
from tinydiffusion.diffusion.ddpm import DDPM
from tinydiffusion.diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType
from tinydiffusion.diffusion.plms import PLMS_COEFFICIENTS, plms_sample
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


def test_every_coefficient_row_sums_to_one():
    """Adams-Bashforth extrapolates; it does not rescale.

    A row that summed to anything else would multiply the noise estimate by a
    constant at every step, which reads as a schedule bug rather than as the
    typo it is.
    """
    for weights in PLMS_COEFFICIENTS:
        assert sum(weights) == pytest.approx(1.0)


def test_the_rows_ramp_one_order_at_a_time():
    """Row k reads k past estimates, so it is usable exactly when k are held."""
    for order, weights in enumerate(PLMS_COEFFICIENTS, start=1):
        assert len(weights) == order


def test_it_produces_the_shape_it_was_asked_for(diffusion):
    assert plms_sample(diffusion, 3, SIZE, "cpu", num_steps=5).shape == (3, *SIZE)


def test_the_chain_is_deterministic_given_its_latent(diffusion, noise):
    kwargs = {"num_steps": 6, "noise": noise}
    first = plms_sample(diffusion, 2, SIZE, "cpu", **kwargs)
    second = plms_sample(diffusion, 2, SIZE, "cpu", **kwargs)
    assert torch.equal(first, second)


def test_one_step_is_still_just_ddim(diffusion, noise):
    """With no history there is nothing to extrapolate from, so it is a DDIM step."""
    a = plms_sample(diffusion, 2, SIZE, "cpu", num_steps=1, noise=noise)
    b = ddim_sample(diffusion, 2, SIZE, "cpu", num_steps=1, eta=0.0, noise=noise)
    assert torch.allclose(a, b, atol=1e-6)


def test_the_extrapolation_starts_as_soon_as_there_is_history(diffusion, noise):
    """One step of history is enough for the second-order row.

    Unclamped, because the clamp is what the two samplers would otherwise
    agree on: at this schedule's step sizes both predictions saturate, and an
    extrapolation hidden behind a clip is one this test cannot see.
    """
    kwargs = {"num_steps": 3, "noise": noise, "clip_denoised": False}
    a = plms_sample(diffusion, 2, SIZE, "cpu", **kwargs)
    b = ddim_sample(diffusion, 2, SIZE, "cpu", eta=0.0, **kwargs)
    assert not torch.allclose(a, b, atol=1e-6)


def test_it_spends_one_evaluation_a_step(noise):
    """The whole point: fourth order without a second network call.

    A solver that reached its order by evaluating twice would be a different
    trade, and this test is what stops the history quietly being replaced by
    one.
    """
    net = _Counting()
    process = DDPM(net, betas=linear_beta_schedule(1e-4, 0.02, T), num_timesteps=T)
    plms_sample(process, 2, SIZE, "cpu", num_steps=7, noise=noise)
    assert net.calls == 7


def test_a_short_chain_lands_near_a_long_one(noise):
    """Fourth-order convergence on the trajectory DDIM integrates first-order.

    A full 1,000-step DDIM run stands in for the exact answer. Run unclamped:
    clipping x_0 kinks the trajectory, and a cubic fitted across a kink is not
    describing the curve the next step is taken along.
    """
    long_schedule = DDPM(_Echo(), betas=linear_beta_schedule(1e-4, 0.02, 1000), num_timesteps=1000)
    kwargs = {"noise": noise, "clip_denoised": False}

    reference = ddim_sample(long_schedule, 2, SIZE, "cpu", num_steps=1000, eta=0.0, **kwargs)
    solver = plms_sample(long_schedule, 2, SIZE, "cpu", num_steps=20, **kwargs)
    baseline = ddim_sample(long_schedule, 2, SIZE, "cpu", num_steps=20, eta=0.0, **kwargs)

    solver_error = (solver - reference).abs().mean()
    baseline_error = (baseline - reference).abs().mean()
    assert solver_error < baseline_error / 8


def test_a_stochastic_eta_is_refused_rather_than_ignored(diffusion):
    with pytest.raises(ValueError, match="deterministic solver"):
        plms_sample(diffusion, 1, SIZE, "cpu", num_steps=4, eta=0.5)


def test_it_honours_an_explicit_timestep_grid(diffusion, noise):
    steps = uniform_timesteps(T, 4)
    a = plms_sample(diffusion, 2, SIZE, "cpu", timesteps=steps, noise=noise)
    b = plms_sample(diffusion, 2, SIZE, "cpu", num_steps=4, noise=noise)
    assert torch.equal(a, b)


def test_a_mis_shaped_latent_is_caught_before_the_chain(diffusion):
    with pytest.raises(ValueError, match="noise must be shaped"):
        plms_sample(diffusion, 3, SIZE, "cpu", num_steps=4, noise=torch.randn(2, *SIZE))


def test_it_routes_a_non_epsilon_process_through_p_mean_variance():
    """A v-prediction model has to be read as one, not as raw epsilon."""
    process = GaussianDiffusion(
        _Echo(),
        betas=linear_beta_schedule(1e-4, 0.02, T),
        num_timesteps=T,
        model_mean_type=ModelMeanType.V,
    )
    samples = plms_sample(process, 2, SIZE, "cpu", num_steps=5)

    assert samples.shape == (2, *SIZE)
    assert torch.isfinite(samples).all()


def test_the_history_never_outgrows_the_longest_formula(diffusion, noise):
    """A long chain must not accumulate estimates it has no coefficients for.

    The buffer is trimmed each step, and an untrimmed one would index past the
    coefficient table on step five rather than degrade quietly.
    """
    samples = plms_sample(diffusion, 2, SIZE, "cpu", num_steps=30, noise=noise)
    assert torch.isfinite(samples).all()


def test_it_is_reachable_by_name():
    assert get_sampler("plms") is plms_sample
