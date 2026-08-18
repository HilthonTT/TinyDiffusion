import pytest
import torch
import torch.nn as nn

from tinydiffusion.diffusion.gaussian_diffusion import GaussianDiffusion
from tinydiffusion.diffusion.schedules import linear_beta_schedule
from tinydiffusion.diffusion.timesteps import (
    LossSecondMomentResampler,
    UniformSampler,
    timestep_sampler,
)
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.model import build_model

T = 8


def _warm(sampler: LossSecondMomentResampler, losses: list[float]) -> None:
    """Fill every timestep's history with its entry from `losses`."""
    t = torch.arange(len(losses))
    for _ in range(sampler.history):
        sampler.update(t, torch.tensor(losses))


def test_a_uniform_draw_stays_in_range_and_costs_nothing():
    t, weights = UniformSampler(T).sample(64, "cpu")

    assert t.dtype == torch.long
    assert t.min() >= 0 and t.max() < T
    assert torch.equal(weights, torch.ones(64))


def test_the_proposal_is_uniform_until_every_timestep_has_a_history():
    sampler = LossSecondMomentResampler(T, history=2)
    assert not sampler.warm
    assert torch.allclose(sampler.weights(), torch.full((T,), 1 / T, dtype=torch.float64))

    # One timestep short of warm is still not warm.
    for _ in range(2):
        sampler.update(torch.arange(T - 1), torch.ones(T - 1))
    assert not sampler.warm

    sampler.update(torch.tensor([T - 1]), torch.ones(1))
    sampler.update(torch.tensor([T - 1]), torch.ones(1))
    assert sampler.warm


def test_the_loud_timesteps_get_the_samples():
    sampler = LossSecondMomentResampler(T, history=2)
    losses = [100.0] + [1.0] * (T - 1)
    _warm(sampler, losses)

    probs = sampler.weights()
    assert probs.argmax() == 0
    assert probs[0] > 10 * probs[1]
    assert probs.sum() == pytest.approx(1.0)


def test_the_weights_undo_the_proposal():
    """1/(T*p) is what keeps a non-uniform draw an unbiased estimator."""
    sampler = LossSecondMomentResampler(T, history=2)
    _warm(sampler, [100.0] + [1.0] * (T - 1))

    t, weights = sampler.sample(256, "cpu")
    probs = sampler.weights()
    assert torch.allclose(weights.double(), 1.0 / (T * probs[t]))
    # The oversampled timestep is the one counted down the most.
    assert weights[t == 0].max() < 1.0


def test_a_uniform_history_leaves_the_draw_uniform():
    sampler = LossSecondMomentResampler(T, history=2)
    _warm(sampler, [3.0] * T)
    assert torch.allclose(sampler.weights(), torch.full((T,), 1 / T, dtype=torch.float64))


def test_the_history_is_a_window_not_a_running_total():
    sampler = LossSecondMomentResampler(T, history=2)
    _warm(sampler, [100.0] * T)
    for _ in range(2):
        sampler.update(torch.arange(T), torch.ones(T))

    # The 100s have been pushed out, so every timestep is back to level.
    assert torch.allclose(sampler.weights(), torch.full((T,), 1 / T, dtype=torch.float64))


def test_no_timestep_can_be_starved_completely():
    sampler = LossSecondMomentResampler(T, history=1, uniform_prob=0.1)
    _warm(sampler, [1.0] + [0.0] * (T - 1))
    assert (sampler.weights() > 0).all()


def test_a_zero_history_does_not_divide_by_itself():
    sampler = LossSecondMomentResampler(T, history=1)
    _warm(sampler, [0.0] * T)
    assert torch.allclose(sampler.weights(), torch.full((T,), 1 / T, dtype=torch.float64))


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError, match="history must be positive"):
        LossSecondMomentResampler(T, history=0)
    with pytest.raises(ValueError, match="uniform_prob must lie"):
        LossSecondMomentResampler(T, uniform_prob=1.0)
    with pytest.raises(ValueError, match="unknown timestep_sampler"):
        timestep_sampler("adaptive", T)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x, t):
        return self.conv(x)


def test_the_process_draws_from_its_sampler_and_feeds_it_back():
    resampler = LossSecondMomentResampler(T, history=1)
    process = GaussianDiffusion(
        _Net(),
        betas=linear_beta_schedule(1e-4, 0.02, T),
        num_timesteps=T,
        timestep_sampler=resampler,
    )

    terms = process.loss_terms(torch.randn(16, 1, 8, 8))
    assert terms.timesteps.min() >= 0 and terms.timesteps.max() < T
    # Every timestep the draw touched now has a recorded loss.
    assert sum(resampler._counts.tolist()) > 0


def test_the_config_wires_the_resampler_through():
    cfg = TrainConfig(
        timestep_sampler="loss_second_moment",
        image_size=8,
        num_timesteps=T,
        sample_steps=4,
        val_steps=4,
    )
    model = build_model(cfg)

    assert isinstance(model, GaussianDiffusion)
    assert isinstance(model.timestep_sampler, LossSecondMomentResampler)


def test_the_config_rejects_a_sampler_that_does_not_exist():
    with pytest.raises(ValueError, match="unknown timestep_sampler"):
        TrainConfig(
            timestep_sampler="magic", image_size=8, num_timesteps=T, sample_steps=4, val_steps=4
        )
