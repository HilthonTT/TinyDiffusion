import pytest
import torch
import torch.nn as nn

from tinydiffusion.diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossWeighting,
    ModelMeanType,
)
from tinydiffusion.diffusion.schedules import (
    cosine_beta_schedule,
    enforce_zero_terminal_snr,
    linear_beta_schedule,
)
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.model import build_model

T = 20
SHAPE = (2, 1, 8, 8)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x, t):
        return self.conv(x)


def _process(mean_type=ModelMeanType.V, betas=None, **kwargs):
    return GaussianDiffusion(
        _Net(),
        betas=linear_beta_schedule(1e-4, 0.02, T) if betas is None else betas,
        num_timesteps=T,
        model_mean_type=mean_type,
        **kwargs,
    )


def test_the_velocity_target_inverts_back_to_the_image():
    """_predict_xstart_from_v has to undo exactly what _loss_target built."""
    process = _process()
    x0 = torch.randn(*SHAPE)
    eps = torch.randn(*SHAPE)
    t = torch.randint(0, T, (SHAPE[0],))
    x_t = process.q_sample(x0, t, noise=eps)

    v = process._loss_target(x0, x_t, t, eps)
    assert torch.allclose(process._predict_xstart_from_v(x_t, t, v), x0, atol=1e-5)


def test_the_velocity_target_is_bounded_where_epsilon_is_not():
    """The point of v: a usable target at both ends of the schedule.

    Under a zero-terminal-SNR schedule the t=T epsilon target says nothing
    about x_0, and recovering one divides by a vanishing sqrt(alphabar). The
    velocity target stays finite and stays informative.
    """
    betas = enforce_zero_terminal_snr(cosine_beta_schedule(T))
    process = _process(betas=betas)
    x0 = torch.randn(*SHAPE)
    eps = torch.randn(*SHAPE)
    last = torch.full((SHAPE[0],), T - 1)
    x_t = process.q_sample(x0, last, noise=eps)

    v = process._loss_target(x0, x_t, last, eps)
    # At zero SNR the velocity is -x_0, which is as informative as a target
    # gets; epsilon prediction there would be regressing on pure noise.
    assert torch.allclose(v, -x0, atol=1e-3)
    assert torch.isfinite(process._predict_xstart_from_v(x_t, last, v)).all()


def test_a_v_prediction_model_trains_and_samples():
    process = _process()
    x = torch.randn(*SHAPE)

    terms = process.loss_terms(x)
    terms.loss.backward()

    assert terms.loss.isfinite()
    assert any(p.grad is not None and p.grad.any() for p in process.parameters())
    assert process.sample(2, SHAPE[1:], "cpu").shape == SHAPE


def test_the_config_wires_v_prediction_through():
    cfg = TrainConfig(predict="v", image_size=8, num_timesteps=T, sample_steps=4)
    model = build_model(cfg)

    assert isinstance(model, GaussianDiffusion)
    assert model.model_mean_type is ModelMeanType.V


def test_zero_snr_refuses_epsilon_prediction():
    """It leaves epsilon with nothing to predict at t=T; say so up front."""
    with pytest.raises(ValueError, match="epsilon prediction cannot invert"):
        TrainConfig(zero_snr=True, image_size=8, num_timesteps=T, sample_steps=4)


def test_zero_snr_reaches_the_built_schedule():
    cfg = TrainConfig(
        predict="v",
        zero_snr=True,
        schedule="linear",
        image_size=8,
        num_timesteps=T,
        sample_steps=4,
    )
    model = build_model(cfg)

    assert isinstance(model, GaussianDiffusion)
    assert model.alphabar_t[-1].sqrt() < 1e-3
    assert model.loss_weighting is LossWeighting.UNIFORM
