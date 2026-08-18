import pytest
import torch
import torch.nn as nn

from tinydiffusion.diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    LossWeighting,
    ModelMeanType,
    ModelVarType,
)
from tinydiffusion.diffusion.schedules import linear_beta_schedule
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.model import build_model

T = 20
GAMMA = 5.0


class _Net(nn.Module):
    def __init__(self, out_mult: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(1, out_mult, 3, padding=1)

    def forward(self, x, t):
        return self.conv(x)


def _process(mean_type=ModelMeanType.EPSILON, weighting=LossWeighting.MIN_SNR, **kwargs):
    return GaussianDiffusion(
        _Net(),
        betas=linear_beta_schedule(1e-4, 0.02, T),
        num_timesteps=T,
        model_mean_type=mean_type,
        loss_weighting=weighting,
        min_snr_gamma=GAMMA,
        **kwargs,
    )


def test_uniform_weighting_leaves_every_timestep_alone():
    process = _process(weighting=LossWeighting.UNIFORM)
    assert torch.equal(process.loss_weights(torch.arange(T)), torch.ones(T))


@pytest.mark.parametrize(
    ("mean_type", "expected"),
    [
        (ModelMeanType.START_X, lambda snr: snr.clamp(max=GAMMA)),
        (ModelMeanType.EPSILON, lambda snr: snr.clamp(max=GAMMA) / snr),
        (ModelMeanType.V, lambda snr: snr.clamp(max=GAMMA) / (snr + 1.0)),
    ],
)
def test_the_weight_is_stated_in_the_space_the_target_lives_in(mean_type, expected):
    process = _process(mean_type)
    t = torch.arange(T)
    assert torch.allclose(process.loss_weights(t), expected(process.snr[t]))


def test_the_clamp_is_what_bounds_the_low_noise_end():
    """Without it, x_0-space weight grows without limit as t goes to 0."""
    process = _process(ModelMeanType.START_X)
    weights = process.loss_weights(torch.arange(T))

    assert weights.max() <= GAMMA
    # t=0 is the highest-SNR step, and the one the clamp is there to hold back.
    assert process.snr[0] > GAMMA
    assert weights[0] == pytest.approx(GAMMA)


def test_weighting_changes_the_gradient_but_not_the_logged_mse():
    """The logged per-timestep MSE has to stay comparable across buckets."""
    x = torch.randn(2, 1, 8, 8)
    t = torch.zeros(2, dtype=torch.long)
    noise = torch.randn_like(x)

    plain = _process(weighting=LossWeighting.UNIFORM)
    weighted = _process()
    weighted.load_state_dict(plain.state_dict())

    a = plain.training_losses(x, t, noise=noise)
    b = weighted.training_losses(x, t, noise=noise)

    assert torch.allclose(a["mse"], b["mse"])
    assert not torch.allclose(a["loss"], b["loss"])
    assert torch.allclose(b["loss"], b["mse"] * weighted.loss_weights(t))


def test_min_snr_is_refused_where_it_has_no_meaning():
    with pytest.raises(ValueError, match="which a variational objective does not have"):
        _process(loss_type=LossType.KL)
    with pytest.raises(ValueError, match="not defined for PREVIOUS_X"):
        _process(ModelMeanType.PREVIOUS_X)


def test_a_learned_variance_keeps_its_own_unweighted_term():
    """MIN_SNR reweights L_simple; the bound that trains the variance is its own scale."""
    process = _process(
        model_var_type=ModelVarType.LEARNED_RANGE,
        loss_type=LossType.RESCALED_MSE,
    )
    process.model = _Net(out_mult=2)
    x = torch.randn(2, 1, 8, 8)
    t = torch.zeros(2, dtype=torch.long)

    terms = process.training_losses(x, t)
    assert torch.allclose(terms["loss"], terms["mse"] * process.loss_weights(t) + terms["vb"])


def test_the_config_wires_the_weighting_through():
    cfg = TrainConfig(
        loss_weighting="min_snr", min_snr_gamma=3.0, image_size=8, num_timesteps=T, sample_steps=4
    )
    model = build_model(cfg)

    assert isinstance(model, GaussianDiffusion)
    assert model.loss_weighting is LossWeighting.MIN_SNR
    assert model.min_snr_gamma == 3.0


def test_the_config_rejects_a_weighting_it_cannot_apply():
    with pytest.raises(ValueError, match="does not have"):
        TrainConfig(
            loss_weighting="min_snr", objective="kl", image_size=8, num_timesteps=T, sample_steps=4
        )
    with pytest.raises(ValueError, match="min_snr_gamma must be positive"):
        TrainConfig(min_snr_gamma=0.0, image_size=8, num_timesteps=T, sample_steps=4)
