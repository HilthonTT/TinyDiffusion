import pytest
import torch
import torch.nn as nn

from tinydiffusion.diffusion.guidance import (
    ClassifierFreeGuidance,
    Conditioned,
    conditioned,
    rescale_guided,
)
from tinydiffusion.training.config import TrainConfig

NUM_CLASSES = 4
NULL = NUM_CLASSES


class ScaledEcho(nn.Module):
    """A net whose output is the input scaled by the label, plus one.

    Unlike ``LabelEcho`` in test_guidance.py, this leaves real spatial variance
    in the prediction — a constant output has zero standard deviation, which is
    exactly the quantity the rescale is defined in terms of.
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * (y.float().reshape(-1, 1, 1, 1) + 1.0)


class WideScaledEcho(ScaledEcho):
    """The same, emitting 2C channels: the prediction, then a fixed variance."""

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean = super().forward(x, t, y)
        return torch.cat([mean, torch.full_like(mean, -7.0)], dim=1)


@pytest.fixture
def batch():
    x = torch.randn(3, 1, 8, 8)
    t = torch.zeros(3, dtype=torch.long)
    labels = torch.tensor([0, 1, 2])
    return x, t, labels


def std_per_sample(x: torch.Tensor) -> torch.Tensor:
    return x.std(dim=list(range(1, x.ndim)))


def test_rescale_of_zero_is_the_identity():
    guided, cond = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
    assert torch.equal(rescale_guided(guided, cond, 0.0), guided)


def test_full_rescale_matches_the_conditional_scale():
    guided = torch.randn(4, 1, 8, 8) * 5.0
    cond = torch.randn(4, 1, 8, 8)
    out = rescale_guided(guided, cond, 1.0)
    torch.testing.assert_close(std_per_sample(out), std_per_sample(cond))


def test_rescale_is_a_linear_blend_of_the_two_ends():
    guided = torch.randn(4, 1, 8, 8) * 5.0
    cond = torch.randn(4, 1, 8, 8)
    full = rescale_guided(guided, cond, 1.0)
    half = rescale_guided(guided, cond, 0.5)
    torch.testing.assert_close(half, 0.5 * full + 0.5 * guided)


def test_the_correction_is_per_sample():
    # One sample over-extrapolated ten times as far as the other. A correction
    # taken over the whole batch would leave both wrong; a per-sample one puts
    # each on its own conditional scale.
    cond = torch.randn(2, 1, 8, 8)
    guided = torch.stack([cond[0] * 2.0, cond[1] * 20.0])
    out = rescale_guided(guided, cond, 1.0)
    torch.testing.assert_close(std_per_sample(out), std_per_sample(cond))


def test_a_degenerate_prediction_does_not_produce_nan():
    # A constant guided prediction has zero standard deviation. Dividing by it
    # would put NaN into the chain with nothing downstream to report it.
    guided = torch.full((2, 1, 4, 4), 3.0)
    cond = torch.randn(2, 1, 4, 4)
    out = rescale_guided(guided, cond, 0.7)
    assert torch.isfinite(out).all()


def test_rescale_pulls_a_guided_prediction_back_towards_the_conditional(batch):
    x, t, labels = batch
    plain = ClassifierFreeGuidance(ScaledEcho(), labels, 5.0, NUM_CLASSES)(x, t)
    corrected = ClassifierFreeGuidance(ScaledEcho(), labels, 5.0, NUM_CLASSES, 0.7)(x, t)
    cond = Conditioned(ScaledEcho(), labels)(x, t)

    # Guidance at scale 5 inflates the scale; the correction shrinks it back
    # part of the way, without going all the way to the conditional.
    assert (std_per_sample(plain) > std_per_sample(cond)).all()
    assert (std_per_sample(corrected) < std_per_sample(plain)).all()
    assert (std_per_sample(corrected) > std_per_sample(cond)).all()


def test_rescale_leaves_a_learned_variance_alone(batch):
    x, t, labels = batch
    out = ClassifierFreeGuidance(WideScaledEcho(), labels, 5.0, NUM_CLASSES, 1.0)(x, t)

    mean, variance = out.split(x.shape[1], dim=1)
    assert torch.equal(variance, torch.full_like(variance, -7.0))
    # The mean channels are corrected against the conditional mean, not against
    # a scale the untouched variance channels would have skewed.
    cond_mean = Conditioned(WideScaledEcho(), labels)(x, t).split(x.shape[1], dim=1)[0]
    torch.testing.assert_close(std_per_sample(mean), std_per_sample(cond_mean))


def test_scale_of_one_ignores_the_rescale(batch):
    # At scale 1 the guided prediction *is* the conditional one, so there is
    # nothing to correct and the cheap single-width branch stays correct.
    _, _, labels = batch
    wrapped = conditioned(ScaledEcho(), labels, num_classes=NUM_CLASSES, scale=1.0, rescale=0.7)
    assert isinstance(wrapped, Conditioned)


def test_conditioned_passes_the_rescale_through(batch):
    _, _, labels = batch
    wrapped = conditioned(ScaledEcho(), labels, num_classes=NUM_CLASSES, scale=3.0, rescale=0.7)
    assert isinstance(wrapped, ClassifierFreeGuidance)
    assert wrapped.rescale == 0.7


@pytest.mark.parametrize("rescale", [-0.1, 1.5])
def test_a_rescale_outside_the_unit_interval_is_rejected(batch, rescale):
    _, _, labels = batch
    with pytest.raises(ValueError, match="rescale"):
        conditioned(ScaledEcho(), labels, num_classes=NUM_CLASSES, scale=3.0, rescale=rescale)


@pytest.mark.parametrize("rescale", [-0.1, 1.5])
def test_the_config_rejects_a_rescale_outside_the_unit_interval(rescale):
    with pytest.raises(ValueError, match="guidance_rescale"):
        TrainConfig(num_classes=10, guidance=3.0, guidance_rescale=rescale)


def test_the_config_rejects_a_rescale_with_nothing_to_correct():
    # It would be a silent no-op otherwise: at guidance 1 there is no
    # extrapolation, so the config would promise a correction it cannot make.
    with pytest.raises(ValueError, match="nothing to correct"):
        TrainConfig(num_classes=10, guidance=1.0, guidance_rescale=0.7)


def test_the_config_defaults_to_plain_guidance():
    assert TrainConfig().guidance_rescale == 0.0
