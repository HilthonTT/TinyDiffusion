import pytest
import torch
import torch.nn as nn

from tinydiffusion.diffusion.guidance import (
    ClassifierFreeGuidance,
    Conditioned,
    conditioned,
    cycled_labels,
    drop_labels,
)

NUM_CLASSES = 4
NULL = NUM_CLASSES


class LabelEcho(nn.Module):
    """A net whose output is entirely determined by the label it was given.

    Returns the label broadcast over the image, so a guided prediction can be
    read straight off the output: ``uncond + scale * (cond - uncond)`` becomes
    ``null + scale * (label - null)`` in plain arithmetic.
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y.float().reshape(-1, 1, 1, 1).expand_as(x).clone()


class WideLabelEcho(LabelEcho):
    """The same, but emitting 2C channels: the label, then a fixed variance."""

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean = super().forward(x, t, y)
        # Distinguishable from any label so a leaked variance is obvious.
        return torch.cat([mean, torch.full_like(mean, -7.0)], dim=1)


@pytest.fixture
def batch():
    x = torch.randn(3, 1, 4, 4)
    t = torch.zeros(3, dtype=torch.long)
    y = torch.tensor([0, 1, 3])
    return x, t, y


def test_conditioned_passes_the_bound_labels_through(batch):
    x, t, y = batch
    out = Conditioned(LabelEcho(), y)(x, t)
    assert torch.equal(out[:, 0, 0, 0], y.float())


def test_guidance_extrapolates_away_from_the_null_prediction(batch):
    x, t, y = batch
    scale = 3.0
    out = ClassifierFreeGuidance(LabelEcho(), y, scale, NUM_CLASSES)(x, t)
    # null + scale * (label - null), with the echo net making both terms exact.
    expected = NULL + scale * (y.float() - NULL)
    assert torch.allclose(out[:, 0, 0, 0], expected)


@pytest.mark.parametrize(("scale", "expected"), [(1.0, "cond"), (0.0, "uncond")])
def test_the_guidance_endpoints_are_the_two_predictions(batch, scale, expected):
    x, t, y = batch
    out = ClassifierFreeGuidance(LabelEcho(), y, scale, NUM_CLASSES)(x, t)
    wanted = y.float() if expected == "cond" else torch.full_like(y.float(), NULL)
    assert torch.allclose(out[:, 0, 0, 0], wanted)


def test_guidance_leaves_a_learned_variance_alone(batch):
    # Extrapolating a log-variance has no interpretation and readily leaves the
    # schedule's bracket, so only the mean channels are guided.
    x, t, y = batch
    out = ClassifierFreeGuidance(WideLabelEcho(), y, 5.0, NUM_CLASSES)(x, t)

    assert out.shape == (3, 2, 4, 4)
    assert torch.allclose(out[:, 0, 0, 0], NULL + 5.0 * (y.float() - NULL))
    assert torch.all(out[:, 1] == -7.0)


def test_the_wrappers_adopt_the_networks_eval_mode(batch):
    # eval_mode() restores whatever mode it found on the module it was handed,
    # so a wrapper stuck at nn.Module's default would put an eval net back into
    # training after sampling.
    _, _, y = batch
    net = LabelEcho().eval()
    assert not Conditioned(net, y).training
    assert not ClassifierFreeGuidance(net, y, 2.0, NUM_CLASSES).training
    assert Conditioned(LabelEcho().train(), y).training


def test_conditioned_returns_an_unconditional_net_untouched():
    net = LabelEcho()
    assert conditioned(net, None) is net


def test_scale_of_one_skips_the_doubled_batch(batch):
    # Guidance at 1.0 multiplies the unconditional pass out of the result, so
    # paying for it would be pure waste.
    _, _, y = batch
    assert isinstance(conditioned(LabelEcho(), y, num_classes=NUM_CLASSES), Conditioned)
    assert isinstance(
        conditioned(LabelEcho(), y, num_classes=NUM_CLASSES, scale=2.0), ClassifierFreeGuidance
    )


def test_guidance_without_a_class_count_is_rejected(batch):
    _, _, y = batch
    with pytest.raises(ValueError, match="num_classes"):
        conditioned(LabelEcho(), y, scale=2.0)


@pytest.mark.parametrize(("p", "dropped"), [(0.0, False), (1.0, True)])
def test_label_dropout_at_its_extremes(p, dropped):
    labels = torch.tensor([0, 1, 2, 3])
    out = drop_labels(labels, NUM_CLASSES, p)
    assert torch.all(out == NULL) if dropped else torch.equal(out, labels)


def test_label_dropout_drops_roughly_the_requested_fraction():
    labels = torch.zeros(4000, dtype=torch.long)
    fraction = (drop_labels(labels, NUM_CLASSES, 0.25) == NULL).float().mean()
    assert fraction == pytest.approx(0.25, abs=0.03)


def test_label_dropout_does_not_touch_its_input():
    labels = torch.tensor([0, 1, 2, 3])
    drop_labels(labels, NUM_CLASSES, 1.0)
    assert torch.equal(labels, torch.tensor([0, 1, 2, 3]))


def test_cycled_labels_wraps_round_the_class_space():
    assert cycled_labels(5, 3, "cpu").tolist() == [0, 1, 2, 0, 1]
