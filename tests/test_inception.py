import pytest
import torch

from tinydiffusion.metrics.inception import (
    INCEPTION_DIM,
    INCEPTION_SIZE,
    FeatureExtractor,
    InceptionFeatures,
)


@pytest.fixture(scope="module")
def untrained():
    """Inception with random weights: the same graph, without a 100MB download."""
    return InceptionFeatures(weights=None)


def test_preprocess_expands_greyscale_and_resizes(untrained):
    out = untrained.preprocess(torch.zeros(2, 1, 8, 8))
    assert out.shape == (2, 3, INCEPTION_SIZE, INCEPTION_SIZE)


def test_preprocess_keeps_three_channel_input(untrained):
    out = untrained.preprocess(torch.zeros(2, 3, INCEPTION_SIZE, INCEPTION_SIZE))
    assert out.shape == (2, 3, INCEPTION_SIZE, INCEPTION_SIZE)


def test_preprocess_normalises_to_imagenet_statistics(untrained):
    # A [-1, 1] image of -1 is black, which ImageNet normalisation maps to
    # -mean/std per channel.
    out = untrained.preprocess(torch.full((1, 1, 8, 8), -1.0))
    want = -untrained.mean / untrained.std
    assert torch.allclose(out[0, :, 0, 0], want.flatten(), atol=1e-6)


def test_preprocess_clamps_out_of_range_input(untrained):
    # The sampler can overshoot [-1, 1]; denormalize clamps, so a wild value
    # lands on white rather than off the end of the scale.
    hot = untrained.preprocess(torch.full((1, 1, 8, 8), 5.0))
    white = untrained.preprocess(torch.ones(1, 1, 8, 8))
    assert torch.allclose(hot, white)


@pytest.mark.parametrize("bad", [torch.zeros(3, 8, 8), torch.zeros(1, 2, 8, 8)])
def test_preprocess_rejects_unusable_batches(untrained, bad):
    with pytest.raises(ValueError):
        untrained.preprocess(bad)


def test_forward_returns_pooled_features(untrained):
    feats = untrained(torch.randn(2, 1, 32, 32).clamp(-1, 1))
    assert feats.shape == (2, INCEPTION_DIM)
    assert feats.requires_grad is False


def test_it_satisfies_the_extractor_protocol(untrained):
    assert isinstance(untrained, FeatureExtractor)
    assert untrained.dim == INCEPTION_DIM


def test_it_stays_in_eval_mode(untrained):
    # Dropout would make the features non-deterministic, which would show up as
    # score noise rather than as an obvious failure.
    assert untrained.net.training is False
