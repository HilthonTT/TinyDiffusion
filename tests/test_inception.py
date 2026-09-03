import pytest
import torch

from tinydiffusion.metrics.inception import (
    INCEPTION_CLASSES,
    INCEPTION_DIM,
    INCEPTION_SIZE,
    SFID_DIM,
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
    out = untrained.preprocess(torch.full((1, 1, 8, 8), -1.0))
    want = -untrained.mean / untrained.std
    assert torch.allclose(out[0, :, 0, 0], want.flatten(), atol=1e-6)


def test_preprocess_clamps_out_of_range_input(untrained):
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
    assert untrained.net.training is False


def test_analyse_returns_all_three_heads(untrained):
    outputs = untrained.analyse(torch.zeros(2, 1, 8, 8))
    assert outputs.pool.shape == (2, INCEPTION_DIM)
    assert outputs.spatial.shape == (2, SFID_DIM)
    assert outputs.probs.shape == (2, INCEPTION_CLASSES)


def test_the_pooled_head_is_what_a_plain_call_returns(untrained):
    """One pass has to give the same features as the pass FID already took.

    If it did not, `--sfid` would silently move the FID it is reported beside.
    """
    images = torch.randn(2, 1, 8, 8).clamp(-1, 1)
    assert torch.allclose(untrained.analyse(images).pool, untrained(images), atol=1e-5)


def test_the_probabilities_are_a_distribution(untrained):
    probs = untrained.analyse(torch.zeros(2, 1, 8, 8)).probs
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert (probs >= 0).all()


def test_the_spatial_head_keeps_geometry_the_pooled_one_averages_away(untrained):
    """The point of sFID: two images with the same content differently arranged.

    Rolling an image shifts the feature map without changing much of what is in
    it, so the spatial reading has to move further than the pooled one. Without
    that the second score would be a noisier copy of the first.
    """
    images = torch.randn(1, 1, 64, 64).clamp(-1, 1)
    rolled = images.roll(shifts=24, dims=-1)

    first, second = untrained.analyse(images), untrained.analyse(rolled)
    pooled_shift = (first.pool - second.pool).norm() / first.pool.norm()
    spatial_shift = (first.spatial - second.spatial).norm() / first.spatial.norm()
    assert spatial_shift > pooled_shift


def test_analyse_leaves_no_hook_behind(untrained):
    """A hook left registered would append to a dead list on the next call."""
    untrained.analyse(torch.zeros(1, 1, 8, 8))
    untrained.analyse(torch.zeros(1, 1, 8, 8))
    assert not untrained.net.Mixed_6e._forward_hooks


def test_the_classifier_survives_being_moved_aside(untrained):
    """It is the head FID discards and the Inception Score needs."""
    assert untrained.net.fc.__class__.__name__ == "Identity"
    assert untrained.classifier.out_features == INCEPTION_CLASSES


def test_the_network_rescales_imagenet_input_to_what_its_weights_expect(untrained):
    """torchvision's ported Inception weights were trained on [-1, 1] input.

    ``transform_input`` is what turns the ImageNet-normalised batch from
    ``preprocess`` into that convention; without it every feature the metrics
    read comes from a mis-scaled forward pass.
    """
    assert untrained.net.transform_input is True
