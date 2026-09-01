import math

import pytest
import torch

from tinydiffusion.metrics.inception_score import (
    DEFAULT_IS_SPLITS,
    inception_score_from_probs,
)


def one_hot(indices, classes):
    return torch.eye(classes)[torch.tensor(indices)]


def test_a_confident_and_varied_set_scores_the_class_count():
    """The best a set can do: every sample decisive, the classes evenly covered.

    KL(p_i || mean p) is then log(k) for every sample, so the exponentiated
    mean is exactly k. That upper bound is the whole shape of the metric.
    """
    probs = one_hot([0, 1, 2, 3] * 4, classes=4)
    result = inception_score_from_probs(probs, splits=1)
    assert result.mean == pytest.approx(4.0)


def test_a_confident_but_identical_set_scores_one():
    """Decisive and not varied: the marginal is the sample, so the KL is zero."""
    probs = one_hot([2] * 16, classes=4)
    assert inception_score_from_probs(probs, splits=1).mean == pytest.approx(1.0)


def test_an_undecided_set_scores_one():
    """Varied and not decisive: every sample is the marginal already."""
    probs = torch.full((16, 4), 0.25)
    assert inception_score_from_probs(probs, splits=1).mean == pytest.approx(1.0)


def test_a_zero_probability_does_not_produce_a_nan():
    """p log p is 0 at p = 0, but log p is not, and a one-hot row is all zeros bar one."""
    probs = one_hot([0, 1], classes=2)
    result = inception_score_from_probs(probs, splits=1)
    assert math.isfinite(result.mean)


def test_partial_coverage_lands_between_the_two_extremes():
    """Two classes out of four covered: the score is the classes actually used."""
    probs = one_hot([0, 1] * 8, classes=4)
    assert inception_score_from_probs(probs, splits=1).mean == pytest.approx(2.0)


def test_the_splits_are_disjoint_and_in_order():
    """Each chunk is scored on its own, so a set split by class scores 1 per chunk.

    Sixteen samples of class 0 followed by sixteen of class 1: as one set the
    score is 2, and as two ordered chunks each chunk is uniform and scores 1.
    That is what makes the split size part of the number rather than an
    implementation detail.
    """
    probs = one_hot([0] * 16 + [1] * 16, classes=2)
    assert inception_score_from_probs(probs, splits=1).mean == pytest.approx(2.0)
    assert inception_score_from_probs(probs, splits=2).mean == pytest.approx(1.0)


def test_the_spread_is_reported_across_splits():
    probs = one_hot([0, 1, 2, 3] * 8, classes=4)
    result = inception_score_from_probs(probs, splits=4)
    assert result.splits == 4
    assert result.split_size == 8
    assert result.std == pytest.approx(0.0, abs=1e-6)


def test_a_single_split_has_no_spread_rather_than_a_nan():
    """torch's unbiased std of one value is nan, which reads as a failure."""
    result = inception_score_from_probs(torch.full((4, 2), 0.5), splits=1)
    assert result.std == 0.0


def test_a_ragged_tail_is_dropped_rather_than_unbalancing_a_chunk():
    """The score depends on the chunk size, so chunks have to be one size."""
    result = inception_score_from_probs(torch.full((10, 2), 0.5), splits=3)
    assert result.split_size == 3


def test_the_default_split_count_is_the_published_one():
    assert DEFAULT_IS_SPLITS == 10


def test_a_non_matrix_is_refused():
    with pytest.raises(ValueError, match=r"\(N, classes\)"):
        inception_score_from_probs(torch.zeros(4), splits=1)


def test_an_empty_set_is_refused():
    with pytest.raises(ValueError, match="no probabilities"):
        inception_score_from_probs(torch.zeros(0, 4), splits=1)


def test_more_splits_than_samples_is_refused():
    with pytest.raises(ValueError, match="cannot split 4 samples into 5"):
        inception_score_from_probs(torch.full((4, 2), 0.5), splits=5)


def test_a_non_positive_split_count_is_refused():
    with pytest.raises(ValueError, match="splits must be positive"):
        inception_score_from_probs(torch.full((4, 2), 0.5), splits=0)
