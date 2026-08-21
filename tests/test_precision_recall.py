import pytest
import torch

from tinydiffusion.metrics.features import FeatureBank
from tinydiffusion.metrics.precision_recall import (
    DEFAULT_NEIGHBOURS,
    compute_precision_recall,
    precision_recall_from_features,
)

REAL_SEED = 0


def draw(n, dim=4, shift=0.0, scale=1.0, seed=REAL_SEED):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g) * scale + shift


@pytest.fixture
def real():
    return draw(400)


def test_samples_from_the_same_distribution_score_high_on_both(real):
    result = compute_precision_recall(draw(400, seed=1), real)
    assert result.precision > 0.9
    assert result.recall > 0.9


def test_a_collapsed_model_keeps_its_precision_and_loses_its_recall(real):
    # Everything it draws is plausible; it just draws the same small region
    # every time. A single number cannot tell this from the case below.
    result = compute_precision_recall(draw(400, scale=0.05, seed=1), real)
    assert result.precision > 0.9
    assert result.recall < 0.1


def test_an_over_dispersed_model_keeps_its_recall_and_loses_its_precision(real):
    result = compute_precision_recall(draw(400, scale=3.0, seed=1), real)
    assert result.precision < 0.5
    assert result.recall > 0.9


def test_a_model_that_misses_entirely_scores_zero_on_both(real):
    result = compute_precision_recall(draw(400, shift=20.0, seed=1), real)
    assert (result.precision, result.recall) == (0.0, 0.0)


def test_the_counts_are_reported_alongside_the_fractions(real):
    result = compute_precision_recall(draw(120, seed=1), real, neighbours=5)
    assert (result.num_generated, result.num_real) == (120, 400)
    assert result.neighbours == 5


def test_a_bigger_neighbourhood_never_shrinks_the_manifold(real):
    fake = draw(300, scale=1.6, seed=1)
    tight = compute_precision_recall(fake, real, neighbours=1)
    loose = compute_precision_recall(fake, real, neighbours=9)
    # Wider balls can only swallow more of the other set.
    assert loose.precision >= tight.precision


def test_uneven_set_sizes_are_allowed(real):
    assert compute_precision_recall(draw(37, seed=1), real).num_generated == 37


def test_the_default_neighbourhood_is_the_published_one():
    assert DEFAULT_NEIGHBOURS == 3


def test_a_set_no_larger_than_the_neighbourhood_is_refused(real):
    with pytest.raises(ValueError, match="3-th neighbour"):
        compute_precision_recall(draw(3, seed=1), real)


def test_a_meaningless_neighbourhood_is_refused(real):
    with pytest.raises(ValueError, match="neighbours must be"):
        compute_precision_recall(draw(40, seed=1), real, neighbours=0)


def test_sets_that_disagree_on_the_feature_dimension_are_refused():
    with pytest.raises(ValueError, match="dimensions differ"):
        compute_precision_recall(draw(40, dim=4), draw(40, dim=6, seed=1))


def test_features_that_are_not_a_matrix_are_refused(real):
    with pytest.raises(ValueError, match=r"\(n, dim\)"):
        compute_precision_recall(torch.zeros(40), real)


def test_scoring_banks_matches_scoring_their_features(real):
    fake_bank, real_bank = FeatureBank(4), FeatureBank(4)
    fake_bank.update(draw(200, scale=1.5, seed=1))
    real_bank.update(real)

    assert precision_recall_from_features(fake_bank, real_bank) == compute_precision_recall(
        fake_bank.features, real_bank.features
    )


def test_banks_that_disagree_on_the_feature_dimension_are_refused():
    with pytest.raises(ValueError, match="dimensions differ"):
        precision_recall_from_features(FeatureBank(4), FeatureBank(6))
