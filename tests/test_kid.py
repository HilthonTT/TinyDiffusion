import pytest
import torch

from tinydiffusion.metrics.features import FeatureBank
from tinydiffusion.metrics.kid import (
    DEFAULT_KID_SUBSET_SIZE,
    DEFAULT_KID_SUBSETS,
    _mmd2,
    compute_kid,
    kid_from_features,
)


def draw(n, dim=8, shift=0.0, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g) * scale + shift


def naive_mmd2(x, y):
    """The estimator written out, as the paper states it, for the fast one to match."""
    dim = x.shape[1]
    kxx = (x @ x.T / dim + 1) ** 3
    kyy = (y @ y.T / dim + 1) ** 3
    kxy = (x @ y.T / dim + 1) ** 3
    m, n = x.shape[0], y.shape[0]
    return (
        (kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))
        + (kyy.sum() - kyy.diagonal().sum()) / (n * (n - 1))
        - 2 * kxy.mean()
    )


def test_the_chunked_estimator_matches_the_written_out_one():
    x, y = draw(37).double(), draw(53, shift=0.5, seed=1).double()
    assert _mmd2(x, y).item() == pytest.approx(naive_mmd2(x, y).item(), rel=1e-9)


def test_uneven_set_sizes_are_allowed():
    assert compute_kid(draw(40), draw(90, seed=1), subsets=2, subset_size=20).subsets == 2


@pytest.mark.parametrize("n", [25, 100, 400])
def test_the_estimate_does_not_drift_with_the_sample_count(n):
    g = torch.Generator().manual_seed(7)
    values = [
        float(_mmd2(torch.randn(n, 8, generator=g), torch.randn(n, 8, generator=g)))
        for _ in range(200)
    ]
    mean = sum(values) / len(values)
    standard_error = (sum((v - mean) ** 2 for v in values) / len(values) / len(values)) ** 0.5

    assert abs(mean) < 3 * standard_error


def test_two_different_distributions_score_above_zero():
    apart = compute_kid(draw(200, shift=1.0), draw(200, seed=1), subsets=4, subset_size=100)
    together = compute_kid(draw(200, seed=2), draw(200, seed=1), subsets=4, subset_size=100)
    assert apart.mean > together.mean
    assert apart.mean > 0


def test_a_subset_larger_than_the_data_is_clamped_to_it():
    result = compute_kid(draw(30), draw(50, seed=1), subsets=2, subset_size=1000)
    assert result.subset_size == 30


def test_the_same_generator_gives_the_same_score():
    fake, real = draw(80), draw(80, shift=0.3, seed=1)
    scores = [
        compute_kid(
            fake,
            real,
            subsets=8,
            subset_size=20,
            generator=torch.Generator().manual_seed(3),
        ).mean
        for _ in range(2)
    ]
    assert scores[0] == scores[1]


def test_different_subsets_disagree_enough_to_report_a_spread():
    result = compute_kid(draw(200), draw(200, shift=0.5, seed=1), subsets=16, subset_size=32)
    assert result.std > 0


def test_a_single_subset_reports_no_spread_rather_than_a_nan():
    result = compute_kid(draw(40), draw(40, seed=1), subsets=1, subset_size=20)
    assert result.std == 0.0


def test_the_defaults_are_the_published_ones():
    assert (DEFAULT_KID_SUBSETS, DEFAULT_KID_SUBSET_SIZE) == (100, 1000)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"subsets": 0}, "subsets must be"),
        ({"subset_size": 1}, "subset_size must be"),
    ],
)
def test_a_meaningless_subset_request_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        compute_kid(draw(20), draw(20, seed=1), **kwargs)


def test_sets_that_disagree_on_the_feature_dimension_are_refused():
    with pytest.raises(ValueError, match="dimensions differ"):
        compute_kid(draw(20, dim=8), draw(20, dim=6, seed=1))


def test_a_single_vector_leaves_the_estimator_undefined():
    with pytest.raises(ValueError, match="at least 2"):
        compute_kid(draw(1), draw(20, seed=1))


def test_features_that_are_not_a_matrix_are_refused():
    with pytest.raises(ValueError, match=r"\(n, dim\)"):
        compute_kid(torch.zeros(20), draw(20))


def test_scoring_banks_matches_scoring_their_features():
    fake, real = FeatureBank(8), FeatureBank(8)
    fake.update(draw(60))
    real.update(draw(60, shift=0.4, seed=1))

    banked = kid_from_features(
        fake, real, subsets=4, subset_size=30, generator=torch.Generator().manual_seed(0)
    )
    direct = compute_kid(
        fake.features,
        real.features,
        subsets=4,
        subset_size=30,
        generator=torch.Generator().manual_seed(0),
    )
    assert banked == direct


def test_banks_that_disagree_on_the_feature_dimension_are_refused():
    with pytest.raises(ValueError, match="dimensions differ"):
        kid_from_features(FeatureBank(8), FeatureBank(6))
