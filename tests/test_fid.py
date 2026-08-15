import pytest
import torch

from tinydiffusion.metrics import FeatureStats, compute_fid, fid_from_stats


def gaussian(n, dim, *, mean=0.0, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g) * scale + mean


def reference_stats(feats):
    mu = feats.double().mean(0)
    cov = feats.double().T.cov()
    return mu, cov


def test_identical_gaussians_score_zero():
    mu = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    a = torch.randn(3, 3, dtype=torch.float64)
    cov = a @ a.T + torch.eye(3, dtype=torch.float64)
    assert compute_fid(mu, cov, mu.clone(), cov.clone()).item() == pytest.approx(0.0, abs=1e-8)


def test_mean_shift_only_is_squared_distance():
    dim = 4
    cov = torch.eye(dim, dtype=torch.float64)
    mu1 = torch.zeros(dim, dtype=torch.float64)
    mu2 = torch.full((dim,), 0.5, dtype=torch.float64)
    assert compute_fid(mu1, cov, mu2, cov).item() == pytest.approx(dim * 0.25)


def test_scaled_covariance_matches_closed_form():
    # For sigma1 = I and sigma2 = s^2 I with equal means, the distance reduces
    # to d * (1 - s)^2.
    dim, s = 5, 3.0
    mu = torch.zeros(dim, dtype=torch.float64)
    eye = torch.eye(dim, dtype=torch.float64)
    got = compute_fid(mu, eye, mu, eye * s**2).item()
    assert got == pytest.approx(dim * (1 - s) ** 2)


def test_symmetric_in_its_arguments():
    mu1, sigma1 = reference_stats(gaussian(64, 6, seed=1))
    mu2, sigma2 = reference_stats(gaussian(64, 6, mean=0.3, scale=1.5, seed=2))
    forward = compute_fid(mu1, sigma1, mu2, sigma2).item()
    backward = compute_fid(mu2, sigma2, mu1, sigma1).item()
    assert forward == pytest.approx(backward)


def test_singular_covariance_is_handled():
    # A constant feature makes the covariance rank-deficient, which is the
    # normal case for real activations with dead units.
    feats1 = gaussian(32, 4, seed=3)
    feats1[:, 0] = 1.0
    feats2 = gaussian(32, 4, mean=0.2, seed=4)
    feats2[:, 0] = 1.0
    score = compute_fid(*reference_stats(feats1), *reference_stats(feats2)).item()
    assert score >= 0.0
    assert score == pytest.approx(score)  # not NaN


def test_score_grows_with_separation():
    ref = reference_stats(gaussian(256, 8, seed=5))
    near = compute_fid(*reference_stats(gaussian(256, 8, mean=0.1, seed=6)), *ref).item()
    far = compute_fid(*reference_stats(gaussian(256, 8, mean=2.0, seed=6)), *ref).item()
    assert near < far


@pytest.mark.parametrize(
    ("mu1", "sigma1", "mu2", "sigma2"),
    [
        (torch.zeros(2, 2), torch.eye(2), torch.zeros(2), torch.eye(2)),  # mu not 1-D
        (torch.zeros(2), torch.eye(3), torch.zeros(2), torch.eye(2)),  # sigma1 wrong size
        (torch.zeros(2), torch.eye(2), torch.zeros(2), torch.eye(3)),  # sigma2 wrong size
        (torch.zeros(2), torch.eye(2), torch.zeros(3), torch.eye(2)),  # dims disagree
    ],
)
def test_bad_shapes_rejected(mu1, sigma1, mu2, sigma2):
    with pytest.raises(ValueError):
        compute_fid(mu1, sigma1, mu2, sigma2)


def test_streaming_matches_one_shot():
    feats = gaussian(100, 7, seed=7)
    stats = FeatureStats(7)
    for chunk in feats.split(13):
        stats.update(chunk)
    assert len(stats) == 100
    mu, cov = stats.mean_cov()
    want_mu, want_cov = reference_stats(feats)
    assert torch.allclose(mu, want_mu, atol=1e-10)
    assert torch.allclose(cov, want_cov, atol=1e-10)


def test_update_ignores_empty_batch():
    stats = FeatureStats(3)
    stats.update(gaussian(4, 3, seed=8))
    stats.update(torch.empty(0, 3))
    assert len(stats) == 4


def test_update_casts_dtype_and_rejects_bad_shapes():
    stats = FeatureStats(3)
    stats.update(gaussian(4, 3, seed=9).half())
    assert stats.sum.dtype == torch.float64
    with pytest.raises(ValueError):
        stats.update(torch.zeros(4))
    with pytest.raises(ValueError):
        stats.update(torch.zeros(4, 5))


def test_merge_equals_single_accumulator():
    feats = gaussian(60, 5, seed=10)
    left, right = feats[:25], feats[25:]

    combined = FeatureStats(5)
    combined.update(feats)

    a, b = FeatureStats(5), FeatureStats(5)
    a.update(left)
    b.update(right)
    a.merge(b)

    assert len(a) == len(combined)
    for got, want in zip(a.mean_cov(), combined.mean_cov(), strict=True):
        assert torch.allclose(got, want, atol=1e-10)


def test_merge_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        FeatureStats(4).merge(FeatureStats(5))


def test_reset_clears_state():
    stats = FeatureStats(3)
    stats.update(gaussian(8, 3, seed=11))
    stats.reset()
    assert len(stats) == 0
    assert not stats.sum.any()
    assert not stats.outer.any()


@pytest.mark.parametrize("n", [0, 1])
def test_mean_cov_needs_two_samples(n):
    stats = FeatureStats(3)
    if n:
        stats.update(gaussian(1, 3, seed=12))
    with pytest.raises(ValueError):
        stats.mean_cov()


def test_bad_dim_rejected():
    with pytest.raises(ValueError):
        FeatureStats(0)


def test_fid_from_stats_matches_compute_fid():
    fake_feats = gaussian(80, 6, mean=0.4, seed=13)
    real_feats = gaussian(80, 6, seed=14)

    fake, real = FeatureStats(6), FeatureStats(6)
    fake.update(fake_feats)
    real.update(real_feats)

    want = compute_fid(*reference_stats(fake_feats), *reference_stats(real_feats)).item()
    assert fid_from_stats(fake, real) == pytest.approx(want)


def test_fid_from_stats_rejects_dimension_mismatch():
    fake, real = FeatureStats(4), FeatureStats(5)
    fake.update(gaussian(4, 4, seed=15))
    real.update(gaussian(4, 5, seed=16))
    with pytest.raises(ValueError):
        fid_from_stats(fake, real)
