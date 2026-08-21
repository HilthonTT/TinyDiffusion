import pytest
import torch

from tinydiffusion.metrics.features import FeatureBank
from tinydiffusion.metrics.fid import FeatureStats


def features(n, dim=6, seed=0):
    return torch.randn(n, dim, generator=torch.Generator().manual_seed(seed))


def test_a_bank_keeps_every_vector_in_order():
    bank = FeatureBank(6)
    first, second = features(4), features(3, seed=1)

    bank.update(first)
    bank.update(second)

    assert bank.n == len(bank) == 7
    assert torch.equal(bank.features, torch.cat([first, second]))


def test_the_moments_match_what_the_streaming_accumulator_would_have_made():
    # The whole reason a bank can stand in for FeatureStats: a run that retains
    # features still reports the same FID as one that only ever kept moments.
    # Not to the last bit — the bank sums over its own chunks rather than over
    # the batches the images arrived in — but to far beyond what a score prints.
    bank, stats = FeatureBank(6), FeatureStats(6)
    for start in range(0, 40, 7):
        batch = features(40)[start : start + 7]
        bank.update(batch)
        stats.update(batch)

    banked_mu, banked_cov = bank.stats.mean_cov()
    streamed_mu, streamed_cov = stats.mean_cov()

    assert torch.allclose(banked_mu, streamed_mu, rtol=0, atol=1e-12)
    assert torch.allclose(banked_cov, streamed_cov, rtol=0, atol=1e-12)


def test_the_moments_are_rebuilt_after_more_features_arrive():
    bank = FeatureBank(6)
    bank.update(features(8))
    first = bank.stats.n
    bank.update(features(8, seed=2))

    assert (first, bank.stats.n) == (8, 16)


def test_features_are_kept_as_float32_whatever_arrives():
    bank = FeatureBank(6)
    bank.update(features(4).double())
    assert bank.features.dtype is torch.float32


def test_an_empty_batch_changes_nothing():
    bank = FeatureBank(6)
    bank.update(features(4))
    bank.update(torch.zeros(0, 6))
    assert bank.n == 4


def test_an_empty_bank_still_has_a_shape():
    assert FeatureBank(6).features.shape == (0, 6)


def test_a_round_trip_through_a_state_dict_keeps_every_vector():
    bank = FeatureBank(6)
    bank.update(features(9))

    restored = FeatureBank.from_state_dict(bank.state_dict())

    assert restored.n == 9
    assert torch.equal(restored.features, bank.features)
    assert torch.equal(restored.stats.mean_cov()[0], bank.stats.mean_cov()[0])


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({"n": 2, "features": torch.zeros(2, 6)}, "missing"),
        ({"dim": 6, "n": -1, "features": torch.zeros(2, 6)}, "negative"),
        ({"dim": 6, "n": 2, "features": torch.zeros(3, 6)}, "must hold"),
        ({"dim": 6, "n": 2, "features": torch.zeros(2, 5)}, "must hold"),
    ],
)
def test_a_payload_that_is_not_ours_is_refused_rather_than_half_loaded(state, message):
    with pytest.raises(ValueError, match=message):
        FeatureBank.from_state_dict(state)


def test_a_bank_rejects_a_dimension_it_was_not_built_for():
    bank = FeatureBank(6)
    with pytest.raises(ValueError, match="6 features per row"):
        bank.update(features(4, dim=5))


def test_a_bank_rejects_features_that_are_not_a_matrix():
    with pytest.raises(ValueError, match=r"\(batch, dim\)"):
        FeatureBank(6).update(torch.zeros(6))


def test_a_bank_needs_a_positive_dimension():
    with pytest.raises(ValueError, match="must be positive"):
        FeatureBank(0)


def test_moving_a_bank_takes_its_vectors_with_it():
    bank = FeatureBank(6)
    bank.update(features(4))
    assert bank.to("cpu").features.device.type == "cpu"
    assert bank.device.type == "cpu"
