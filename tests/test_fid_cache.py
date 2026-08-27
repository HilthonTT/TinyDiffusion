import dataclasses

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tinydiffusion.metrics import evaluate
from tinydiffusion.metrics.cache import (
    CACHE_DIRNAME,
    extractor_id,
    load_reference_features,
    load_reference_stats,
    reference_features_path,
    reference_stats_path,
    save_reference_features,
    save_reference_stats,
    spatial_stats_path,
)
from tinydiffusion.metrics.evaluate import fid_for_checkpoint
from tinydiffusion.metrics.features import FeatureBank
from tinydiffusion.metrics.fid import FeatureStats
from tinydiffusion.training.checkpoints import save_checkpoint
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.ema import EMA
from tinydiffusion.training.model import build_model

TINY = TrainConfig(
    image_size=8,
    base_channels=4,
    channel_mult=(1,),
    num_res_blocks=1,
    attn_resolutions=(),
    num_timesteps=20,
    sample_steps=4,
    num_samples=2,
    batch_size=4,
    num_workers=0,
    device="cpu",
)

REAL_IMAGES = 12


class StubExtractor(nn.Module):
    """A cheap stand-in for Inception, counting the images it is shown."""

    def __init__(self, dim=6, image_size=8):
        super().__init__()
        self.dim = dim
        g = torch.Generator().manual_seed(0)
        self.register_buffer("weight", torch.randn(image_size * image_size, dim, generator=g))
        self.seen = 0

    def forward(self, images):
        self.seen += images.shape[0]
        return images.flatten(1) @ self.weight


class OtherExtractor(StubExtractor):
    """A different feature network of the same width."""


def stats_over(n, dim=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    stats = FeatureStats(dim)
    stats.update(torch.randn(n, dim, generator=g))
    return stats


# --- FeatureStats serialisation --------------------------------------------


def test_state_dict_round_trips_exactly():
    original = stats_over(20)
    restored = FeatureStats.from_state_dict(original.state_dict())
    assert restored.n == original.n
    assert restored.dim == original.dim
    mu, cov = original.mean_cov()
    mu2, cov2 = restored.mean_cov()
    assert torch.equal(mu, mu2)
    assert torch.equal(cov, cov2)


def test_a_restored_accumulator_can_still_be_extended():
    # The raw moments are what is stored, so a restored set is not a dead end.
    straight = stats_over(10, seed=1)
    straight.update(torch.ones(4, 6))
    restored = FeatureStats.from_state_dict(stats_over(10, seed=1).state_dict())
    restored.update(torch.ones(4, 6))
    assert restored.n == straight.n
    assert torch.equal(restored.mean_cov()[0], straight.mean_cov()[0])


def test_a_state_dict_missing_a_key_is_rejected():
    state = stats_over(4).state_dict()
    del state["outer"]
    with pytest.raises(ValueError, match="missing"):
        FeatureStats.from_state_dict(state)


def test_a_state_dict_with_the_wrong_shapes_is_rejected():
    state = stats_over(4).state_dict()
    state["sum"] = torch.zeros(3)
    with pytest.raises(ValueError, match="must hold"):
        FeatureStats.from_state_dict(state)


# --- the cache key ----------------------------------------------------------


def test_the_path_lives_under_the_dataset_root(tmp_path):
    path = reference_stats_path(
        tmp_path,
        dataset="mnist",
        split="train",
        num_images=100,
        image_size=32,
        extractor=StubExtractor(),
    )
    assert path.parent == tmp_path / CACHE_DIRNAME


@pytest.mark.parametrize(
    "changed",
    [
        {"dataset": "cifar10"},
        {"split": "test"},
        {"num_images": 200},
        {"image_size": 64},
        {"extractor": OtherExtractor()},
    ],
)
def test_every_input_that_moves_the_statistics_moves_the_key(tmp_path, changed):
    base = {
        "dataset": "mnist",
        "split": "train",
        "num_images": 100,
        "image_size": 32,
        "extractor": StubExtractor(),
    }
    assert reference_stats_path(tmp_path, **base) != reference_stats_path(
        tmp_path, **{**base, **changed}
    )


def test_the_extractor_id_separates_classes_of_the_same_width():
    assert extractor_id(StubExtractor()) != extractor_id(OtherExtractor())


# --- reading and writing ----------------------------------------------------


def test_a_saved_entry_reads_back(tmp_path):
    path = tmp_path / CACHE_DIRNAME / "entry.pt"
    save_reference_stats(path, stats_over(20))
    loaded = load_reference_stats(path, dim=6)
    assert loaded is not None
    assert loaded.n == 20


def test_a_missing_entry_is_simply_absent(tmp_path):
    assert load_reference_stats(tmp_path / "nothing.pt", dim=6) is None


def test_a_corrupt_entry_is_treated_as_absent(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a checkpoint at all")
    assert load_reference_stats(path, dim=6) is None


def test_a_truncated_entry_is_treated_as_absent(tmp_path):
    path = tmp_path / "half.pt"
    save_reference_stats(path, stats_over(20))
    payload = path.read_bytes()
    path.write_bytes(payload[: len(payload) // 2])
    assert load_reference_stats(path, dim=6) is None


def test_an_entry_of_another_width_is_refused(tmp_path):
    path = tmp_path / "wide.pt"
    save_reference_stats(path, stats_over(20, dim=8))
    assert load_reference_stats(path, dim=6) is None


def test_an_entry_too_small_for_a_covariance_is_refused(tmp_path):
    path = tmp_path / "tiny.pt"
    save_reference_stats(path, stats_over(1))
    assert load_reference_stats(path, dim=6) is None


def test_an_entry_from_another_format_is_refused(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"format": 0, **stats_over(20).state_dict()}, path)
    assert load_reference_stats(path, dim=6) is None


def test_saving_leaves_no_temporary_files_behind(tmp_path):
    path = tmp_path / CACHE_DIRNAME / "entry.pt"
    save_reference_stats(path, stats_over(20))
    assert [p.name for p in path.parent.iterdir()] == ["entry.pt"]


def test_an_unwritable_destination_is_not_fatal(tmp_path, monkeypatch):
    # The score has already been computed by this point; failing to memoise it
    # is not a reason to throw it away.
    def refuse(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(torch, "save", refuse)
    save_reference_stats(tmp_path / "entry.pt", stats_over(20))
    assert not (tmp_path / "entry.pt").exists()


# --- the cache inside a scoring run ----------------------------------------


@pytest.fixture
def checkpoint(tmp_path, monkeypatch, wake):
    cfg = dataclasses.replace(TINY, data_root=tmp_path / "data")
    diffusion = build_model(cfg)
    wake(diffusion.net)
    ema = EMA(diffusion.net, decay=0.9, warmup=0)
    optim = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path, epoch=0, diffusion=diffusion, ema=ema, optim=optim, scaler=scaler, cfg=cfg
    )

    def fake_loader(*args, **kwargs):
        g = torch.Generator().manual_seed(1)
        images = torch.randn(REAL_IMAGES, 1, cfg.image_size, cfg.image_size, generator=g).clamp(
            -1, 1
        )
        y = torch.zeros(REAL_IMAGES, dtype=torch.long)
        return DataLoader(TensorDataset(images, y), batch_size=kwargs.get("batch_size", 4))

    monkeypatch.setattr(evaluate, "image_dataloader", fake_loader)
    return path


def score(checkpoint, extractor, **kwargs):
    return fid_for_checkpoint(
        checkpoint, num_images=8, num_steps=2, extractor=extractor, progress=False, **kwargs
    )


def test_the_second_score_does_not_re_read_the_real_images(checkpoint):
    extractor = StubExtractor()
    first = score(checkpoint, extractor)
    after_first = extractor.seen
    second = score(checkpoint, extractor)
    # 8 real + 8 generated the first time; only the 8 generated the second.
    assert after_first == 16
    assert extractor.seen - after_first == 8
    assert first.fid == pytest.approx(second.fid)


def test_the_cache_leaves_the_score_unchanged(checkpoint):
    cached = score(checkpoint, StubExtractor())
    uncached = score(checkpoint, StubExtractor(), cache=False)
    assert cached.fid == pytest.approx(uncached.fid)
    assert cached.num_real == uncached.num_real


def test_disabling_the_cache_recomputes_every_time(checkpoint):
    extractor = StubExtractor()
    score(checkpoint, extractor, cache=False)
    after_first = extractor.seen
    score(checkpoint, extractor, cache=False)
    assert extractor.seen - after_first == 16


def test_the_entry_lands_under_the_dataset_root(checkpoint, tmp_path):
    score(checkpoint, StubExtractor())
    cached = list((tmp_path / "data" / CACHE_DIRNAME).glob("*.pt"))
    assert len(cached) == 1
    assert cached[0].name == "mnist_train_8_8px_stubextractor6.pt"


def test_a_different_image_count_does_not_reuse_the_entry(checkpoint):
    extractor = StubExtractor()
    fid_for_checkpoint(checkpoint, num_images=8, num_steps=2, extractor=extractor, progress=False)
    after_first = extractor.seen
    fid_for_checkpoint(checkpoint, num_images=6, num_steps=2, extractor=extractor, progress=False)
    # A different reference set, so its features are computed rather than read.
    assert extractor.seen - after_first == 12


def test_a_stale_entry_for_another_extractor_is_not_reused(checkpoint):
    score(checkpoint, StubExtractor())
    other = OtherExtractor()
    score(checkpoint, other)
    assert other.seen == 16


def test_a_run_wanting_features_writes_a_second_entry(checkpoint, tmp_path):
    score(checkpoint, StubExtractor(), kid=True, kid_subsets=2, kid_subset_size=4)
    cached = sorted(p.name for p in (tmp_path / "data" / CACHE_DIRNAME).glob("*.pt"))
    assert cached == [
        "mnist_train_8_8px_stubextractor6.pt",
        "mnist_train_8_8px_stubextractor6_features.pt",
    ]


def test_a_fid_only_run_writes_no_feature_entry(checkpoint, tmp_path):
    score(checkpoint, StubExtractor())
    entries = list((tmp_path / "data" / CACHE_DIRNAME).glob("*_features.pt"))
    assert entries == []


def test_the_second_scored_run_reuses_the_cached_features(checkpoint):
    extractor = StubExtractor()
    kwargs = {"kid": True, "kid_subsets": 2, "kid_subset_size": 4}
    first = score(checkpoint, extractor, **kwargs)
    after_first = extractor.seen
    second = score(checkpoint, extractor, **kwargs)

    # 8 real + 8 generated the first time; only the 8 generated the second.
    assert after_first == 16
    assert extractor.seen - after_first == 8
    assert first.fid == pytest.approx(second.fid)
    assert first.kid.mean == pytest.approx(second.kid.mean)


def test_a_moments_entry_cannot_stand_in_for_a_feature_one(checkpoint):
    # The moments hold no vectors, so KID has to re-read the real images even
    # though a FID-shaped entry for this exact set is already on disk.
    extractor = StubExtractor()
    score(checkpoint, extractor)
    after_fid_only = extractor.seen
    score(checkpoint, extractor, kid=True, kid_subsets=2, kid_subset_size=4)
    assert extractor.seen - after_fid_only == 16


def test_a_feature_entry_of_another_width_is_refused(tmp_path):
    bank = FeatureBank(4)
    bank.update(torch.randn(5, 4))
    path = tmp_path / "features.pt"
    save_reference_features(path, bank)
    assert load_reference_features(path, dim=6) is None
    assert load_reference_features(path, dim=4) is not None


def test_a_corrupt_feature_entry_is_treated_as_absent(tmp_path):
    path = tmp_path / "features.pt"
    path.write_bytes(b"not a checkpoint")
    assert load_reference_features(path, dim=4) is None


def test_a_feature_entry_reads_back_every_vector(tmp_path):
    bank = FeatureBank(4)
    bank.update(torch.randn(7, 4))
    path = tmp_path / "features.pt"
    save_reference_features(path, bank)

    restored = load_reference_features(path, dim=4)
    assert restored is not None
    assert torch.equal(restored.features, bank.features)


def test_the_two_kinds_of_entry_key_the_same_way(tmp_path):
    key = {
        "dataset": "mnist",
        "split": "train",
        "num_images": 8,
        "image_size": 8,
        "extractor": StubExtractor(),
    }
    stats = reference_stats_path(tmp_path, **key)
    features = reference_features_path(tmp_path, **key)
    spatial = spatial_stats_path(tmp_path, **key)
    assert features.parent == stats.parent == spatial.parent
    assert features.name == f"{stats.stem}_features.pt"
    assert spatial.name == f"{stats.stem}_spatial.pt"
    # Three distinct files: the moments, the vectors, and the spatial moments
    # sFID is taken in. One overwriting another would silently score a set of
    # images in the wrong feature space.
    assert len({stats, features, spatial}) == 3
