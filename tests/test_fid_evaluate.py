import dataclasses

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tinydiffusion.metrics import evaluate
from tinydiffusion.metrics.evaluate import FidResult, accumulate_features, fid_for_checkpoint
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

CONDITIONAL = dataclasses.replace(TINY, num_classes=10, guidance=2.0)

REAL_IMAGES = 12


class StubExtractor(nn.Module):
    """A cheap stand-in for Inception: a fixed random projection of the pixels."""

    def __init__(self, image_size=8, dim=6):
        super().__init__()
        self.dim = dim
        g = torch.Generator().manual_seed(0)
        self.register_buffer("weight", torch.randn(image_size * image_size, dim, generator=g))
        self.seen = 0

    def forward(self, images):
        self.seen += images.shape[0]
        return images.flatten(1) @ self.weight


@pytest.fixture
def make_checkpoint(tmp_path, monkeypatch, wake):
    """Write a real checkpoint over a tiny model, and stand in for MNIST."""

    def build(cfg=TINY):
        cfg = dataclasses.replace(cfg, data_root=tmp_path / "data")
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

    return build


@pytest.fixture
def checkpoint(make_checkpoint):
    return make_checkpoint()


@pytest.fixture
def extractor():
    return StubExtractor()


def batches(total, size, dim=8):
    while total > 0:
        n = min(size, total)
        yield torch.randn(n, 1, dim, dim).clamp(-1, 1)
        total -= n


def test_accumulate_features_counts_every_image(extractor):
    stats = accumulate_features(batches(10, 4), extractor)
    assert stats.n == 10
    assert stats.dim == extractor.dim


def test_accumulate_features_respects_the_limit(extractor):
    stats = accumulate_features(batches(100, 7), extractor)
    limited = accumulate_features(batches(100, 7), extractor, limit=10)
    assert stats.n == 100
    assert limited.n == 10


def test_accumulate_features_reuses_a_given_accumulator(extractor):
    stats = FeatureStats(extractor.dim)
    accumulate_features(batches(4, 4), extractor, stats=stats)
    accumulate_features(batches(4, 4), extractor, stats=stats)
    assert stats.n == 8


def test_generate_images_yields_exactly_what_was_asked(checkpoint, extractor):
    from tinydiffusion.sampling import load_for_sampling

    diffusion, ema, cfg = load_for_sampling(checkpoint, "cpu")
    produced = list(
        evaluate.generate_images(
            diffusion,
            ema.module,
            cfg,
            num_images=7,
            batch_size=3,
            num_steps=2,
            eta=0.0,
            guidance=1.0,
        )
    )
    assert [b.shape[0] for b in produced] == [3, 3, 1]
    assert all(b.shape[1:] == (1, cfg.image_size, cfg.image_size) for b in produced)


def test_the_generated_class_mix_stays_balanced_across_batches(make_checkpoint, monkeypatch):
    """The label cycle continues across batches instead of restarting."""
    from tinydiffusion.sampling import load_for_sampling

    path = make_checkpoint(CONDITIONAL)
    diffusion, ema, cfg = load_for_sampling(path, "cpu")

    seen = []
    monkeypatch.setattr(
        evaluate, "conditioned", lambda net, y, **kw: seen.append(y) or torch.nn.Identity()
    )
    monkeypatch.setattr(
        evaluate, "get_sampler", lambda name: lambda d, n, *a, **k: torch.zeros(n, 1, 8, 8)
    )

    list(
        evaluate.generate_images(
            diffusion,
            ema.module,
            cfg,
            num_images=26,
            batch_size=4,
            num_steps=2,
            eta=0.0,
            guidance=1.0,
        )
    )
    counts = torch.cat(seen).bincount(minlength=CONDITIONAL.num_classes)
    assert counts.max() - counts.min() <= 1


def test_an_unconditional_checkpoint_generates_without_labels(checkpoint, monkeypatch):
    from tinydiffusion.sampling import load_for_sampling

    diffusion, ema, cfg = load_for_sampling(checkpoint, "cpu")
    seen = []
    monkeypatch.setattr(
        evaluate, "conditioned", lambda net, y, **kw: seen.append(y) or torch.nn.Identity()
    )
    monkeypatch.setattr(
        evaluate, "get_sampler", lambda name: lambda d, n, *a, **k: torch.zeros(n, 1, 8, 8)
    )

    list(
        evaluate.generate_images(
            diffusion,
            ema.module,
            cfg,
            num_images=6,
            batch_size=4,
            num_steps=2,
            eta=0.0,
            guidance=1.0,
        )
    )
    assert seen == [None, None]


def test_fid_for_checkpoint_returns_a_result(checkpoint, extractor):
    result = fid_for_checkpoint(
        checkpoint, num_images=8, num_steps=2, extractor=extractor, progress=False
    )
    assert isinstance(result, FidResult)
    assert result.fid >= 0.0
    assert result.num_generated == 8
    assert result.num_real == 8
    assert result.feature_dim == extractor.dim
    assert result.num_steps == 2
    assert result.used_ema is True
    assert result.guidance is None
    assert extractor.seen == 16


def test_fid_is_reproducible_for_a_seed(checkpoint, extractor):
    kwargs = {"num_images": 6, "num_steps": 2, "extractor": extractor, "progress": False}
    first = fid_for_checkpoint(checkpoint, **kwargs)
    second = fid_for_checkpoint(checkpoint, **kwargs)
    assert first.fid == pytest.approx(second.fid)


def test_seed_changes_the_samples(checkpoint, extractor):
    kwargs = {"num_images": 6, "num_steps": 2, "extractor": extractor, "progress": False}
    first = fid_for_checkpoint(checkpoint, seed=0, **kwargs)
    second = fid_for_checkpoint(checkpoint, seed=1, **kwargs)
    assert first.fid != second.fid


def test_scoring_against_itself_is_near_zero(extractor):
    images = [torch.randn(8, 1, 8, 8).clamp(-1, 1) for _ in range(3)]
    a = accumulate_features(iter(images), extractor)
    b = accumulate_features(iter(images), extractor)
    from tinydiffusion.metrics import fid_from_stats

    assert fid_from_stats(a, b) == pytest.approx(0.0, abs=1e-8)


def test_conditional_checkpoint_reports_its_guidance(make_checkpoint, extractor):
    path = make_checkpoint(CONDITIONAL)
    result = fid_for_checkpoint(
        path, num_images=4, num_steps=2, guidance=3.0, extractor=extractor, progress=False
    )
    assert result.guidance == 3.0


def test_raw_weights_can_be_scored(checkpoint, extractor):
    result = fid_for_checkpoint(
        checkpoint, num_images=4, num_steps=2, use_ema=False, extractor=extractor, progress=False
    )
    assert result.used_ema is False


def test_short_split_is_scored_against_what_exists(checkpoint, extractor, capsys):
    result = fid_for_checkpoint(
        checkpoint, num_images=REAL_IMAGES + 20, num_steps=2, extractor=extractor, progress=False
    )
    assert result.num_real == REAL_IMAGES
    assert result.num_generated == REAL_IMAGES + 20
    assert f"only {REAL_IMAGES} images" in capsys.readouterr().out


def test_too_few_images_is_rejected(checkpoint, extractor):
    with pytest.raises(ValueError, match="at least 2"):
        fid_for_checkpoint(checkpoint, num_images=1, extractor=extractor, progress=False)


def test_unknown_split_is_rejected(checkpoint, extractor):
    with pytest.raises(ValueError, match="unknown split"):
        fid_for_checkpoint(checkpoint, split="valid", extractor=extractor, progress=False)


def test_result_format_flags_undersampling():
    result = FidResult(
        checkpoint="last.pt",
        split="train",
        fid=12.5,
        num_generated=100,
        num_real=100,
        feature_dim=2048,
        num_steps=50,
        guidance=2.0,
        guidance_rescale=0.7,
        used_ema=True,
    )
    assert result.undersampled is True
    text = result.format()
    assert "fid 12.500" in text
    assert "guidance 2" in text
    assert "rescale 0.7" in text
    assert "warning" in text


def test_result_format_stays_quiet_when_well_sampled():
    result = FidResult(
        checkpoint="last.pt",
        split="train",
        fid=3.0,
        num_generated=10_000,
        num_real=10_000,
        feature_dim=2048,
        num_steps=50,
        guidance=None,
        guidance_rescale=0.0,
        used_ema=False,
    )
    assert result.undersampled is False
    assert "warning" not in result.format()
    assert "guidance" not in result.format()
    assert "rescale" not in result.format()
    assert "fp32" not in result.format()


def test_result_format_names_a_non_default_precision():
    result = FidResult(
        checkpoint="last.pt",
        split="train",
        fid=3.0,
        num_generated=10_000,
        num_real=10_000,
        feature_dim=2048,
        num_steps=50,
        guidance=None,
        guidance_rescale=0.0,
        used_ema=False,
        sample_precision="fp16",
    )
    assert "fp16" in result.format()


def test_the_precision_the_samples_were_drawn_at_is_recorded(checkpoint, extractor):
    result = fid_for_checkpoint(
        checkpoint,
        num_images=4,
        num_steps=2,
        extractor=extractor,
        progress=False,
        device="cpu",
        sample_precision="fp16",
    )
    assert result.sample_precision == "fp32"


def test_the_opt_in_metrics_are_absent_unless_asked_for(checkpoint, extractor):
    result = fid_for_checkpoint(
        checkpoint, num_images=8, num_steps=2, extractor=extractor, progress=False
    )
    assert result.kid is None
    assert result.precision_recall is None
    assert "kid" not in result.format()


def test_kid_is_reported_with_the_spread_it_was_averaged_over(checkpoint, extractor):
    result = fid_for_checkpoint(
        checkpoint,
        num_images=8,
        num_steps=2,
        extractor=extractor,
        progress=False,
        kid=True,
        kid_subsets=3,
        kid_subset_size=4,
    )
    assert result.kid is not None
    assert (result.kid.subsets, result.kid.subset_size) == (3, 4)
    assert "kid" in result.format()


def test_precision_and_recall_are_reported_as_fractions(checkpoint, extractor):
    result = fid_for_checkpoint(
        checkpoint,
        num_images=8,
        num_steps=2,
        extractor=extractor,
        progress=False,
        precision_recall=True,
        neighbours=2,
    )
    assert result.precision_recall is not None
    assert 0.0 <= result.precision_recall.precision <= 1.0
    assert 0.0 <= result.precision_recall.recall <= 1.0
    assert result.precision_recall.neighbours == 2
    assert "precision" in result.format()


def test_retaining_features_does_not_move_the_fid(checkpoint):
    plain = fid_for_checkpoint(
        checkpoint, num_images=8, num_steps=2, extractor=StubExtractor(), progress=False
    )
    retained = fid_for_checkpoint(
        checkpoint,
        num_images=8,
        num_steps=2,
        extractor=StubExtractor(),
        progress=False,
        cache=False,
        kid=True,
        kid_subsets=2,
        kid_subset_size=4,
    )
    assert plain.fid == pytest.approx(retained.fid, rel=1e-9)


def test_the_kid_score_does_not_depend_on_how_many_batches_sampling_drew(checkpoint):
    scores = [
        fid_for_checkpoint(
            checkpoint,
            num_images=8,
            num_steps=2,
            batch_size=size,
            extractor=StubExtractor(),
            progress=False,
            cache=False,
            kid=True,
            kid_subsets=3,
            kid_subset_size=4,
        ).kid.mean
        for size in (8, 8)
    ]
    assert scores[0] == scores[1]


def test_an_undersampled_score_points_at_the_metric_that_is_not(checkpoint, extractor):
    result = fid_for_checkpoint(
        checkpoint, num_images=4, num_steps=2, extractor=extractor, progress=False
    )
    assert result.undersampled
    assert "--kid is unbiased" in result.format()


class HeadedStubExtractor(StubExtractor):
    """StubExtractor plus the other two heads, so sFID and the IS have something to read.

    Deliberately not Inception: the point of these tests is the wiring — that
    one pass feeds three accumulators, that the spatial half is cached under
    its own key, and that the numbers land on the result — none of which is a
    claim about Inception's features.
    """

    SPATIAL_DIM = 5
    CLASSES = 4

    def __init__(self, image_size=8, dim=6):
        super().__init__(image_size=image_size, dim=dim)
        g = torch.Generator().manual_seed(1)
        self.register_buffer(
            "spatial_weight", torch.randn(image_size * image_size, self.SPATIAL_DIM, generator=g)
        )
        self.register_buffer(
            "class_weight", torch.randn(image_size * image_size, self.CLASSES, generator=g)
        )

    def analyse(self, images):
        from tinydiffusion.metrics.inception import InceptionOutputs

        flat = images.flatten(1)
        return InceptionOutputs(
            pool=self(images),
            spatial=flat @ self.spatial_weight,
            probs=(flat @ self.class_weight).softmax(dim=-1),
        )


@pytest.fixture
def headed_extractor(monkeypatch):
    """A stub with all three heads, and SFID_DIM narrowed to match it.

    The spatial accumulator is built at Inception's width, which a stub cannot
    produce without doing Inception's work; pointing the constant at the stub's
    own width is what keeps the test about the wiring.
    """
    monkeypatch.setattr(evaluate, "SFID_DIM", HeadedStubExtractor.SPATIAL_DIM)
    monkeypatch.setattr(evaluate, "INCEPTION_CLASSES", HeadedStubExtractor.CLASSES)
    return HeadedStubExtractor()


def test_the_spatial_and_classifier_metrics_are_absent_unless_asked_for(checkpoint, extractor):
    result = fid_for_checkpoint(
        checkpoint, num_images=8, num_steps=2, extractor=extractor, progress=False
    )
    assert result.sfid is None
    assert result.inception_score is None
    lines = result.format().splitlines()
    assert not any(line.startswith(("sfid", "inception score")) for line in lines)


def test_sfid_is_reported_beside_the_fid(checkpoint, headed_extractor):
    result = fid_for_checkpoint(
        checkpoint,
        num_images=8,
        num_steps=2,
        extractor=headed_extractor,
        progress=False,
        sfid=True,
    )
    assert result.sfid is not None
    assert result.sfid >= 0.0
    assert any(line.startswith("sfid") for line in result.format().splitlines())


def test_the_inception_score_is_reported_with_its_spread(checkpoint, headed_extractor):
    result = fid_for_checkpoint(
        checkpoint,
        num_images=8,
        num_steps=2,
        extractor=headed_extractor,
        progress=False,
        inception_score=True,
        is_splits=2,
    )
    assert result.inception_score is not None
    assert result.inception_score.splits == 2
    assert 1.0 <= result.inception_score.mean <= HeadedStubExtractor.CLASSES + 1e-6
    assert any(line.startswith("inception score") for line in result.format().splitlines())


def test_the_extra_heads_ride_along_on_one_pass(checkpoint, headed_extractor):
    """Two extra metrics must not cost two extra passes over every image.

    The stub counts images through its pooled head, which every reading shares,
    so asking for all three has to leave that count where a plain FID left it.
    """
    plain = fid_for_checkpoint(
        checkpoint, num_images=8, num_steps=2, extractor=headed_extractor, progress=False
    )
    seen_plain = headed_extractor.seen

    headed_extractor.seen = 0
    everything = fid_for_checkpoint(
        checkpoint,
        num_images=8,
        num_steps=2,
        extractor=headed_extractor,
        progress=False,
        sfid=True,
        inception_score=True,
        is_splits=2,
        cache=False,
    )
    assert headed_extractor.seen == seen_plain
    assert everything.fid == pytest.approx(plain.fid)


def test_the_spatial_reference_half_is_cached_under_its_own_key(checkpoint, headed_extractor):
    """A second sFID must not repeat the real pass, and must not collide with the first."""
    from tinydiffusion.metrics.cache import CACHE_DIRNAME

    kwargs = {
        "num_images": 8,
        "num_steps": 2,
        "extractor": headed_extractor,
        "progress": False,
        "sfid": True,
    }
    first = fid_for_checkpoint(checkpoint, **kwargs)

    cache_dir = next(p for p in checkpoint.parent.rglob(CACHE_DIRNAME) if p.is_dir())
    entries = {p.name for p in cache_dir.iterdir()}
    assert any(name.endswith("_spatial.pt") for name in entries)
    assert any(not name.endswith("_spatial.pt") for name in entries)

    headed_extractor.seen = 0
    second = fid_for_checkpoint(checkpoint, **kwargs)
    assert second.sfid == pytest.approx(first.sfid)


def test_an_extractor_without_the_extra_heads_says_so(checkpoint, extractor):
    """The stand-ins that keep FID testable are exactly this case."""
    with pytest.raises(ValueError, match="does not expose"):
        fid_for_checkpoint(
            checkpoint,
            num_images=8,
            num_steps=2,
            extractor=extractor,
            progress=False,
            sfid=True,
        )
