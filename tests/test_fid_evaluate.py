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
        # A buffer, not an attribute: a bare tensor does not follow the module
        # across `.to(device)`, which passes on a CPU-only machine and fails
        # everywhere else.
        self.register_buffer("weight", torch.randn(image_size * image_size, dim, generator=g))
        self.seen = 0

    def forward(self, images):
        self.seen += images.shape[0]
        return images.flatten(1) @ self.weight


@pytest.fixture
def make_checkpoint(tmp_path, monkeypatch, wake):
    """Write a real checkpoint over a tiny model, and stand in for MNIST."""

    def build(cfg=TINY):
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
    # The batch straddling the limit is truncated, so the count lands exactly.
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

    # 26 images in batches of 4 over 4 classes: restarting the cycle each batch
    # would give class 0 seven samples and classes 2 and 3 six each.
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
    # A continuous cycle can only ever be off by one.
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
    assert result.guidance is None  # unconditional checkpoint
    # Both sides went through the extractor.
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
    # Sanity check on the plumbing rather than the model: the same images down
    # both sides must score zero.
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
        used_ema=True,
    )
    assert result.undersampled is True
    text = result.format()
    assert "fid 12.500" in text
    assert "guidance 2" in text
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
        used_ema=False,
    )
    assert result.undersampled is False
    assert "warning" not in result.format()
    assert "guidance" not in result.format()
