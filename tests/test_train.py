import contextlib
import dataclasses
import importlib
import json
import sys

import pytest
import torch
from PIL import Image

from tinydiffusion.diffusion.gaussian_diffusion import GaussianDiffusion, LossTerms
from tinydiffusion.sampling import load_for_sampling, sample_from_checkpoint
from tinydiffusion.training import artifacts as artifacts_module
from tinydiffusion.training import batches as batches_module
from tinydiffusion.training import checkpoints as ckpt_module
from tinydiffusion.training import loop as loop_module
from tinydiffusion.training import lr as lr_module
from tinydiffusion.training import model as model_module
from tinydiffusion.training import reporting as reporting_module
from tinydiffusion.training import setup as train_setup
from tinydiffusion.training import train as train_module
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.ema import EMA
from tinydiffusion.training.interrupt import InterruptChoice
from tinydiffusion.utils.tracking import METRICS_FILENAME


def _patch_loader(monkeypatch, factory):
    """Stand a fake dataloader in everywhere a run reaches for one.

    A run builds three: its training loader, and the two fixed reads in
    :mod:`~tinydiffusion.training.batches`. A fixture that replaced only the
    first would leave the held-out slice and the real strip coming off the
    actual dataset.
    """
    for module in (train_module, batches_module):
        monkeypatch.setattr(module, "image_dataloader", factory)


@pytest.fixture
def tiny_cfg(tmp_path) -> TrainConfig:
    """A config small enough to train end to end inside a test."""
    return TrainConfig(
        image_size=16,
        batch_size=4,
        num_workers=0,
        base_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=(),
        num_timesteps=10,
        num_epochs=2,
        ema_warmup=0,
        lr_warmup=0,
        amp=False,
        device="cpu",
        sample_every=0,
        num_samples=2,
        sample_steps=5,
        out_dir=tmp_path / "contents",
        ckpt_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "runs",
    )


@pytest.fixture
def fake_loader(monkeypatch):
    """Two batches of noise, standing in for the MNIST dataloader."""
    batches = [
        (torch.randn(4, 1, 16, 16), torch.arange(4, dtype=torch.long) % 10) for _ in range(2)
    ]
    _patch_loader(monkeypatch, lambda *a, **k: batches)


def _records(cfg) -> list[dict]:
    lines = (cfg.log_dir / METRICS_FILENAME).read_text().splitlines()
    return [json.loads(line) for line in lines if line]


def test_training_writes_one_metrics_record_per_epoch(tiny_cfg, fake_loader):
    train_module.train(tiny_cfg)
    assert [r["step"] for r in _records(tiny_cfg)] == [0, 1]


def test_the_logged_metrics_cover_loss_timesteps_and_throughput(tiny_cfg, fake_loader):
    train_module.train(tiny_cfg)
    record = _records(tiny_cfg)[0]

    assert record["train/loss"] > 0
    assert record["train/grad_norm"] > 0
    assert record["train/lr"] == pytest.approx(tiny_cfg.lr)
    assert record["time/epoch_seconds"] > 0
    assert record["time/images_per_second"] > 0
    quartiles = {key for key in record if key.startswith("train/loss_q")}
    assert quartiles
    assert quartiles <= {f"train/loss_q{i}" for i in range(4)}


def test_the_quartile_losses_average_to_the_epoch_loss(tiny_cfg, fake_loader):
    train_module.train(tiny_cfg)
    record = _records(tiny_cfg)[0]
    quartiles = [v for k, v in record.items() if k.startswith("train/loss_q")]
    assert min(quartiles) <= record["train/loss"] * 2
    assert max(quartiles) >= record["train/loss"] / 2


def test_logging_can_be_turned_off(tiny_cfg, fake_loader, capsys):
    cfg = dataclasses.replace(tiny_cfg, log_console=False, log_jsonl=False)
    train_module.train(cfg)
    assert not (cfg.log_dir / METRICS_FILENAME).exists()
    assert "train/loss" not in capsys.readouterr().out


def test_a_console_table_is_printed_each_epoch(tiny_cfg, fake_loader, capsys):
    train_module.train(tiny_cfg)
    out = capsys.readouterr().out
    assert "train/loss" in out
    headers = [line.split("|")[1].strip() for line in out.splitlines() if line.startswith("| step")]
    assert headers == [f"step {epoch}" for epoch in range(tiny_cfg.num_epochs)]


def test_a_conditional_run_trains_end_to_end(tiny_cfg, fake_loader):
    """The whole loop with a label embedding and a guided sample grid."""
    cfg = dataclasses.replace(
        tiny_cfg, num_classes=10, class_dropout=0.1, guidance=2.0, sample_every=1
    )
    diffusion = train_module.train(cfg)

    assert diffusion.net.num_classes == 10
    assert diffusion.net.label_embed.embed.num_embeddings == 11
    assert _records(cfg)[0]["train/loss"] > 0
    assert (cfg.out_dir / "sample_0002.png").exists()


def test_every_epoch_grid_redraws_the_same_latents(tiny_cfg, fake_loader, monkeypatch):
    """The grids are a flipbook of one latent set, not a fresh draw per epoch."""
    seen = []
    real_sample = artifacts_module.get_sampler("ddim")

    def spy(*args, **kwargs):
        seen.append(kwargs["noise"])
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(artifacts_module, "get_sampler", lambda name: spy)
    cfg = dataclasses.replace(tiny_cfg, sample_every=1)
    train_module.train(cfg)

    assert len(seen) == cfg.num_epochs
    assert all(torch.equal(seen[0], later) for later in seen[1:])
    assert seen[0].shape == (cfg.num_samples, 1, cfg.image_size, cfg.image_size)


def test_the_grid_latents_follow_the_seed(tiny_cfg, fake_loader, monkeypatch):
    """Derived from cfg.seed, so a --resume continues the same grid."""
    seen = []
    real_sample = artifacts_module.get_sampler("ddim")

    def spy(*args, **kwargs):
        seen.append(kwargs["noise"])
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(artifacts_module, "get_sampler", lambda name: spy)
    for seed in (0, 0, 1):
        train_module.train(dataclasses.replace(tiny_cfg, sample_every=1, num_epochs=1, seed=seed))

    assert torch.equal(seen[0], seen[1])
    assert not torch.equal(seen[0], seen[2])


@pytest.fixture
def shuffling_loader(monkeypatch):
    """A loader that reshuffles per epoch, the way the real one does.

    The training loader draws a fresh permutation from its generator at the
    start of every epoch, so *which* images the loop sees first is a function
    of the epoch index. A fixture handing back one fixed order would hide
    anything that depends on it.
    """
    images = torch.randn(8, 1, 16, 16)
    labels = torch.arange(8, dtype=torch.long) % 10

    class Loader:
        def __init__(self, batch_size, generator):
            self.batch_size = batch_size
            self.generator = generator

        def __iter__(self):
            order = (
                torch.arange(len(images))
                if self.generator is None
                else torch.randperm(len(images), generator=self.generator)
            )
            return iter(
                [
                    (images[chunk], labels[chunk])
                    for chunk in order.split(self.batch_size)
                    if len(chunk)
                ]
            )

        def __len__(self):
            return -(-len(images) // self.batch_size)

    _patch_loader(
        monkeypatch, lambda *a, batch_size=4, generator=None, **k: Loader(batch_size, generator)
    )
    return images, labels


@pytest.fixture
def grid_inputs(monkeypatch):
    """Record the real strip and labels each epoch's sample grid is built on."""
    seen = []
    real_save = artifacts_module.save_samples

    def spy(diffusion, ema, real, cfg, epoch, labels=None, noise=None):
        seen.append((real.clone(), None if labels is None else labels.clone()))
        return real_save(diffusion, ema, real, cfg, epoch, labels=labels, noise=noise)

    monkeypatch.setattr(loop_module, "save_samples", spy)
    return seen


def test_the_real_strip_is_the_front_of_the_unshuffled_split(
    tiny_cfg, shuffling_loader, grid_inputs
):
    images, labels = shuffling_loader
    cfg = dataclasses.replace(tiny_cfg, num_classes=10, sample_every=1)

    train_module.train(cfg)

    for real, shown in grid_inputs:
        assert torch.equal(real, images[: cfg.num_samples])
        assert torch.equal(shown, labels[: cfg.num_samples])


def test_the_real_strip_survives_a_resume(tiny_cfg, shuffling_loader, grid_inputs):
    """The grids stay a flipbook of one set of images across a --resume.

    Lifting the strip off the loop took it from whichever batch the shuffle put
    first, which is a function of the epoch — so a resumed run compared against
    different images than the epochs before it, and being conditional,
    generated on their labels too.
    """
    cfg = dataclasses.replace(tiny_cfg, num_classes=10, sample_every=1)
    train_module.train(dataclasses.replace(cfg, num_epochs=1))
    train_module.train(cfg, resume=cfg.ckpt_dir / ckpt_module.LAST_CHECKPOINT)

    (fresh, fresh_labels), (resumed, resumed_labels) = grid_inputs[0], grid_inputs[-1]
    assert torch.equal(fresh, resumed)
    assert torch.equal(fresh_labels, resumed_labels)


def test_an_unconditional_run_has_no_strip_labels(tiny_cfg, shuffling_loader, grid_inputs):
    train_module.train(dataclasses.replace(tiny_cfg, sample_every=1, num_epochs=1))
    assert grid_inputs[0][1] is None


def test_no_reference_is_read_when_no_grid_is_drawn(tiny_cfg):
    assert train_module.reference_batch(dataclasses.replace(tiny_cfg, sample_every=0)) == (
        None,
        None,
    )


def test_a_conditional_run_reports_its_classes(tiny_cfg, fake_loader, capsys):
    train_module.train(dataclasses.replace(tiny_cfg, num_classes=10))
    assert "10 classes, 0.1 label dropout" in capsys.readouterr().out


def test_an_unconditional_run_says_so(tiny_cfg, fake_loader, capsys):
    train_module.train(tiny_cfg)
    assert "unconditional" in capsys.readouterr().out


def test_a_conditional_checkpoint_reloads(tiny_cfg, fake_loader, tmp_path):
    cfg = dataclasses.replace(tiny_cfg, num_classes=10)
    train_module.train(cfg)

    diffusion, _, restored = load_for_sampling(cfg.ckpt_dir / "last.pt", "cpu")
    assert restored.num_classes == 10
    assert diffusion.net.num_classes == 10


def test_the_hybrid_objective_trains_end_to_end(tiny_cfg, fake_loader):
    """The whole loop — loss, EMA, sampling, checkpointing — on GaussianDiffusion."""
    cfg = dataclasses.replace(
        tiny_cfg, variance="learned_range", objective="rescaled_mse", sample_every=1
    )
    diffusion = train_module.train(cfg)

    assert isinstance(diffusion, GaussianDiffusion)
    assert [r["step"] for r in _records(cfg)] == [0, 1]
    assert _records(cfg)[0]["train/loss"] > 0
    assert (cfg.ckpt_dir / "last.pt").exists()
    assert (cfg.out_dir / "sample_0002.png").exists()


@pytest.mark.parametrize(
    ("step", "warmup", "expected"),
    [
        (0, 500, 0.0),
        (250, 500, 0.5),
        (500, 500, 1.0),
        (900, 500, 1.0),
        (0, 0, 1.0),
        (7, 0, 1.0),
    ],
)
def test_the_warmup_factor_ramps_then_holds(step, warmup, expected):
    assert lr_module._warmup_lr(step, warmup) == pytest.approx(expected)


def test_the_learning_rate_ramps_over_the_configured_steps(tiny_cfg, fake_loader):
    cfg = dataclasses.replace(tiny_cfg, lr_warmup=8)
    train_module.train(cfg)

    first, second = (r["train/lr"] for r in _records(cfg))
    assert first == pytest.approx(cfg.lr * 2 / 8)
    assert second == pytest.approx(cfg.lr * 4 / 8)


def test_the_warmup_survives_a_resume(tiny_cfg, fake_loader, tmp_path):
    cfg = dataclasses.replace(tiny_cfg, lr_warmup=8)
    train_module.train(cfg)

    diffusion = model_module.build_model(cfg)
    ema = EMA(diffusion.net, decay=cfg.ema_decay, warmup=cfg.ema_warmup)
    optim = torch.optim.Adam(diffusion.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: lr_module._warmup_lr(step, cfg.lr_warmup)
    )
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    ckpt = ckpt_module.read_checkpoint(cfg.ckpt_dir / "last.pt")
    ckpt_module.restore_checkpoint(
        ckpt, diffusion=diffusion, ema=ema, optim=optim, scaler=scaler, sched=sched
    )

    assert sched.get_last_lr()[0] == pytest.approx(cfg.lr * 4 / 8)


def test_a_checkpoint_without_a_schedule_still_restores(tiny_cfg, tmp_path):
    path = _checkpoint(tmp_path, tiny_cfg)
    ckpt = ckpt_module.read_checkpoint(path)
    assert ckpt["sched"] is None

    diffusion = model_module.build_model(tiny_cfg)
    ema = EMA(diffusion.net, decay=0.9, warmup=0)
    optim = torch.optim.Adam(diffusion.parameters(), lr=tiny_cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: lr_module._warmup_lr(step, 8)
    )
    ckpt_module.restore_checkpoint(ckpt, diffusion=diffusion, ema=ema, sched=sched)
    assert sched.get_last_lr()[0] == pytest.approx(0.0)


def test_a_resume_picks_the_random_stream_up_where_it_stopped(tiny_cfg, tmp_path):
    torch.manual_seed(1234)
    torch.randn(5)
    path = _checkpoint(tmp_path, tiny_cfg)
    expected = torch.randn(4)

    torch.manual_seed(9999)
    ckpt = ckpt_module.read_checkpoint(path)
    assert ckpt_module.restore_rng_state(ckpt)
    assert torch.equal(torch.randn(4), expected)


def test_a_checkpoint_without_rng_state_leaves_the_stream_alone(tiny_cfg, tmp_path):
    ckpt = ckpt_module.read_checkpoint(_checkpoint(tmp_path, tiny_cfg))
    del ckpt["rng"]

    torch.manual_seed(9999)
    expected = torch.randn(4)
    torch.manual_seed(9999)
    assert not ckpt_module.restore_rng_state(ckpt)
    assert torch.equal(torch.randn(4), expected)


def test_a_resumed_run_draws_the_noise_the_epoch_would_have_drawn(tiny_cfg, fake_loader, tmp_path):
    cfg = dataclasses.replace(tiny_cfg, num_epochs=2, ckpt_dir=tmp_path)
    train_module.train(cfg)
    straight = torch.randn(4)

    stopped = dataclasses.replace(cfg, num_epochs=1)
    train_module.train(stopped)
    train_module.train(cfg, resume=tmp_path / ckpt_module.LAST_CHECKPOINT)
    assert torch.equal(torch.randn(4), straight)


def _checkpoint(tmp_path, cfg, *, best_val=None):
    """Write a real checkpoint for `cfg` and return its path."""
    diffusion = model_module.build_model(cfg)
    ema = EMA(diffusion.net, decay=0.9, warmup=0)
    optim = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "source.pt"
    ckpt_module.save_checkpoint(
        path,
        epoch=0,
        diffusion=diffusion,
        ema=ema,
        optim=optim,
        scaler=scaler,
        cfg=cfg,
        best_val=best_val,
    )
    return path


def test_an_unchanged_config_resumes(tiny_cfg, tmp_path):
    ckpt = ckpt_module.read_checkpoint(_checkpoint(tmp_path, tiny_cfg))
    ckpt_module.check_resume_compatible(ckpt, tiny_cfg)


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        ("base_channels", {"base_channels": 16}),
        ("channel_mult", {"channel_mult": (1, 2)}),
        ("num_classes", {"num_classes": 10}),
        ("num_timesteps", {"num_timesteps": 20, "sample_steps": 5, "val_steps": 5}),
        ("schedule", {"schedule": "linear"}),
        ("variance", {"variance": "learned_range", "objective": "rescaled_mse"}),
        ("zero_snr", {"zero_snr": True, "predict": "v"}),
    ],
)
def test_a_changed_architecture_refuses_to_resume(tiny_cfg, tmp_path, field, overrides):
    ckpt = ckpt_module.read_checkpoint(_checkpoint(tmp_path, tiny_cfg))
    changed = dataclasses.replace(tiny_cfg, **overrides)

    with pytest.raises(ValueError, match=field):
        ckpt_module.check_resume_compatible(ckpt, changed)


def test_a_changed_batch_size_still_resumes(tiny_cfg, tmp_path):
    """Only the settings the weights depend on are checked."""
    ckpt = ckpt_module.read_checkpoint(_checkpoint(tmp_path, tiny_cfg))
    ckpt_module.check_resume_compatible(ckpt, dataclasses.replace(tiny_cfg, batch_size=64, lr=1e-3))


def test_a_checkpoint_without_provenance_is_left_to_load_state_dict(tiny_cfg):
    ckpt_module.check_resume_compatible({"epoch": 0}, tiny_cfg)


def test_the_refusal_names_the_file_and_both_values(tiny_cfg, tmp_path):
    path = _checkpoint(tmp_path, tiny_cfg)
    ckpt = ckpt_module.read_checkpoint(path)
    changed = dataclasses.replace(tiny_cfg, base_channels=16)

    with pytest.raises(ValueError) as excinfo:
        ckpt_module.check_resume_compatible(ckpt, changed, path=path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "checkpoint 8" in message
    assert "config 16" in message


def test_training_refuses_a_mismatched_resume(tiny_cfg, fake_loader, tmp_path):
    path = _checkpoint(tmp_path, tiny_cfg)
    with pytest.raises(ValueError, match="base_channels"):
        train_module.train(dataclasses.replace(tiny_cfg, base_channels=16), resume=path)


class _StubGuard:
    """Requests an interrupt once `after` batch boundaries have passed."""

    def __init__(self, after):
        self.remaining = after

    @property
    def requested(self):
        self.remaining -= 1
        return self.remaining < 0

    def resolve(self):
        return InterruptChoice(stop=True, save=True)

    def clear(self):
        pass


@pytest.fixture
def interrupt_after(monkeypatch):
    def install(batches):
        @contextlib.contextmanager
        def guard():
            yield _StubGuard(batches)

        monkeypatch.setattr(train_module, "interrupt_guard", guard)

    return install


def test_an_interrupt_saves_beside_last_rather_than_over_it(tiny_cfg, fake_loader, interrupt_after):
    interrupt_after(2)
    train_module.train(tiny_cfg)

    last = tiny_cfg.ckpt_dir / ckpt_module.LAST_CHECKPOINT
    interrupted = tiny_cfg.ckpt_dir / ckpt_module.INTERRUPTED_CHECKPOINT
    assert last.exists()
    assert interrupted.exists()

    complete = ckpt_module.read_checkpoint(last)
    partial = ckpt_module.read_checkpoint(interrupted)
    key = next(iter(complete["model"]))
    assert not torch.equal(complete["model"][key], partial["model"][key])


def test_an_interrupt_in_the_first_epoch_writes_no_last(tiny_cfg, fake_loader, interrupt_after):
    interrupt_after(1)
    train_module.train(tiny_cfg)

    assert not (tiny_cfg.ckpt_dir / ckpt_module.LAST_CHECKPOINT).exists()
    assert (tiny_cfg.ckpt_dir / ckpt_module.INTERRUPTED_CHECKPOINT).exists()


def test_the_interrupt_message_points_at_the_file_it_wrote(
    tiny_cfg, fake_loader, interrupt_after, capsys
):
    interrupt_after(1)
    train_module.train(tiny_cfg)
    expected = tiny_cfg.ckpt_dir / ckpt_module.INTERRUPTED_CHECKPOINT
    assert f"--resume {expected}" in capsys.readouterr().out


def test_validation_is_logged_and_a_best_checkpoint_is_kept(tiny_cfg, fake_loader):
    train_module.train(tiny_cfg)

    records = _records(tiny_cfg)
    assert all("val/loss" in record for record in records)
    assert all(record["val/best_loss"] <= record["val/loss"] for record in records)
    assert (tiny_cfg.ckpt_dir / ckpt_module.BEST_CHECKPOINT).exists()


def test_the_best_checkpoint_records_the_score_that_won_it(tiny_cfg, fake_loader):
    train_module.train(tiny_cfg)

    best = ckpt_module.read_checkpoint(tiny_cfg.ckpt_dir / ckpt_module.BEST_CHECKPOINT)
    scores = [record["val/loss"] for record in _records(tiny_cfg)]
    assert best["best_val"] == pytest.approx(min(scores))


def test_a_resume_keeps_comparing_against_the_earlier_best(tiny_cfg, fake_loader):
    train_module.train(tiny_cfg)
    before = ckpt_module.read_checkpoint(tiny_cfg.ckpt_dir / ckpt_module.BEST_CHECKPOINT)

    longer = dataclasses.replace(tiny_cfg, num_epochs=3)
    train_module.train(longer, resume=tiny_cfg.ckpt_dir / ckpt_module.LAST_CHECKPOINT)

    after = ckpt_module.read_checkpoint(longer.ckpt_dir / ckpt_module.BEST_CHECKPOINT)
    assert after["best_val"] <= before["best_val"]


def test_validation_can_be_switched_off(tiny_cfg, fake_loader):
    cfg = dataclasses.replace(tiny_cfg, val_every=0)
    train_module.train(cfg)

    assert not (cfg.ckpt_dir / ckpt_module.BEST_CHECKPOINT).exists()
    assert all("val/loss" not in record for record in _records(cfg))


def test_epoch_snapshots_are_kept_and_pruned(tiny_cfg, fake_loader):
    cfg = dataclasses.replace(tiny_cfg, keep_last=1, num_epochs=3)
    train_module.train(cfg)

    assert sorted(p.name for p in cfg.ckpt_dir.glob("epoch_*.pt")) == ["epoch_0003.pt"]


def test_no_snapshots_are_kept_by_default(tiny_cfg, fake_loader):
    train_module.train(tiny_cfg)
    assert list(tiny_cfg.ckpt_dir.glob("epoch_*.pt")) == []


def test_epoch_seed_depends_on_both_the_seed_and_the_epoch():
    assert train_module.epoch_seed(0, 0) == train_module.epoch_seed(0, 0)
    assert train_module.epoch_seed(0, 0) != train_module.epoch_seed(0, 1)
    assert train_module.epoch_seed(0, 1) != train_module.epoch_seed(1, 1)
    assert 0 <= train_module.epoch_seed(2**40, 5) < 2**63


@pytest.fixture
def shuffle_seeds(monkeypatch):
    """Record the seed the loader's RNG carries at the start of each epoch."""
    seeds: list[int] = []
    batches = [
        (torch.randn(4, 1, 16, 16), torch.arange(4, dtype=torch.long) % 10) for _ in range(2)
    ]

    class RecordingLoader:
        def __init__(self, generator):
            self.generator = generator

        def __iter__(self):
            seeds.append(self.generator.initial_seed())
            return iter(batches)

        def __len__(self):
            return len(batches)

    def loader(*a, generator=None, **k):
        return RecordingLoader(generator) if generator is not None else batches

    _patch_loader(monkeypatch, loader)
    return seeds


def test_each_epoch_shuffles_from_its_own_seed(tiny_cfg, shuffle_seeds):
    train_module.train(tiny_cfg)

    assert shuffle_seeds == [
        train_module.epoch_seed(tiny_cfg.seed, 0),
        train_module.epoch_seed(tiny_cfg.seed, 1),
    ]


def test_a_resumed_epoch_shuffles_as_it_would_have_unresumed(tiny_cfg, shuffle_seeds):
    train_module.train(dataclasses.replace(tiny_cfg, num_epochs=1))
    straight_through = list(shuffle_seeds)
    shuffle_seeds.clear()

    train_module.train(tiny_cfg, resume=tiny_cfg.ckpt_dir / ckpt_module.LAST_CHECKPOINT)

    assert straight_through == [train_module.epoch_seed(tiny_cfg.seed, 0)]
    assert shuffle_seeds == [train_module.epoch_seed(tiny_cfg.seed, 1)]


def test_deterministic_reaches_the_rng_and_leaves_the_autotuner_off(
    tiny_cfg, fake_loader, monkeypatch
):
    recorded = {}
    monkeypatch.setattr(
        train_module,
        "seed_everything",
        lambda seed, *, deterministic=False: recorded.update(
            seed=seed, deterministic=deterministic
        ),
    )
    cfg = dataclasses.replace(tiny_cfg, deterministic=True, num_epochs=1)

    train_module.train(cfg)

    assert recorded == {"seed": cfg.seed, "deterministic": True}
    assert torch.backends.cudnn.benchmark is False


def test_the_model_is_built_with_the_datasets_channel_count():
    rgb = TrainConfig(
        dataset="cifar10",
        image_size=16,
        base_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=(),
        num_timesteps=10,
        sample_steps=5,
        val_steps=5,
        device="cpu",
    )
    diffusion = model_module.build_model(rgb)

    out = diffusion.net(torch.randn(2, 3, 16, 16), torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 3, 16, 16)


def test_a_checkpoint_cannot_resume_into_a_different_dataset(tiny_cfg, fake_loader):
    train_module.train(dataclasses.replace(tiny_cfg, num_epochs=1))
    ckpt = ckpt_module.read_checkpoint(tiny_cfg.ckpt_dir / ckpt_module.LAST_CHECKPOINT)

    with pytest.raises(ValueError, match="dataset: checkpoint 'mnist', config 'cifar10'"):
        ckpt_module.check_resume_compatible(ckpt, dataclasses.replace(tiny_cfg, dataset="cifar10"))


def _has_colour(path) -> bool:
    """Whether a written grid holds anything other than grey pixels."""
    with Image.open(path) as image:
        pixels = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    channels = pixels.view(-1, 3)
    return bool((channels != channels[:, :1]).any())


def test_a_three_channel_run_trains_samples_and_reloads(tmp_path, monkeypatch):
    cfg = TrainConfig(
        dataset="cifar10",
        image_size=16,
        batch_size=4,
        num_workers=0,
        base_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=(),
        num_timesteps=10,
        num_epochs=1,
        ema_warmup=0,
        lr_warmup=0,
        amp=False,
        device="cpu",
        sample_every=1,
        num_samples=2,
        sample_steps=5,
        val_steps=5,
        val_every=0,
        num_classes=10,
        out_dir=tmp_path / "contents",
        ckpt_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "runs",
    )
    batches = [(torch.randn(4, 3, 16, 16), torch.arange(4, dtype=torch.long)) for _ in range(2)]
    _patch_loader(monkeypatch, lambda *a, **k: batches)

    train_module.train(cfg)

    grid = cfg.out_dir / "sample_0001.png"
    assert grid.is_file()
    assert _has_colour(grid)

    _, _, loaded = load_for_sampling(cfg.ckpt_dir / ckpt_module.LAST_CHECKPOINT, device="cpu")
    assert loaded.dataset == "cifar10"

    out = sample_from_checkpoint(
        cfg.ckpt_dir / ckpt_module.LAST_CHECKPOINT,
        tmp_path / "gen.png",
        num_images=2,
        num_steps=2,
        device="cpu",
    )
    assert _has_colour(out)


def test_the_warmup_ramp_is_unchanged_by_a_constant_schedule():
    factor = lambda step: lr_module.lr_factor(  # noqa: E731
        step, warmup=10, total=100, schedule="constant"
    )
    assert factor(0) == 0.0
    assert factor(5) == pytest.approx(0.5)
    assert factor(10) == 1.0
    assert factor(99) == 1.0


def test_cosine_decays_from_the_end_of_the_ramp_to_zero():
    factor = lambda step: lr_module.lr_factor(  # noqa: E731
        step, warmup=10, total=100, schedule="cosine"
    )
    assert factor(10) == pytest.approx(1.0)
    assert factor(55) == pytest.approx(0.5, abs=1e-6)
    assert factor(100) == pytest.approx(0.0, abs=1e-9)
    assert factor(5) == pytest.approx(0.5)


def test_a_run_shorter_than_its_warmup_still_has_a_schedule():
    assert lr_module.lr_factor(3, warmup=10, total=5, schedule="cosine") == pytest.approx(0.3)


def test_the_cosine_schedule_is_monotone_after_the_ramp():
    factors = [
        lr_module.lr_factor(step, warmup=4, total=40, schedule="cosine") for step in range(4, 41)
    ]
    assert factors == sorted(factors, reverse=True)


@pytest.fixture
def counted_loader(monkeypatch):
    """A loader of `n` batches, for counting optimiser steps against."""

    def build(n):
        batches = [
            (torch.randn(4, 1, 16, 16), torch.arange(4, dtype=torch.long) % 10) for _ in range(n)
        ]
        _patch_loader(monkeypatch, lambda *a, **k: batches)

    return build


def _ema_steps(cfg) -> int:
    ckpt = ckpt_module.read_checkpoint(cfg.ckpt_dir / ckpt_module.LAST_CHECKPOINT)
    return int(ckpt["ema_step"])


def test_accumulation_takes_one_optimiser_step_per_group(tiny_cfg, counted_loader):
    counted_loader(4)
    cfg = dataclasses.replace(tiny_cfg, num_epochs=1, grad_accum=2)

    train_module.train(cfg)

    assert _ema_steps(cfg) == 2


def test_a_ragged_accumulation_group_still_steps(tiny_cfg, counted_loader):
    counted_loader(3)
    cfg = dataclasses.replace(tiny_cfg, num_epochs=1, grad_accum=2)

    train_module.train(cfg)

    assert _ema_steps(cfg) == 2


def test_accumulation_logs_one_step_outcome_per_group(tiny_cfg, counted_loader):
    counted_loader(4)
    cfg = dataclasses.replace(tiny_cfg, num_epochs=1, grad_accum=2, val_every=0)

    train_module.train(cfg)

    record = _records(cfg)[0]
    assert record["train/skipped_step"] == 0.0
    assert record["train/grad_norm"] > 0


def test_without_accumulation_every_batch_steps(tiny_cfg, counted_loader):
    counted_loader(4)
    cfg = dataclasses.replace(tiny_cfg, num_epochs=1, grad_accum=1)

    train_module.train(cfg)

    assert _ema_steps(cfg) == 4


def test_the_optimiser_carries_the_configured_betas_and_decay(tiny_cfg, fake_loader):
    cfg = dataclasses.replace(
        tiny_cfg, num_epochs=1, betas=(0.85, 0.995), weight_decay=0.01, lr_warmup=0
    )

    train_module.train(cfg)

    group = ckpt_module.read_checkpoint(cfg.ckpt_dir / ckpt_module.LAST_CHECKPOINT)["optim"][
        "param_groups"
    ][0]
    assert group["betas"] == (0.85, 0.995)
    assert group["weight_decay"] == 0.01


def test_compiling_leaves_the_checkpoint_an_ordinary_one(tiny_cfg, fake_loader, monkeypatch):
    compiled = []

    def fake_compile(module, **kwargs):
        compiled.append(module)
        wrapper = torch.nn.Module()
        wrapper._orig_mod = module
        wrapper.forward = module.forward
        return wrapper

    monkeypatch.setattr(torch, "compile", fake_compile)
    cfg = dataclasses.replace(tiny_cfg, num_epochs=1, compile=True)

    train_module.train(cfg)

    assert compiled, "torch.compile was never called"
    keys = ckpt_module.read_checkpoint(cfg.ckpt_dir / ckpt_module.LAST_CHECKPOINT)["model"]
    assert not any(key.startswith("_orig_mod.") for key in keys)
    load_for_sampling(cfg.ckpt_dir / ckpt_module.LAST_CHECKPOINT, device="cpu")


class _LinearDiffusion(torch.nn.Module):
    """A stand-in whose loss is exactly the mean of the network's output.

    Deterministic, unlike the real objective, which draws a timestep and noise
    per image — so two runs over the same images are comparable tensor for
    tensor.
    """

    num_timesteps = 10

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Conv2d(1, 1, 1, bias=False)
        torch.nn.init.constant_(self.net.weight, 0.5)

    def loss_terms(self, x, model=None):
        out = (model if model is not None else self.net)(x)
        return LossTerms(
            loss=out.mean(),
            per_sample=out.detach().flatten(1).mean(dim=1),
            timesteps=torch.zeros(x.shape[0], dtype=torch.long),
        )


def _weight_after(cfg, batches, monkeypatch):
    monkeypatch.setattr(train_setup, "build_model", lambda cfg: _LinearDiffusion())
    _patch_loader(monkeypatch, lambda *a, **k: batches)
    return train_module.train(cfg).net.weight.detach().clone()


def test_accumulating_two_batches_updates_as_one_batch_of_both(tiny_cfg, tmp_path, monkeypatch):
    images = torch.randn(8, 1, 16, 16)
    labels = torch.zeros(8, dtype=torch.long)
    base = dataclasses.replace(
        tiny_cfg,
        num_epochs=1,
        num_classes=None,
        lr_warmup=0,
        grad_clip=0.0,
        ema_warmup=0,
        val_every=0,
        sample_every=0,
    )

    with monkeypatch.context() as m:
        split = dataclasses.replace(base, grad_accum=2, ckpt_dir=tmp_path / "split")
        accumulated = _weight_after(split, [(images[:4], labels[:4]), (images[4:], labels[4:])], m)

    with monkeypatch.context() as m:
        whole = dataclasses.replace(base, grad_accum=1, ckpt_dir=tmp_path / "whole")
        one_batch = _weight_after(whole, [(images, labels)], m)

    assert torch.allclose(accumulated, one_batch, atol=1e-7)


def test_a_ragged_group_is_averaged_over_what_it_holds(tiny_cfg, tmp_path, monkeypatch):
    images = torch.randn(4, 1, 16, 16)
    labels = torch.zeros(4, dtype=torch.long)
    base = dataclasses.replace(
        tiny_cfg,
        num_epochs=1,
        num_classes=None,
        lr_warmup=0,
        grad_clip=0.0,
        ema_warmup=0,
        val_every=0,
        sample_every=0,
    )

    with monkeypatch.context() as m:
        ragged = dataclasses.replace(base, grad_accum=3, ckpt_dir=tmp_path / "ragged")
        accumulated = _weight_after(ragged, [(images, labels)], m)

    with monkeypatch.context() as m:
        plain = dataclasses.replace(base, grad_accum=1, ckpt_dir=tmp_path / "plain")
        unaccumulated = _weight_after(plain, [(images, labels)], m)

    assert torch.allclose(accumulated, unaccumulated, atol=1e-7)


@pytest.mark.gpu
def test_compiling_is_skipped_when_triton_is_missing(tiny_cfg, fake_loader, monkeypatch, capsys):
    monkeypatch.setattr(train_setup.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(torch, "compile", lambda *a, **k: pytest.fail("should not compile"))
    cfg = dataclasses.replace(tiny_cfg, num_epochs=1, compile=True, device="cuda")

    train_module.train(cfg)

    assert "Triton is not installed" in capsys.readouterr().out


def test_compiling_goes_ahead_on_cpu_where_triton_is_not_needed():
    assert train_setup.can_compile("cpu")


def test_compiling_needs_triton_on_cuda(monkeypatch):
    monkeypatch.setattr(train_setup.importlib.util, "find_spec", lambda name: None)
    assert not train_setup.can_compile("cuda")
    monkeypatch.setattr(train_setup.importlib.util, "find_spec", lambda name: object())
    assert train_setup.can_compile("cuda")


def test_the_old_train_mnist_module_still_works_and_warns():
    """0.2 code imported the loop from train_mnist; that has to keep resolving."""
    sys.modules.pop("tinydiffusion.training.train_mnist", None)
    with pytest.warns(DeprecationWarning, match="tinydiffusion.training.train"):
        legacy = importlib.import_module("tinydiffusion.training.train_mnist")

    assert legacy.train_mnist is train_module.train
    assert legacy.build_model is model_module.build_model
    assert legacy.save_checkpoint is ckpt_module.save_checkpoint
    assert legacy.lr_factor is lr_module.lr_factor
    assert legacy.LAST_CHECKPOINT == ckpt_module.LAST_CHECKPOINT


def test_drain_replays_every_batch_in_order():
    from tinydiffusion.utils.tracking import null_logger

    pending = [{"train/loss": torch.tensor(float(i))} for i in (1.0, 2.0, 3.0)]
    with null_logger() as logger:
        loss_ema = reporting_module.drain_metrics(pending, logger, None)
        expected = None
        for value in (1.0, 2.0, 3.0):
            expected = value if expected is None else 0.9 * expected + 0.1 * value
        assert loss_ema == pytest.approx(expected)
        assert logger.means["train/loss"] == pytest.approx(2.0)
    assert pending == []


def test_drain_carries_the_smoothed_loss_across_calls():
    from tinydiffusion.utils.tracking import null_logger

    with null_logger() as logger:
        first = reporting_module.drain_metrics([{"train/loss": torch.tensor(1.0)}], logger, None)
        second = reporting_module.drain_metrics([{"train/loss": torch.tensor(2.0)}], logger, first)
    assert first == pytest.approx(1.0)
    assert second == pytest.approx(0.9 * 1.0 + 0.1 * 2.0)


def test_drain_of_nothing_leaves_the_smoothed_loss_alone():
    from tinydiffusion.utils.tracking import null_logger

    with null_logger() as logger:
        assert reporting_module.drain_metrics([], logger, 0.5) == 0.5
        assert reporting_module.drain_metrics([], logger, None) is None


def test_drain_keeps_tensor_and_float_metrics_on_their_own_keys():
    from tinydiffusion.utils.tracking import null_logger

    pending = [
        {
            "train/loss": torch.tensor(2.0),
            "train/skipped_step": 0.0,
            "train/grad_norm": torch.tensor(7.0),
        },
        {"train/loss": torch.tensor(4.0), "train/skipped_step": 1.0},
    ]
    with null_logger() as logger:
        reporting_module.drain_metrics(pending, logger, None)
        means = logger.means
    assert means["train/loss"] == pytest.approx(3.0)
    assert means["train/grad_norm"] == pytest.approx(7.0)
    assert means["train/skipped_step"] == pytest.approx(0.5)


def test_drain_handles_a_batch_of_floats_only():
    from tinydiffusion.utils.tracking import null_logger

    with null_logger() as logger:
        loss_ema = reporting_module.drain_metrics([{"train/loss": 3.0}], logger, None)
    assert loss_ema == pytest.approx(3.0)


@pytest.fixture
def many_batches(monkeypatch):
    """Enough batches that the buffer drains several times mid-epoch."""
    g = torch.Generator().manual_seed(0)
    batches = [
        (
            torch.randn(4, 1, 16, 16, generator=g),
            torch.arange(4, dtype=torch.long) % 10,
        )
        for _ in range(20)
    ]
    _patch_loader(monkeypatch, lambda *a, **k: batches)


DETERMINISTIC_KEYS = ("train/loss", "train/grad_norm", "train/lr", "train/skipped_step")


def test_buffering_does_not_change_a_single_logged_number(tmp_path, many_batches, monkeypatch):
    def run(log_dir, drain_every):
        monkeypatch.setattr(loop_module, "DRAIN_EVERY", drain_every)
        cfg = TrainConfig(
            image_size=16,
            batch_size=4,
            num_workers=0,
            base_channels=8,
            channel_mult=(1,),
            num_res_blocks=1,
            attn_resolutions=(),
            num_timesteps=10,
            num_epochs=1,
            ema_warmup=0,
            lr_warmup=0,
            amp=False,
            device="cpu",
            sample_every=0,
            num_samples=2,
            sample_steps=5,
            out_dir=tmp_path / f"c{drain_every}",
            ckpt_dir=tmp_path / f"k{drain_every}",
            log_dir=log_dir,
        )
        train_module.train(cfg)
        return _records(cfg)[0]

    every_batch = run(tmp_path / "eager", 1)
    buffered = run(tmp_path / "buffered", 8)

    for key in DETERMINISTIC_KEYS:
        assert buffered[key] == pytest.approx(every_batch[key], rel=0, abs=0), key
    quartiles = {k for k in every_batch if k.startswith("train/loss_q")}
    assert quartiles
    for key in quartiles:
        assert buffered[key] == pytest.approx(every_batch[key], rel=0, abs=0), key


def test_a_partial_final_run_still_reaches_the_log(tmp_path, many_batches, monkeypatch):
    monkeypatch.setattr(loop_module, "DRAIN_EVERY", 8)
    seen = []
    real_drain = loop_module.drain_metrics

    def counting(pending, logger, loss_ema):
        seen.append(len(pending))
        return real_drain(pending, logger, loss_ema)

    monkeypatch.setattr(loop_module, "drain_metrics", counting)
    cfg = TrainConfig(
        image_size=16,
        batch_size=4,
        num_workers=0,
        base_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=(),
        num_timesteps=10,
        num_epochs=1,
        ema_warmup=0,
        lr_warmup=0,
        amp=False,
        device="cpu",
        sample_every=0,
        num_samples=2,
        sample_steps=5,
        out_dir=tmp_path / "contents",
        ckpt_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "runs",
    )
    train_module.train(cfg)
    assert seen == [8, 8, 4]


@pytest.fixture
def teardown_calls(monkeypatch):
    """Record the order `train` waits at the barrier and leaves the group in."""
    calls: list[str] = []
    monkeypatch.setattr(train_module, "barrier", lambda group: calls.append("barrier"))
    monkeypatch.setattr(train_module, "distributed_shutdown", lambda: calls.append("shutdown"))
    return calls


def test_a_clean_run_waits_at_the_barrier_then_leaves_the_group(
    tiny_cfg, fake_loader, teardown_calls
):
    train_module.train(tiny_cfg)

    assert teardown_calls == ["barrier", "shutdown"]


def test_the_group_is_left_even_when_the_loop_raises(
    tiny_cfg, fake_loader, teardown_calls, monkeypatch
):
    """The failure that reads as a hang: an abandoned communicator.

    A rank that raises without leaving the group holds its GPU memory until the
    process is reaped, and every other rank sits in a collective this one will
    now never reach.
    """

    def explode(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(train_module, "run_epoch", explode)

    with pytest.raises(RuntimeError, match="out of memory"):
        train_module.train(tiny_cfg)

    assert teardown_calls == ["shutdown"]


def test_a_failing_rank_does_not_wait_at_the_barrier(
    tiny_cfg, fake_loader, teardown_calls, monkeypatch
):
    """Waiting for a rank that raised is the hang the teardown exists to avoid."""

    def explode(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(train_module, "run_epoch", explode)

    with pytest.raises(RuntimeError):
        train_module.train(tiny_cfg)

    assert "barrier" not in teardown_calls


def test_a_resume_with_fresh_moments_starts_at_the_schedule_lr(tiny_cfg, fake_loader, tmp_path):
    """Without the optimiser's own state the param groups hold the step-0 rate.

    Under any warmup that is zero, and the first optimiser step after the
    resume would be wasted. ``restore_run`` must put the schedule's current
    rate into the groups itself.
    """
    cfg = dataclasses.replace(tiny_cfg, lr_warmup=8)
    train_module.train(cfg)

    diffusion = model_module.build_model(cfg)
    ema = EMA(diffusion.net, decay=cfg.ema_decay, warmup=cfg.ema_warmup)
    optim = torch.optim.AdamW(diffusion.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: lr_module._warmup_lr(step, cfg.lr_warmup)
    )
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    assert optim.param_groups[0]["lr"] == 0.0

    ckpt = ckpt_module.read_checkpoint(cfg.ckpt_dir / "last.pt")
    train_setup.restore_run(
        ckpt,
        resume=cfg.ckpt_dir / "last.pt",
        diffusion=diffusion,
        ema=ema,
        optim=optim,
        scaler=scaler,
        sched=sched,
        full_fp16=True,  # disagrees with the checkpoint, so the moments are not loaded
        say=lambda _message: None,
    )

    assert optim.param_groups[0]["lr"] == pytest.approx(cfg.lr * 4 / 8)
    assert optim.param_groups[0]["lr"] == pytest.approx(sched.get_last_lr()[0])


def test_only_the_main_rank_replays_the_checkpoint_rng(tiny_cfg, tmp_path):
    """Copying rank 0's stream to every rank would make them all draw the same noise."""
    path = _checkpoint(tmp_path, tiny_cfg)
    ckpt = ckpt_module.read_checkpoint(path)
    assert ckpt.get("rng")

    def restore(restore_rng):
        diffusion = model_module.build_model(tiny_cfg)
        ema = EMA(diffusion.net, decay=0.9, warmup=0)
        optim = torch.optim.AdamW(diffusion.parameters(), lr=tiny_cfg.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda _step: 1.0)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        torch.manual_seed(1234)
        train_setup.restore_run(
            ckpt,
            resume=path,
            diffusion=diffusion,
            ema=ema,
            optim=optim,
            scaler=scaler,
            sched=sched,
            full_fp16=False,
            say=lambda _message: None,
            restore_rng=restore_rng,
        )
        return torch.rand(4)

    torch.manual_seed(1234)
    own_stream = torch.rand(4)
    assert not torch.equal(restore(True), own_stream)
    assert torch.equal(restore(False), own_stream)
