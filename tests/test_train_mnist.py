import contextlib
import dataclasses
import json

import pytest
import torch
from PIL import Image

from tinydiffusion.diffusion.gaussian_diffusion import GaussianDiffusion
from tinydiffusion.sampling import load_for_sampling, sample_from_checkpoint
from tinydiffusion.training import train_mnist as train_module
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.ema import EMA
from tinydiffusion.training.interrupt import InterruptChoice
from tinydiffusion.utils.tracking import METRICS_FILENAME


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
        # Four optimiser steps in total, so a ramp would leave the LR near zero
        # and the weights barely moved. Warmup gets its own test below.
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
    monkeypatch.setattr(train_module, "image_dataloader", lambda *a, **k: batches)


def _records(cfg) -> list[dict]:
    lines = (cfg.log_dir / METRICS_FILENAME).read_text().splitlines()
    return [json.loads(line) for line in lines if line]


def test_training_writes_one_metrics_record_per_epoch(tiny_cfg, fake_loader):
    train_module.train_mnist(tiny_cfg)
    assert [r["step"] for r in _records(tiny_cfg)] == [0, 1]


def test_the_logged_metrics_cover_loss_timesteps_and_throughput(tiny_cfg, fake_loader):
    train_module.train_mnist(tiny_cfg)
    record = _records(tiny_cfg)[0]

    assert record["train/loss"] > 0
    assert record["train/grad_norm"] > 0
    assert record["train/lr"] == pytest.approx(tiny_cfg.lr)
    assert record["time/epoch_seconds"] > 0
    assert record["time/images_per_second"] > 0
    # Eight images over ten timesteps will not hit every quartile, but the
    # ones that do fire must be named after the schedule's quartiles.
    quartiles = {key for key in record if key.startswith("train/loss_q")}
    assert quartiles
    assert quartiles <= {f"train/loss_q{i}" for i in range(4)}


def test_the_quartile_losses_average_to_the_epoch_loss(tiny_cfg, fake_loader):
    # Not exactly equal — the quartile means are per batch — but a quartile
    # that had drifted off the loss entirely would show up here.
    train_module.train_mnist(tiny_cfg)
    record = _records(tiny_cfg)[0]
    quartiles = [v for k, v in record.items() if k.startswith("train/loss_q")]
    assert min(quartiles) <= record["train/loss"] * 2
    assert max(quartiles) >= record["train/loss"] / 2


def test_logging_can_be_turned_off(tiny_cfg, fake_loader, capsys):
    cfg = dataclasses.replace(tiny_cfg, log_console=False, log_jsonl=False)
    train_module.train_mnist(cfg)
    assert not (cfg.log_dir / METRICS_FILENAME).exists()
    assert "train/loss" not in capsys.readouterr().out


def test_a_console_table_is_printed_each_epoch(tiny_cfg, fake_loader, capsys):
    train_module.train_mnist(tiny_cfg)
    out = capsys.readouterr().out
    assert "train/loss" in out
    headers = [line.split("|")[1].strip() for line in out.splitlines() if line.startswith("| step")]
    assert headers == [f"step {epoch}" for epoch in range(tiny_cfg.num_epochs)]


def test_a_conditional_run_trains_end_to_end(tiny_cfg, fake_loader):
    """The whole loop with a label embedding and a guided sample grid."""
    cfg = dataclasses.replace(
        tiny_cfg, num_classes=10, class_dropout=0.1, guidance=2.0, sample_every=1
    )
    diffusion = train_module.train_mnist(cfg)

    assert diffusion.net.num_classes == 10
    # The reserved null row is what guidance extrapolates from, so the table
    # has to be one wider than the class count.
    assert diffusion.net.label_embed.embed.num_embeddings == 11
    assert _records(cfg)[0]["train/loss"] > 0
    assert (cfg.out_dir / "sample_0002.png").exists()


def test_every_epoch_grid_redraws_the_same_latents(tiny_cfg, fake_loader, monkeypatch):
    """The grids are a flipbook of one latent set, not a fresh draw per epoch."""
    seen = []
    real_sample = train_module.ddim_sample

    def spy(*args, **kwargs):
        seen.append(kwargs["noise"])
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(train_module, "ddim_sample", spy)
    cfg = dataclasses.replace(tiny_cfg, sample_every=1)
    train_module.train_mnist(cfg)

    assert len(seen) == cfg.num_epochs
    assert all(torch.equal(seen[0], later) for later in seen[1:])
    assert seen[0].shape == (cfg.num_samples, 1, cfg.image_size, cfg.image_size)


def test_the_grid_latents_follow_the_seed(tiny_cfg, fake_loader, monkeypatch):
    """Derived from cfg.seed, so a --resume continues the same grid."""
    seen = []
    real_sample = train_module.ddim_sample

    def spy(*args, **kwargs):
        seen.append(kwargs["noise"])
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(train_module, "ddim_sample", spy)
    for seed in (0, 0, 1):
        train_module.train_mnist(
            dataclasses.replace(tiny_cfg, sample_every=1, num_epochs=1, seed=seed)
        )

    assert torch.equal(seen[0], seen[1])
    assert not torch.equal(seen[0], seen[2])


def test_a_conditional_run_reports_its_classes(tiny_cfg, fake_loader, capsys):
    train_module.train_mnist(dataclasses.replace(tiny_cfg, num_classes=10))
    assert "10 classes, 0.1 label dropout" in capsys.readouterr().out


def test_an_unconditional_run_says_so(tiny_cfg, fake_loader, capsys):
    train_module.train_mnist(tiny_cfg)
    assert "unconditional" in capsys.readouterr().out


def test_a_conditional_checkpoint_reloads(tiny_cfg, fake_loader, tmp_path):
    # The label embedding is part of the state dict, so a conditional run has
    # to round-trip through save/load or --resume breaks.
    cfg = dataclasses.replace(tiny_cfg, num_classes=10)
    train_module.train_mnist(cfg)

    diffusion, _, restored = load_for_sampling(cfg.ckpt_dir / "last.pt", "cpu")
    assert restored.num_classes == 10
    assert diffusion.net.num_classes == 10


def test_the_hybrid_objective_trains_end_to_end(tiny_cfg, fake_loader):
    """The whole loop — loss, EMA, sampling, checkpointing — on GaussianDiffusion."""
    cfg = dataclasses.replace(
        tiny_cfg, variance="learned_range", objective="rescaled_mse", sample_every=1
    )
    diffusion = train_module.train_mnist(cfg)

    assert isinstance(diffusion, GaussianDiffusion)
    assert [r["step"] for r in _records(cfg)] == [0, 1]
    assert _records(cfg)[0]["train/loss"] > 0
    assert (cfg.ckpt_dir / "last.pt").exists()
    assert (cfg.out_dir / "sample_0002.png").exists()


# --- learning rate warmup -------------------------------------------------


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
    assert train_module._warmup_lr(step, warmup) == pytest.approx(expected)


def test_the_learning_rate_ramps_over_the_configured_steps(tiny_cfg, fake_loader):
    # Two batches an epoch over two epochs, so the run ends mid-ramp and the
    # logged rate must still be climbing.
    cfg = dataclasses.replace(tiny_cfg, lr_warmup=8)
    train_module.train_mnist(cfg)

    first, second = (r["train/lr"] for r in _records(cfg))
    assert first == pytest.approx(cfg.lr * 2 / 8)
    assert second == pytest.approx(cfg.lr * 4 / 8)


def test_the_warmup_survives_a_resume(tiny_cfg, fake_loader, tmp_path):
    cfg = dataclasses.replace(tiny_cfg, lr_warmup=8)
    train_module.train_mnist(cfg)

    diffusion = train_module.build_model(cfg)
    ema = EMA(diffusion.net, decay=cfg.ema_decay, warmup=cfg.ema_warmup)
    optim = torch.optim.Adam(diffusion.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: train_module._warmup_lr(step, cfg.lr_warmup)
    )
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    ckpt = train_module.read_checkpoint(cfg.ckpt_dir / "last.pt")
    train_module.restore_checkpoint(
        ckpt, diffusion=diffusion, ema=ema, optim=optim, scaler=scaler, sched=sched
    )

    # Four steps done, so the resumed run picks the ramp up where it stopped
    # rather than replaying it from zero over already-trained weights.
    assert sched.get_last_lr()[0] == pytest.approx(cfg.lr * 4 / 8)


def test_a_checkpoint_without_a_schedule_still_restores(tiny_cfg, tmp_path):
    path = _checkpoint(tmp_path, tiny_cfg)
    ckpt = train_module.read_checkpoint(path)
    assert ckpt["sched"] is None

    diffusion = train_module.build_model(tiny_cfg)
    ema = EMA(diffusion.net, decay=0.9, warmup=0)
    optim = torch.optim.Adam(diffusion.parameters(), lr=tiny_cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: train_module._warmup_lr(step, 8)
    )
    train_module.restore_checkpoint(ckpt, diffusion=diffusion, ema=ema, sched=sched)
    assert sched.get_last_lr()[0] == pytest.approx(0.0)


# --- resume compatibility -------------------------------------------------


def _checkpoint(tmp_path, cfg, *, best_val=None):
    """Write a real checkpoint for `cfg` and return its path."""
    diffusion = train_module.build_model(cfg)
    ema = EMA(diffusion.net, decay=0.9, warmup=0)
    optim = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "source.pt"
    train_module.save_checkpoint(
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
    ckpt = train_module.read_checkpoint(_checkpoint(tmp_path, tiny_cfg))
    train_module.check_resume_compatible(ckpt, tiny_cfg)


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        ("base_channels", {"base_channels": 16}),
        ("channel_mult", {"channel_mult": (1, 2)}),
        ("num_classes", {"num_classes": 10}),
        ("num_timesteps", {"num_timesteps": 20, "sample_steps": 5, "val_steps": 5}),
        ("schedule", {"schedule": "linear"}),
        # A learned variance needs an objective that trains it, so the pair
        # has to change together for the config to build at all.
        ("variance", {"variance": "learned_range", "objective": "rescaled_mse"}),
    ],
)
def test_a_changed_architecture_refuses_to_resume(tiny_cfg, tmp_path, field, overrides):
    ckpt = train_module.read_checkpoint(_checkpoint(tmp_path, tiny_cfg))
    changed = dataclasses.replace(tiny_cfg, **overrides)

    with pytest.raises(ValueError, match=field):
        train_module.check_resume_compatible(ckpt, changed)


def test_a_changed_batch_size_still_resumes(tiny_cfg, tmp_path):
    """Only the settings the weights depend on are checked."""
    ckpt = train_module.read_checkpoint(_checkpoint(tmp_path, tiny_cfg))
    train_module.check_resume_compatible(
        ckpt, dataclasses.replace(tiny_cfg, batch_size=64, lr=1e-3)
    )


def test_a_checkpoint_without_provenance_is_left_to_load_state_dict(tiny_cfg):
    train_module.check_resume_compatible({"epoch": 0}, tiny_cfg)


def test_the_refusal_names_the_file_and_both_values(tiny_cfg, tmp_path):
    path = _checkpoint(tmp_path, tiny_cfg)
    ckpt = train_module.read_checkpoint(path)
    changed = dataclasses.replace(tiny_cfg, base_channels=16)

    with pytest.raises(ValueError) as excinfo:
        train_module.check_resume_compatible(ckpt, changed, path=path)

    message = str(excinfo.value)
    assert str(path) in message
    assert "checkpoint 8" in message
    assert "config 16" in message


def test_training_refuses_a_mismatched_resume(tiny_cfg, fake_loader, tmp_path):
    path = _checkpoint(tmp_path, tiny_cfg)
    with pytest.raises(ValueError, match="base_channels"):
        train_module.train_mnist(dataclasses.replace(tiny_cfg, base_channels=16), resume=path)


# --- interrupt safety -----------------------------------------------------


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


@pytest.fixture
def interrupt_after(monkeypatch):
    def install(batches):
        @contextlib.contextmanager
        def guard():
            yield _StubGuard(batches)

        monkeypatch.setattr(train_module, "interrupt_guard", guard)

    return install


def test_an_interrupt_saves_beside_last_rather_than_over_it(tiny_cfg, fake_loader, interrupt_after):
    # Two batches per epoch, so this lands on the first batch of epoch 2 —
    # after epoch 1 has already written last.pt.
    interrupt_after(2)
    train_module.train_mnist(tiny_cfg)

    last = tiny_cfg.ckpt_dir / train_module.LAST_CHECKPOINT
    interrupted = tiny_cfg.ckpt_dir / train_module.INTERRUPTED_CHECKPOINT
    assert last.exists()
    assert interrupted.exists()

    complete = train_module.read_checkpoint(last)
    partial = train_module.read_checkpoint(interrupted)
    # The interrupt landed a full optimiser step past the completed epoch, so
    # if last.pt still holds that epoch's own weights the two must differ.
    key = next(iter(complete["model"]))
    assert not torch.equal(complete["model"][key], partial["model"][key])


def test_an_interrupt_in_the_first_epoch_writes_no_last(tiny_cfg, fake_loader, interrupt_after):
    interrupt_after(1)
    train_module.train_mnist(tiny_cfg)

    assert not (tiny_cfg.ckpt_dir / train_module.LAST_CHECKPOINT).exists()
    assert (tiny_cfg.ckpt_dir / train_module.INTERRUPTED_CHECKPOINT).exists()


def test_the_interrupt_message_points_at_the_file_it_wrote(
    tiny_cfg, fake_loader, interrupt_after, capsys
):
    interrupt_after(1)
    train_module.train_mnist(tiny_cfg)
    expected = tiny_cfg.ckpt_dir / train_module.INTERRUPTED_CHECKPOINT
    assert f"--resume {expected}" in capsys.readouterr().out


# --- validation and checkpoint retention ----------------------------------


def test_validation_is_logged_and_a_best_checkpoint_is_kept(tiny_cfg, fake_loader):
    train_module.train_mnist(tiny_cfg)

    records = _records(tiny_cfg)
    assert all("val/loss" in record for record in records)
    assert all(record["val/best_loss"] <= record["val/loss"] for record in records)
    assert (tiny_cfg.ckpt_dir / train_module.BEST_CHECKPOINT).exists()


def test_the_best_checkpoint_records_the_score_that_won_it(tiny_cfg, fake_loader):
    train_module.train_mnist(tiny_cfg)

    best = train_module.read_checkpoint(tiny_cfg.ckpt_dir / train_module.BEST_CHECKPOINT)
    scores = [record["val/loss"] for record in _records(tiny_cfg)]
    assert best["best_val"] == pytest.approx(min(scores))


def test_a_resume_keeps_comparing_against_the_earlier_best(tiny_cfg, fake_loader):
    train_module.train_mnist(tiny_cfg)
    before = train_module.read_checkpoint(tiny_cfg.ckpt_dir / train_module.BEST_CHECKPOINT)

    longer = dataclasses.replace(tiny_cfg, num_epochs=3)
    train_module.train_mnist(longer, resume=tiny_cfg.ckpt_dir / train_module.LAST_CHECKPOINT)

    after = train_module.read_checkpoint(longer.ckpt_dir / train_module.BEST_CHECKPOINT)
    assert after["best_val"] <= before["best_val"]


def test_validation_can_be_switched_off(tiny_cfg, fake_loader):
    cfg = dataclasses.replace(tiny_cfg, val_every=0)
    train_module.train_mnist(cfg)

    assert not (cfg.ckpt_dir / train_module.BEST_CHECKPOINT).exists()
    assert all("val/loss" not in record for record in _records(cfg))


def test_epoch_snapshots_are_kept_and_pruned(tiny_cfg, fake_loader):
    cfg = dataclasses.replace(tiny_cfg, keep_last=1, num_epochs=3)
    train_module.train_mnist(cfg)

    assert sorted(p.name for p in cfg.ckpt_dir.glob("epoch_*.pt")) == ["epoch_0003.pt"]


def test_no_snapshots_are_kept_by_default(tiny_cfg, fake_loader):
    train_module.train_mnist(tiny_cfg)
    assert list(tiny_cfg.ckpt_dir.glob("epoch_*.pt")) == []


def test_epoch_seed_depends_on_both_the_seed_and_the_epoch():
    assert train_module.epoch_seed(0, 0) == train_module.epoch_seed(0, 0)
    assert train_module.epoch_seed(0, 0) != train_module.epoch_seed(0, 1)
    assert train_module.epoch_seed(0, 1) != train_module.epoch_seed(1, 1)
    # manual_seed rejects anything wider than 64 bits.
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
        # The held-out slice is read once, unshuffled, and passes no generator;
        # only the training loader is of interest here.
        return RecordingLoader(generator) if generator is not None else batches

    monkeypatch.setattr(train_module, "image_dataloader", loader)
    return seeds


def test_each_epoch_shuffles_from_its_own_seed(tiny_cfg, shuffle_seeds):
    train_module.train_mnist(tiny_cfg)

    assert shuffle_seeds == [
        train_module.epoch_seed(tiny_cfg.seed, 0),
        train_module.epoch_seed(tiny_cfg.seed, 1),
    ]


def test_a_resumed_epoch_shuffles_as_it_would_have_unresumed(tiny_cfg, shuffle_seeds):
    # Seeding the loader once at startup made a resumed epoch 1 replay epoch 0's
    # ordering, so a run split across two processes saw different data than the
    # same run trained straight through.
    train_module.train_mnist(dataclasses.replace(tiny_cfg, num_epochs=1))
    straight_through = list(shuffle_seeds)
    shuffle_seeds.clear()

    train_module.train_mnist(tiny_cfg, resume=tiny_cfg.ckpt_dir / train_module.LAST_CHECKPOINT)

    assert straight_through == [train_module.epoch_seed(tiny_cfg.seed, 0)]
    assert shuffle_seeds == [train_module.epoch_seed(tiny_cfg.seed, 1)]


def test_deterministic_reaches_the_rng_and_leaves_the_autotuner_off(
    tiny_cfg, fake_loader, monkeypatch
):
    # seed_everything clears cudnn.benchmark; training used to set it straight
    # back, which quietly undid the setting that had just been applied.
    recorded = {}
    monkeypatch.setattr(
        train_module,
        "seed_everything",
        lambda seed, *, deterministic=False: recorded.update(
            seed=seed, deterministic=deterministic
        ),
    )
    cfg = dataclasses.replace(tiny_cfg, deterministic=True, num_epochs=1)

    train_module.train_mnist(cfg)

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
    diffusion = train_module.build_model(rgb)

    out = diffusion.net(torch.randn(2, 3, 16, 16), torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 3, 16, 16)


def test_a_checkpoint_cannot_resume_into_a_different_dataset(tiny_cfg, fake_loader):
    # The channel count is the U-Net's input and output width, so the weights
    # do not fit — and a bare load_state_dict would say so only as a wall of
    # size mismatches.
    train_module.train_mnist(dataclasses.replace(tiny_cfg, num_epochs=1))
    ckpt = train_module.read_checkpoint(tiny_cfg.ckpt_dir / train_module.LAST_CHECKPOINT)

    with pytest.raises(ValueError, match="dataset: checkpoint 'mnist', config 'cifar10'"):
        train_module.check_resume_compatible(ckpt, dataclasses.replace(tiny_cfg, dataset="cifar10"))


def _has_colour(path) -> bool:
    """Whether a written grid holds anything other than grey pixels."""
    with Image.open(path) as image:
        pixels = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    channels = pixels.view(-1, 3)
    return bool((channels != channels[:, :1]).any())


def test_a_three_channel_run_trains_samples_and_reloads(tmp_path, monkeypatch):
    # The whole point of the registry: nothing between the config and the PNG
    # should know how many channels a dataset has.
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
    monkeypatch.setattr(train_module, "image_dataloader", lambda *a, **k: batches)

    train_module.train_mnist(cfg)

    # save_image writes RGB whatever it is handed, so the mode proves nothing:
    # what distinguishes a colour run is that the channels actually differ.
    grid = cfg.out_dir / "sample_0001.png"
    assert grid.is_file()
    assert _has_colour(grid)

    _, _, loaded = load_for_sampling(cfg.ckpt_dir / train_module.LAST_CHECKPOINT, device="cpu")
    assert loaded.dataset == "cifar10"

    out = sample_from_checkpoint(
        cfg.ckpt_dir / train_module.LAST_CHECKPOINT,
        tmp_path / "gen.png",
        num_images=2,
        num_steps=2,
        device="cpu",
    )
    assert _has_colour(out)
