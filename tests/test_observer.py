import dataclasses

import pytest
import torch

from tinydiffusion.training import train as train_module
from tinydiffusion.training.checkpoints import INTERRUPTED_CHECKPOINT
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.observer import BatchProgress, TrainObserver, TrainPlan


class Recorder:
    """A watcher that keeps everything it is told, and can ask for a stop.

    Args:
        stop_after: how many batch reports to accept before asking the run to
            stop, or None to let it run to the end.
    """

    def __init__(self, stop_after: int | None = None):
        self.plan: TrainPlan | None = None
        self.messages: list[str] = []
        self.batches: list[BatchProgress] = []
        self.epochs: list[tuple[int, dict]] = []
        self.samples: list = []
        self._stop_after = stop_after

    def on_plan(self, plan):
        self.plan = plan

    def on_message(self, text):
        self.messages.append(text)

    def on_batch(self, progress):
        self.batches.append(progress)

    def on_epoch(self, step, metrics):
        self.epochs.append((step, dict(metrics)))

    def on_sample(self, path):
        self.samples.append(path)

    def stop_requested(self):
        return self._stop_after is not None and len(self.batches) >= self._stop_after


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
        val_every=0,
        out_dir=tmp_path / "contents",
        ckpt_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        log_console=False,
    )


@pytest.fixture
def fake_loader(monkeypatch):
    """Enough batches of noise that a drain lands inside an epoch."""
    batches = [
        (torch.randn(4, 1, 16, 16), torch.arange(4, dtype=torch.long) % 10) for _ in range(20)
    ]
    monkeypatch.setattr(train_module, "image_dataloader", lambda *a, **k: batches)


def test_a_recorder_satisfies_the_protocol():
    # runtime_checkable, so this is a shape check rather than a subclass one --
    # which is the point of the protocol: a watcher owes six methods, not a
    # base class.
    assert isinstance(Recorder(), TrainObserver)


def test_the_plan_describes_the_run(tiny_cfg, fake_loader):
    observer = Recorder()
    train_module.train(tiny_cfg, observer=observer)

    plan = observer.plan
    assert plan is not None
    assert plan.dataset == tiny_cfg.dataset
    assert plan.image_size == tiny_cfg.image_size
    assert plan.num_epochs == 2
    assert plan.start_epoch == 0
    assert plan.batch_size == tiny_cfg.batch_size
    assert plan.steps_per_epoch == 20
    assert plan.parameters > 0
    assert plan.device == "cpu"
    # val_every is off in this config, so there is nothing held out.
    assert plan.validation_images == 0


def test_the_plan_line_is_delivered_rather_than_printed(tiny_cfg, fake_loader, capsys):
    observer = Recorder()
    train_module.train(tiny_cfg, observer=observer)

    # stdout belongs to whatever is watching; nothing may be written behind it.
    assert capsys.readouterr().out == ""
    assert any("parameters" in message for message in observer.messages)


def test_without_an_observer_the_run_still_prints(tiny_cfg, fake_loader, capsys):
    train_module.train(tiny_cfg)
    assert "parameters" in capsys.readouterr().out


def test_batch_progress_advances_through_the_epoch(tiny_cfg, fake_loader):
    observer = Recorder()
    train_module.train(tiny_cfg, observer=observer)

    assert observer.batches
    first = observer.batches[0]
    assert first.num_batches == 20
    assert first.num_epochs == 2
    assert 0.0 < first.epoch_fraction <= 1.0
    # Reported once per DRAIN_EVERY batches, not once per batch.
    assert len(observer.batches) < 20 * 2
    # Within an epoch the batch index only ever grows.
    per_epoch: dict[int, list[int]] = {}
    for progress in observer.batches:
        per_epoch.setdefault(progress.epoch, []).append(progress.batch)
    for indices in per_epoch.values():
        assert indices == sorted(indices)


def test_the_loss_reaches_the_watcher(tiny_cfg, fake_loader):
    observer = Recorder()
    train_module.train(tiny_cfg, observer=observer)
    assert any(progress.loss is not None for progress in observer.batches)


def test_epoch_metrics_arrive_through_the_backend_fan_out(tiny_cfg, fake_loader):
    observer = Recorder()
    train_module.train(tiny_cfg, observer=observer)

    assert [step for step, _ in observer.epochs] == [0, 1]
    metrics = observer.epochs[0][1]
    assert "train/loss" in metrics
    assert "train/lr" in metrics


def test_the_watcher_gets_each_sample_grid_as_it_is_written(tiny_cfg, fake_loader):
    cfg = dataclasses.replace(tiny_cfg, sample_every=1)
    observer = Recorder()
    train_module.train(cfg, observer=observer)

    assert len(observer.samples) == 2
    for path in observer.samples:
        assert path.is_file()


def test_save_samples_hands_back_the_path_it_wrote(tiny_cfg, fake_loader):
    # The return value is what lets a watcher pick the grid up without
    # reconstructing the filename from the epoch index.
    cfg = dataclasses.replace(tiny_cfg, sample_every=1)
    observer = Recorder()
    train_module.train(cfg, observer=observer)
    assert observer.samples[0].name == "sample_0001.png"


def test_a_watcher_can_stop_the_run(tiny_cfg, fake_loader):
    # Stopping mid-epoch, and without the Ctrl+C prompt -- which has nobody to
    # answer it when a display owns the terminal.
    observer = Recorder(stop_after=1)
    train_module.train(tiny_cfg, observer=observer)

    # It stopped inside the first epoch, so the second never ran.
    assert [step for step, _ in observer.epochs] == [0]
    assert observer.batches[-1].epoch == 0


def test_stopping_leaves_a_resumable_checkpoint(tiny_cfg, fake_loader):
    observer = Recorder(stop_after=1)
    train_module.train(tiny_cfg, observer=observer)

    # The same file a Ctrl+C would have written, so `--resume` picks it up the
    # same way, and last.pt still means the newest complete epoch.
    interrupted = tiny_cfg.ckpt_dir / INTERRUPTED_CHECKPOINT
    assert interrupted.is_file()
    assert any("resume with" in message for message in observer.messages)


def test_a_run_that_was_not_stopped_writes_no_interrupt_checkpoint(tiny_cfg, fake_loader):
    train_module.train(tiny_cfg, observer=Recorder())
    assert not (tiny_cfg.ckpt_dir / INTERRUPTED_CHECKPOINT).exists()
