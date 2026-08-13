import dataclasses
import json

import pytest
import torch

from tinydiffusion.diffusion.gaussian_diffusion import GaussianDiffusion
from tinydiffusion.training import train_mnist as train_module
from tinydiffusion.training.config import TrainConfig
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
    batches = [(torch.randn(4, 1, 16, 16), torch.zeros(4, dtype=torch.long)) for _ in range(2)]
    monkeypatch.setattr(train_module, "mnist_dataloader", lambda *a, **k: batches)


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
