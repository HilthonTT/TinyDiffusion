import dataclasses

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tinydiffusion import evaluation
from tinydiffusion.evaluation import DEFAULT_EVAL_STEPS, EvalResult, eval_timesteps
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


@pytest.fixture
def make_checkpoint(tmp_path, monkeypatch, wake):
    """Write a real checkpoint over a tiny model, and stand in for MNIST."""

    def build(cfg=TINY, labels=None, trained=False):
        diffusion = build_model(cfg)
        if trained:
            # Stand in for training: an all-zero output conv makes the loss
            # independent of everything, conditioning included.
            wake(diffusion.net)
        ema = EMA(diffusion.net, decay=0.9, warmup=0)
        optim = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        path = tmp_path / "last.pt"
        save_checkpoint(
            path, epoch=0, diffusion=diffusion, ema=ema, optim=optim, scaler=scaler, cfg=cfg
        )

        def fake_loader(*args, **kwargs):
            images = torch.randn(6, 1, cfg.image_size, cfg.image_size).clamp(-1, 1)
            y = torch.zeros(6, dtype=torch.long) if labels is None else labels
            return DataLoader(TensorDataset(images, y), batch_size=3)

        monkeypatch.setattr(evaluation, "image_dataloader", fake_loader)
        return path

    return build


@pytest.fixture
def checkpoint(make_checkpoint):
    return make_checkpoint()


def test_eval_timesteps_span_the_schedule():
    steps = eval_timesteps(100, 5)
    assert steps.tolist() == [0, 25, 50, 74, 99]
    assert steps.dtype == torch.long


def test_eval_timesteps_rejects_impossible_counts():
    with pytest.raises(ValueError, match="num_steps"):
        eval_timesteps(10, 0)
    with pytest.raises(ValueError, match="num_steps"):
        eval_timesteps(10, 11)


def test_evaluate_returns_a_result(checkpoint):
    result = evaluation.evaluate_checkpoint(checkpoint, num_steps=3, progress=False)
    assert isinstance(result, EvalResult)
    assert result.num_images == 6
    assert result.split == "test"
    assert result.used_ema is True
    assert len(result.per_timestep) == 3
    assert [t for t, _ in result.per_timestep] == sorted(t for t, _ in result.per_timestep)
    assert result.loss > 0


def test_evaluation_is_reproducible(checkpoint):
    first = evaluation.evaluate_checkpoint(checkpoint, num_steps=3, progress=False)
    second = evaluation.evaluate_checkpoint(checkpoint, num_steps=3, progress=False)
    assert first.loss == second.loss
    assert first.per_timestep == second.per_timestep


def test_seed_changes_the_noise(checkpoint):
    first = evaluation.evaluate_checkpoint(checkpoint, num_steps=3, seed=0, progress=False)
    second = evaluation.evaluate_checkpoint(checkpoint, num_steps=3, seed=1, progress=False)
    assert first.loss != second.loss


def test_raw_weights_can_be_scored(checkpoint):
    result = evaluation.evaluate_checkpoint(checkpoint, num_steps=2, use_ema=False, progress=False)
    assert result.used_ema is False


def test_unknown_split_is_rejected(checkpoint):
    with pytest.raises(ValueError, match="unknown split"):
        evaluation.evaluate_checkpoint(checkpoint, split="valid", progress=False)


def test_format_reports_the_headline_and_table(checkpoint):
    text = evaluation.evaluate_checkpoint(checkpoint, num_steps=2, progress=False).format()
    assert "test split" in text
    assert "6 images" in text
    assert "ema weights" in text
    assert text.count("\n") >= 5


def test_a_conditional_checkpoint_is_scored_on_its_labels(make_checkpoint):
    # The label reaches the network, so scoring the same images under
    # different labels has to give a different loss. Were the labels dropped
    # on the floor, the two would be identical.
    ones = make_checkpoint(CONDITIONAL, labels=torch.ones(6, dtype=torch.long), trained=True)
    with_ones = evaluation.evaluate_checkpoint(ones, num_steps=2, progress=False)

    twos = make_checkpoint(CONDITIONAL, labels=torch.full((6,), 2, dtype=torch.long), trained=True)
    with_twos = evaluation.evaluate_checkpoint(twos, num_steps=2, progress=False)

    assert with_ones.loss != with_twos.loss


def test_default_step_count_is_sane():
    assert 1 <= DEFAULT_EVAL_STEPS <= 1000
