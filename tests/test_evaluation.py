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


def test_each_batch_draws_its_own_noise(checkpoint, monkeypatch):
    """Reseeding with one seed would score every batch against one draw."""
    seeds = []
    real = evaluation.seed_everything

    def record(seed):
        seeds.append(seed)
        real(seed)

    monkeypatch.setattr(evaluation, "seed_everything", record)
    evaluation.evaluate_checkpoint(checkpoint, num_steps=2, seed=7, progress=False)
    # Six images at three a batch, so two batches, on consecutive seeds.
    assert seeds == [7, 8]


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


# The variational bound needs the generalised process, which the config picks
# by asking for anything the plain DDPM path does not implement.
IMPROVED = dataclasses.replace(TINY, variance="learned_range", objective="rescaled_mse")


def test_the_bound_is_absent_unless_it_is_asked_for(checkpoint):
    result = evaluation.evaluate_checkpoint(checkpoint, num_steps=3, progress=False)
    assert result.bpd is None
    assert result.prior_bpd is None
    assert result.num_bpd_images == 0
    assert "bpd" not in result.format()


def test_the_bound_is_reported_in_bits_per_dimension(make_checkpoint):
    path = make_checkpoint(IMPROVED, trained=True)
    result = evaluation.evaluate_checkpoint(
        path, num_steps=3, progress=False, bpd=True, bpd_images=3
    )

    assert result.bpd is not None
    assert result.prior_bpd is not None
    assert torch.isfinite(torch.tensor([result.bpd, result.prior_bpd])).all()
    # The prior term is one summand of the total, and every other term is a KL
    # or a negative log-likelihood, so it can only be the smaller of the two.
    assert result.prior_bpd <= result.bpd
    assert result.num_bpd_images == 3
    assert "bpd" in result.format()


def test_the_bound_stops_once_it_has_the_images_it_was_asked_for(make_checkpoint):
    """It costs a network evaluation per timestep, so the cap has to bind."""
    path = make_checkpoint(IMPROVED, trained=True)
    result = evaluation.evaluate_checkpoint(
        path, num_steps=3, progress=False, bpd=True, bpd_images=3
    )
    # Six images in the stand-in loader, in batches of three: the loss covers
    # all of them and the bound stops after the first batch.
    assert result.num_images == 6
    assert result.num_bpd_images == 3


def test_a_plain_ddpm_checkpoint_says_why_it_has_no_bound(checkpoint):
    """The default parameterisation is served by DDPM, which defines none."""
    with pytest.raises(ValueError, match="generalised process"):
        evaluation.evaluate_checkpoint(checkpoint, num_steps=3, progress=False, bpd=True)


def test_an_empty_bound_slice_is_refused(make_checkpoint):
    path = make_checkpoint(IMPROVED)
    with pytest.raises(ValueError, match="bpd_images must be positive"):
        evaluation.evaluate_checkpoint(path, num_steps=3, progress=False, bpd=True, bpd_images=0)


def test_the_bound_is_reproducible(make_checkpoint):
    """Same weights, same images, same seed: the forward noise is the only draw.

    The stand-in loader draws its images from the global RNG when it is built,
    so it is reseeded here as well — the real one reads the same files twice
    and needs no help.
    """
    path = make_checkpoint(IMPROVED, trained=True)
    kwargs = {"num_steps": 3, "progress": False, "bpd": True, "bpd_images": 3}
    torch.manual_seed(0)
    first = evaluation.evaluate_checkpoint(path, **kwargs)
    torch.manual_seed(0)
    second = evaluation.evaluate_checkpoint(path, **kwargs)
    assert first.bpd == pytest.approx(second.bpd)
