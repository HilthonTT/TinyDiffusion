from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from tinydiffusion.diffusion.ddim import ddim_sample
from tinydiffusion.diffusion.ddpm import DDPM
from tinydiffusion.diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from tinydiffusion.diffusion.schedules import linear_beta_schedule
from tinydiffusion.training.checkpoints import restore_checkpoint, save_checkpoint
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.ema import EMA
from tinydiffusion.training.model import build_model
from tinydiffusion.training.train import save_samples

T = 20
SHAPE = (2, 1, 8, 8)


class _Net(nn.Module):
    """Emits `out_mult` * C channels, shaped like its input."""

    def __init__(self, out_mult: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(1, out_mult, 3, padding=1)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, x, t):
        return self.conv(x) + self.scale * t.float().reshape(-1, 1, 1, 1)


def _betas():
    return linear_beta_schedule(1e-4, 0.02, T)


def _learned():
    return GaussianDiffusion.improved(_Net(out_mult=2), betas=_betas(), num_timesteps=T)


def test_defaults_reproduce_the_ddpm_loss():
    """Epsilon + fixed-small + MSE is DDPM, so both classes must agree."""
    net = _Net()
    x = torch.randn(*SHAPE)
    t = torch.randint(0, T, (SHAPE[0],))
    noise = torch.randn_like(x)

    gaussian = GaussianDiffusion(net, betas=_betas(), num_timesteps=T)
    ddpm = DDPM(net, betas=_betas(), num_timesteps=T)

    assert torch.allclose(
        gaussian.loss_at(x, t, noise=noise), ddpm.loss_at(x, t, noise=noise), atol=1e-6
    )


def test_q_sample_matches_the_closed_form():
    diffusion = GaussianDiffusion(_Net(), betas=_betas(), num_timesteps=T)
    x = torch.randn(*SHAPE)
    t = torch.full((SHAPE[0],), T - 1)
    noise = torch.randn_like(x)
    expected = (
        diffusion.sqrtab[T - 1] * x + diffusion.sqrtmab[T - 1] * noise  # broadcast scalars
    )
    assert torch.allclose(diffusion.q_sample(x, t, noise=noise), expected, atol=1e-6)


def test_hybrid_objective_splits_out_its_terms():
    diffusion = _learned()
    terms = diffusion.training_losses(torch.randn(*SHAPE), torch.randint(0, T, (SHAPE[0],)))
    assert set(terms) == {"loss", "mse", "vb"}
    assert all(term.shape == (SHAPE[0],) for term in terms.values())
    assert torch.allclose(terms["loss"], terms["mse"] + terms["vb"])


def test_loss_terms_matches_the_ddpm_surface():
    diffusion = _learned()
    terms = diffusion.loss_terms(torch.randn(*SHAPE))
    assert terms.loss.ndim == 0
    assert terms.per_sample.shape == (SHAPE[0],)
    assert terms.timesteps.shape == (SHAPE[0],)
    assert not terms.per_sample.requires_grad
    terms.loss.backward()
    assert diffusion.net.conv.weight.grad is not None


def test_sampling_runs_with_a_learned_variance():
    diffusion = _learned()
    samples = diffusion.sample(2, SHAPE[1:], "cpu")
    assert samples.shape == SHAPE
    assert torch.isfinite(samples).all()


def test_ddim_handles_the_doubled_output_channels():
    """ddim_sample must route through p_mean_variance, not call the net raw."""
    diffusion = _learned()
    samples = ddim_sample(diffusion, 2, SHAPE[1:], "cpu", num_steps=4)
    assert samples.shape == SHAPE
    assert torch.isfinite(samples).all()


def test_bpd_loop_totals_the_per_timestep_terms():
    diffusion = _learned()
    result = diffusion.calc_bpd_loop(torch.randn(*SHAPE).clamp(-1, 1))
    assert result["vb"].shape == (SHAPE[0], T)
    assert torch.isfinite(result["total_bpd"]).all()
    assert torch.allclose(
        result["total_bpd"], result["vb"].sum(dim=1) + result["prior_bpd"], atol=1e-4
    )


def test_rescaled_mse_needs_a_learned_variance():
    with pytest.raises(ValueError, match="RESCALED_MSE"):
        GaussianDiffusion(
            _Net(),
            betas=_betas(),
            num_timesteps=T,
            loss_type=LossType.RESCALED_MSE,
        )


# ---------------------------------------------------------------------------
# wiring through the config and the training entry points
# ---------------------------------------------------------------------------

TINY = TrainConfig(
    image_size=8,
    base_channels=4,
    channel_mult=(1,),
    num_res_blocks=1,
    attn_resolutions=(),
    num_timesteps=T,
    sample_steps=4,
    num_samples=2,
    batch_size=2,
    num_workers=0,
    device="cpu",
)


def _improved_config(**overrides):
    return replace(TINY, **{"variance": "learned_range", "objective": "rescaled_mse", **overrides})


def test_default_config_still_builds_a_ddpm():
    assert isinstance(build_model(TINY), DDPM)


def test_improved_config_builds_a_gaussian_diffusion():
    diffusion = build_model(_improved_config())
    assert isinstance(diffusion, GaussianDiffusion)
    assert diffusion.model_mean_type is ModelMeanType.EPSILON
    assert diffusion.model_var_type is ModelVarType.LEARNED_RANGE
    assert diffusion.loss_type is LossType.RESCALED_MSE


def test_learned_variance_doubles_the_output_channels():
    diffusion = build_model(_improved_config())
    out = diffusion.net(torch.randn(1, 1, 8, 8), torch.zeros(1, dtype=torch.long))
    assert out.shape == (1, 2, 8, 8)


def test_learned_variance_needs_an_objective_that_trains_it():
    with pytest.raises(ValueError, match="objective"):
        _improved_config(objective="mse")


def test_rescaled_mse_without_a_learned_variance_is_rejected():
    with pytest.raises(ValueError, match="rescaled_mse"):
        _improved_config(variance="fixed_small")


def test_unknown_parameterisation_is_rejected():
    with pytest.raises(ValueError, match="bad diffusion parameterisation"):
        _improved_config(predict="velocity")


def test_checkpoints_round_trip_a_gaussian_diffusion(tmp_path):
    cfg = _improved_config()
    diffusion = build_model(cfg)
    ema = EMA(diffusion.net, decay=0.9, warmup=0)
    optim = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path, epoch=2, diffusion=diffusion, ema=ema, optim=optim, scaler=scaler, cfg=cfg
    )

    fresh = build_model(cfg)
    fresh_ema = EMA(fresh.net, decay=0.9, warmup=0)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert restore_checkpoint(ckpt, diffusion=fresh, ema=fresh_ema) == 3
    assert ckpt["config"]["variance"] == "learned_range"
    for saved, loaded in zip(
        diffusion.net.state_dict().values(), fresh.net.state_dict().values(), strict=True
    ):
        assert torch.equal(saved, loaded)


def test_sample_grid_writes_for_a_gaussian_diffusion(tmp_path):
    cfg = _improved_config(out_dir=tmp_path)
    diffusion = build_model(cfg)
    ema = EMA(diffusion.net, decay=0.9, warmup=0)
    save_samples(diffusion, ema, torch.randn(2, 1, 8, 8), cfg, 0)
    assert (tmp_path / "sample_0001.png").exists()
