import dataclasses

import pytest
import torch
import torch.nn as nn

from tinydiffusion.diffusion.ddim import (
    DEFAULT_SPACING,
    SPACINGS,
    ddim_sample,
    get_spacing,
    quadratic_timesteps,
    spacing_names,
    uniform_timesteps,
)
from tinydiffusion.diffusion.ddpm import DDPM
from tinydiffusion.diffusion.dpm_solver import dpmpp_sample
from tinydiffusion.diffusion.samplers import SAMPLERS
from tinydiffusion.diffusion.schedules import linear_beta_schedule
from tinydiffusion.training.config import TrainConfig

T = 100


def test_the_registry_holds_both_published_spacings():
    assert spacing_names() == ("quadratic", "uniform")
    assert SPACINGS["uniform"] is uniform_timesteps
    assert SPACINGS["quadratic"] is quadratic_timesteps


def test_the_default_is_uniform():
    assert DEFAULT_SPACING == "uniform"
    assert get_spacing(DEFAULT_SPACING) is uniform_timesteps


def test_an_unknown_spacing_names_the_alternatives():
    with pytest.raises(ValueError, match="unknown timestep spacing 'linear'") as exc:
        get_spacing("linear")
    assert "quadratic" in str(exc.value)
    assert "uniform" in str(exc.value)


def test_quadratic_is_denser_near_zero_than_uniform():
    # The whole point of the option: at a low step count the quadratic grid
    # spends its steps where the chain has least room to correct itself.
    even = uniform_timesteps(T, 10)
    packed = quadratic_timesteps(T, 10)
    assert (packed < T // 2).sum() > (even < T // 2).sum()


class _Recorder(nn.Module):
    """Records every timestep it is asked about, and denoises predictably."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[int] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.seen.append(int(t[0]))
        return x * 0.1


@pytest.fixture
def recorder():
    net = _Recorder()
    return DDPM(net, betas=linear_beta_schedule(1e-4, 0.02, T), num_timesteps=T), net


@pytest.mark.parametrize("sampler", [ddim_sample, dpmpp_sample])
def test_every_sampler_visits_the_timesteps_the_spacing_names(recorder, sampler):
    diffusion, net = recorder
    sampler(diffusion, 1, (1, 4, 4), "cpu", num_steps=8, eta=0.0, spacing="quadratic")
    assert net.seen == quadratic_timesteps(T, 8).tolist()


@pytest.mark.parametrize("sampler", [ddim_sample, dpmpp_sample])
def test_every_sampler_defaults_to_uniform(recorder, sampler):
    diffusion, net = recorder
    sampler(diffusion, 1, (1, 4, 4), "cpu", num_steps=8, eta=0.0)
    assert net.seen == uniform_timesteps(T, 8).tolist()


@pytest.mark.parametrize("sampler", [ddim_sample, dpmpp_sample])
def test_an_explicit_subsequence_still_wins(recorder, sampler):
    diffusion, net = recorder
    asked = torch.tensor([90, 40, 5])
    sampler(
        diffusion,
        1,
        (1, 4, 4),
        "cpu",
        num_steps=8,
        eta=0.0,
        timesteps=asked,
        spacing="quadratic",
    )
    assert net.seen == asked.tolist()


@pytest.mark.parametrize("sampler", [ddim_sample, dpmpp_sample])
def test_an_unknown_spacing_is_rejected_by_every_sampler(recorder, sampler):
    diffusion, _ = recorder
    with pytest.raises(ValueError, match="unknown timestep spacing"):
        sampler(diffusion, 1, (1, 4, 4), "cpu", num_steps=4, eta=0.0, spacing="nope")


def test_the_spacing_changes_the_samples(recorder):
    diffusion, _ = recorder
    noise = torch.randn(2, 1, 4, 4)
    kwargs = {"num_steps": 8, "eta": 0.0, "noise": noise}
    even = ddim_sample(diffusion, 2, (1, 4, 4), "cpu", spacing="uniform", **kwargs)
    packed = ddim_sample(diffusion, 2, (1, 4, 4), "cpu", spacing="quadratic", **kwargs)
    assert not torch.allclose(even, packed)


def test_every_registered_sampler_accepts_the_spacing_keyword(recorder):
    # The Sampler protocol promises it, so a sampler added later that quietly
    # drops the argument should fail here rather than in a scoring run.
    diffusion, _ = recorder
    for draw in SAMPLERS.values():
        out = draw(diffusion, 1, (1, 4, 4), "cpu", num_steps=4, eta=0.0, spacing="quadratic")
        assert out.shape == (1, 1, 4, 4)


def test_the_config_defaults_to_uniform():
    assert TrainConfig().sample_spacing == DEFAULT_SPACING


def test_the_config_accepts_a_registered_spacing():
    cfg = TrainConfig(sample_spacing="quadratic")
    assert cfg.sample_spacing == "quadratic"


def test_the_config_rejects_an_unknown_spacing():
    with pytest.raises(ValueError, match="unknown timestep spacing"):
        TrainConfig(sample_spacing="cosine")


def test_the_spacing_survives_a_config_round_trip():
    cfg = TrainConfig(sample_spacing="quadratic")
    restored = TrainConfig.from_mapping(dataclasses.asdict(cfg))
    assert restored.sample_spacing == "quadratic"
