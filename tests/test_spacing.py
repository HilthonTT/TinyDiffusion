import dataclasses

import pytest
import torch
import torch.nn as nn

from tinydiffusion.diffusion.ddim import (
    DEFAULT_SPACING,
    KARRAS_RHO,
    SPACINGS,
    ddim_sample,
    get_spacing,
    karras_timesteps,
    quadratic_timesteps,
    schedule_sigmas,
    spacing_names,
    uniform_timesteps,
)
from tinydiffusion.diffusion.ddpm import DDPM
from tinydiffusion.diffusion.dpm_solver import dpmpp_sample
from tinydiffusion.diffusion.samplers import SAMPLERS
from tinydiffusion.diffusion.schedules import (
    cosine_beta_schedule,
    ddpm_schedules,
    enforce_zero_terminal_snr,
    linear_beta_schedule,
)
from tinydiffusion.training.config import TrainConfig

T = 100


def test_the_registry_holds_every_published_spacing():
    assert spacing_names() == ("karras", "quadratic", "uniform")
    assert SPACINGS["uniform"] is uniform_timesteps
    assert SPACINGS["quadratic"] is quadratic_timesteps
    assert SPACINGS["karras"] is karras_timesteps


def test_the_default_is_uniform():
    assert DEFAULT_SPACING == "uniform"
    assert get_spacing(DEFAULT_SPACING) is uniform_timesteps


def test_an_unknown_spacing_names_the_alternatives():
    with pytest.raises(ValueError, match="unknown timestep spacing 'linear'") as exc:
        get_spacing("linear")
    assert "quadratic" in str(exc.value)
    assert "uniform" in str(exc.value)


def test_quadratic_is_denser_near_zero_than_uniform():
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


def cosine_alphabar(num_timesteps=T):
    return ddpm_schedules(cosine_beta_schedule(num_timesteps))["alphabar_t"]


def linear_alphabar(num_timesteps=T):
    return ddpm_schedules(linear_beta_schedule(1e-4, 0.02, num_timesteps))["alphabar_t"]


def test_the_index_spacings_ignore_the_schedule_they_are_handed():
    for spacing in (uniform_timesteps, quadratic_timesteps):
        assert torch.equal(spacing(T, 10), spacing(T, 10, alphabar=cosine_alphabar()))


def test_karras_starts_at_pure_noise_and_ends_at_zero():
    steps = karras_timesteps(T, 10, alphabar=linear_alphabar())
    assert steps[0] == T - 1
    assert steps[-1] == 0


def test_karras_is_strictly_descending():
    steps = karras_timesteps(T, 20, alphabar=linear_alphabar())
    assert (steps.diff() < 0).all()


def test_a_single_karras_step_is_the_noisiest_one():
    assert karras_timesteps(T, 1, alphabar=linear_alphabar()).tolist() == [T - 1]


def test_karras_spaces_by_noise_rather_than_by_index():
    alphabar = linear_alphabar(1000)
    sigmas = schedule_sigmas(alphabar)
    rho = KARRAS_RHO

    def ramp_error(steps):
        visited = sigmas[steps].double()
        warped = visited ** (1 / rho)
        return float(warped.diff().std() / warped.diff().abs().mean())

    karras = ramp_error(karras_timesteps(1000, 20, alphabar=alphabar))
    uniform = ramp_error(uniform_timesteps(1000, 20))
    assert karras < uniform


def test_karras_needs_the_schedule():
    with pytest.raises(ValueError, match=r"needs.*alphabar"):
        karras_timesteps(T, 5)


def test_a_schedule_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="expected num_timesteps"):
        karras_timesteps(T, 5, alphabar=linear_alphabar(T + 1))


def test_karras_survives_a_zero_terminal_snr_schedule():
    betas = enforce_zero_terminal_snr(cosine_beta_schedule(T))
    alphabar = ddpm_schedules(betas)["alphabar_t"]
    steps = karras_timesteps(T, 10, alphabar=alphabar)
    assert steps[0] == T - 1
    assert steps[-1] == 0
    assert (steps.diff() < 0).all()


def test_a_cosine_schedule_cannot_honour_the_step_count():
    cosine = karras_timesteps(1000, 20, alphabar=cosine_alphabar(1000))
    linear = karras_timesteps(1000, 20, alphabar=linear_alphabar(1000))
    assert len(cosine) < 20
    assert len(linear) == 20


def test_karras_returns_timesteps_on_the_host_like_the_others():
    alphabar = linear_alphabar()
    assert karras_timesteps(T, 5, alphabar=alphabar).device == uniform_timesteps(T, 5).device


@pytest.mark.parametrize("steps", [2, 5, 17, T])
def test_karras_never_leaves_the_schedule(steps):
    out = karras_timesteps(T, steps, alphabar=linear_alphabar())
    assert out.min() >= 0
    assert out.max() <= T - 1
    assert out.dtype is torch.long
