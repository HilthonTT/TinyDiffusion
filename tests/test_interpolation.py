import dataclasses

import pytest
import torch

from tinydiffusion.interpolation import interpolate_from_checkpoint, latent_walk, slerp
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
    sample_steps=3,
    batch_size=4,
    num_workers=0,
    device="cpu",
)

CONDITIONAL = dataclasses.replace(TINY, num_classes=10, guidance=2.0)


@pytest.fixture
def make_checkpoint(tmp_path, wake):
    def build(cfg=TINY):
        diffusion = build_model(cfg)
        wake(diffusion.net)
        path = tmp_path / "last.pt"
        save_checkpoint(
            path,
            epoch=0,
            diffusion=diffusion,
            ema=EMA(diffusion.net, decay=0.9, warmup=0),
            optim=torch.optim.Adam(diffusion.parameters(), lr=1e-4),
            scaler=torch.amp.GradScaler("cpu", enabled=False),
            cfg=cfg,
        )
        return path

    return build


@pytest.fixture
def checkpoint(make_checkpoint):
    return make_checkpoint()


def endpoints(dim=64):
    g = torch.Generator().manual_seed(0)
    return torch.randn(dim, generator=g), torch.randn(dim, generator=g)


def test_the_path_starts_and_ends_where_it_was_told_to():
    start, end = endpoints()
    path = slerp(start, end, torch.linspace(0, 1, 5))
    assert torch.allclose(path[0], start, atol=1e-5)
    assert torch.allclose(path[-1], end, atol=1e-5)


def test_the_path_keeps_the_shape_of_its_ends():
    start, end = torch.randn(1, 8, 8), torch.randn(1, 8, 8)
    assert slerp(start, end, torch.linspace(0, 1, 6)).shape == (6, 1, 8, 8)


def test_the_walk_stays_on_the_shell_the_latents_were_drawn_from():
    # The whole reason it is spherical. A Gaussian's mass sits in a thin shell
    # at radius sqrt(d); the midpoint of a straight line between two samples
    # falls well inside it, onto latents the model never saw.
    start, end = endpoints(dim=1024)
    weights = torch.linspace(0, 1, 9)

    spherical = slerp(start, end, weights).norm(dim=1)
    linear = torch.stack([(1 - w) * start + w * end for w in weights]).norm(dim=1)

    ends = torch.stack([start.norm(), end.norm()])
    # Never further from the shell than its own two ends are from each other.
    assert (spherical >= ends.min() - 1e-3).all()
    assert (spherical <= ends.max() + 1e-3).all()
    # Where the straight line has collapsed towards the origin.
    assert linear.min() < 0.8 * ends.min()


def test_the_walk_moves_monotonically_away_from_where_it_started():
    start, end = endpoints(dim=256)
    path = slerp(start, end, torch.linspace(0, 1, 12))
    distances = (path - start).norm(dim=1)
    assert (distances.diff() > 0).all()


def test_parallel_latents_fall_back_to_a_straight_line():
    # The great circle through two parallel vectors is not unique and the
    # spherical formula is 0/0 there; the straight line is its limit.
    start = torch.randn(32, generator=torch.Generator().manual_seed(0))
    path = slerp(start, start * 3.0, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(path[1], start * 2.0, atol=1e-5)


def test_identical_latents_give_a_constant_walk():
    start = torch.randn(16, generator=torch.Generator().manual_seed(0))
    path = slerp(start, start, torch.linspace(0, 1, 4))
    assert torch.allclose(path, start.expand(4, 16), atol=1e-5)


def test_latents_that_disagree_on_shape_are_refused():
    with pytest.raises(ValueError, match="differ in shape"):
        slerp(torch.zeros(8), torch.zeros(9), torch.linspace(0, 1, 3))


def test_a_weight_that_is_not_a_sequence_is_refused():
    with pytest.raises(ValueError, match="must be 1-D"):
        slerp(torch.zeros(8), torch.ones(8), torch.zeros(2, 2))


def test_a_seed_pair_names_the_same_walk_every_time():
    kwargs = {"device": "cpu", "seed_start": 3, "seed_end": 8}
    assert torch.equal(latent_walk((1, 4, 4), 5, **kwargs), latent_walk((1, 4, 4), 5, **kwargs))


def test_different_seeds_give_different_ends():
    first = latent_walk((1, 4, 4), 4, device="cpu", seed_start=0, seed_end=1)
    second = latent_walk((1, 4, 4), 4, device="cpu", seed_start=0, seed_end=2)
    # Same start, since that seed did not change.
    assert torch.equal(first[0], second[0])
    assert not torch.equal(first[-1], second[-1])


def test_a_walk_needs_two_ends():
    with pytest.raises(ValueError, match="at least its two ends"):
        latent_walk((1, 4, 4), 1, device="cpu", seed_start=0, seed_end=1)


def test_interpolating_writes_a_strip(checkpoint, tmp_path):
    out = interpolate_from_checkpoint(checkpoint, tmp_path / "walk.png", steps=4, num_steps=2)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_the_same_seeds_draw_the_same_strip(checkpoint, tmp_path):
    first = interpolate_from_checkpoint(checkpoint, tmp_path / "a.png", steps=3, num_steps=2)
    second = interpolate_from_checkpoint(checkpoint, tmp_path / "b.png", steps=3, num_steps=2)
    assert first.read_bytes() == second.read_bytes()


def test_swapping_the_end_seed_changes_the_strip(checkpoint, tmp_path):
    first = interpolate_from_checkpoint(
        checkpoint, tmp_path / "a.png", steps=3, num_steps=2, seed_end=1
    )
    second = interpolate_from_checkpoint(
        checkpoint, tmp_path / "b.png", steps=3, num_steps=2, seed_end=99
    )
    assert first.read_bytes() != second.read_bytes()


def test_a_conditional_checkpoint_holds_its_label_across_the_walk(make_checkpoint, tmp_path):
    path = make_checkpoint(CONDITIONAL)
    out = interpolate_from_checkpoint(path, tmp_path / "walk.png", steps=4, num_steps=2, labels=[2])
    assert out.is_file()


def test_a_label_outside_the_checkpoints_classes_is_refused(make_checkpoint, tmp_path):
    path = make_checkpoint(CONDITIONAL)
    with pytest.raises(ValueError, match=r"outside \[0, 9\]"):
        interpolate_from_checkpoint(path, tmp_path / "w.png", steps=3, num_steps=2, labels=[99])


def test_labels_are_refused_on_an_unconditional_checkpoint(checkpoint, tmp_path):
    with pytest.raises(ValueError, match="unconditional"):
        interpolate_from_checkpoint(checkpoint, tmp_path / "w.png", steps=3, labels=[1])


def test_guidance_is_refused_on_an_unconditional_checkpoint(checkpoint, tmp_path):
    with pytest.raises(ValueError, match="unconditional"):
        interpolate_from_checkpoint(checkpoint, tmp_path / "w.png", steps=3, guidance=2.0)


def test_a_strip_needs_two_ends(checkpoint, tmp_path):
    with pytest.raises(ValueError, match="at least its two ends"):
        interpolate_from_checkpoint(checkpoint, tmp_path / "w.png", steps=1)


def test_every_sampler_can_draw_a_walk(checkpoint, tmp_path):
    for sampler in ("ddim", "dpmpp"):
        out = interpolate_from_checkpoint(
            checkpoint, tmp_path / f"{sampler}.png", steps=3, num_steps=2, sampler=sampler
        )
        assert out.is_file()
