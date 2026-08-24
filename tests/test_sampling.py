import dataclasses

import pytest
import torch

from tinydiffusion import sampling
from tinydiffusion.diffusion.guidance import ClassifierFreeGuidance
from tinydiffusion.sampling import resolve_labels, sample_from_checkpoint
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
    num_workers=0,
    device="cpu",
)
CONDITIONAL = dataclasses.replace(TINY, num_classes=10, guidance=2.0)


@pytest.fixture
def make_checkpoint(tmp_path):
    """Write a real checkpoint over a tiny model."""

    def build(cfg=TINY):
        diffusion = build_model(cfg)
        ema = EMA(diffusion.net, decay=0.9, warmup=0)
        optim = torch.optim.Adam(diffusion.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        path = tmp_path / "last.pt"
        save_checkpoint(
            path, epoch=0, diffusion=diffusion, ema=ema, optim=optim, scaler=scaler, cfg=cfg
        )
        return path

    return build


def test_labels_default_to_one_image_per_class():
    labels = resolve_labels(None, num_images=6, num_classes=4, device="cpu")
    assert labels.tolist() == [0, 1, 2, 3, 0, 1]


@pytest.mark.parametrize(
    ("asked", "num_images", "expected"),
    [([3], 4, [3, 3, 3, 3]), ([0, 2], 5, [0, 2, 0, 2, 0]), ([1, 2, 3], 2, [1, 2])],
)
def test_requested_labels_are_cycled_over_the_grid(asked, num_images, expected):
    labels = resolve_labels(asked, num_images=num_images, num_classes=4, device="cpu")
    assert labels.tolist() == expected


def test_an_unconditional_checkpoint_takes_no_labels():
    assert resolve_labels(None, num_images=4, num_classes=None, device="cpu") is None
    with pytest.raises(ValueError, match="unconditional"):
        resolve_labels([1], num_images=4, num_classes=None, device="cpu")


def test_labels_outside_the_class_space_are_rejected():
    with pytest.raises(ValueError, match=r"label\(s\) 4, 9"):
        resolve_labels([0, 4, 9], num_images=4, num_classes=4, device="cpu")
    with pytest.raises(ValueError, match="no labels"):
        resolve_labels([], num_images=4, num_classes=4, device="cpu")


def test_sampling_writes_a_grid(make_checkpoint, tmp_path):
    out = sample_from_checkpoint(make_checkpoint(), tmp_path / "gen.png", num_images=4, seed=0)
    assert out.exists()


@pytest.fixture
def sampled_model(monkeypatch):
    """Capture the model handed to the sampler, without running the chain."""
    seen = {}

    def spy(diffusion, num_samples, size, device, **kwargs):
        seen["model"] = kwargs["model"]
        return torch.zeros(num_samples, *size)

    monkeypatch.setattr(sampling, "get_sampler", lambda name: spy)
    return seen


def test_a_conditional_checkpoint_samples_its_classes(make_checkpoint, tmp_path, sampled_model):
    sample_from_checkpoint(
        make_checkpoint(CONDITIONAL), tmp_path / "gen.png", num_images=4, labels=[2]
    )
    model = sampled_model["model"]

    # guidance=2.0 comes from the checkpoint, so the sampler gets the guided
    # wrapper rather than a plain conditional one.
    assert isinstance(model, ClassifierFreeGuidance)
    assert model.labels.tolist() == [2, 2, 2, 2]
    assert model.scale == 2.0


def test_guidance_can_be_overridden_at_the_command_line(make_checkpoint, tmp_path, sampled_model):
    sample_from_checkpoint(
        make_checkpoint(CONDITIONAL), tmp_path / "gen.png", num_images=2, guidance=4.0
    )
    assert sampled_model["model"].scale == 4.0


def test_guidance_is_rejected_for_an_unconditional_checkpoint(make_checkpoint, tmp_path):
    with pytest.raises(ValueError, match="unconditional"):
        sample_from_checkpoint(make_checkpoint(), tmp_path / "gen.png", guidance=2.0)


def test_num_images_must_be_positive(make_checkpoint, tmp_path):
    with pytest.raises(ValueError, match="num_images"):
        sample_from_checkpoint(make_checkpoint(), tmp_path / "gen.png", num_images=0)


def test_batch_size_must_be_positive(make_checkpoint, tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        sample_from_checkpoint(make_checkpoint(), tmp_path / "gen.png", batch_size=0)


@pytest.fixture
def sampler_calls(monkeypatch):
    """Record the batch size and labels of every call into the sampler."""
    calls = []

    def spy(diffusion, num_samples, size, device, **kwargs):
        model = kwargs["model"]
        calls.append(
            {
                "num_samples": num_samples,
                "noise": kwargs["noise"],
                "labels": None if not hasattr(model, "labels") else model.labels.tolist(),
            }
        )
        return torch.zeros(num_samples, *size)

    monkeypatch.setattr(sampling, "get_sampler", lambda name: spy)
    return calls


def test_a_batch_size_splits_the_draw_into_chunks(make_checkpoint, tmp_path, sampler_calls):
    """Peak memory follows the chunk, which is the whole point of the flag."""
    sample_from_checkpoint(
        make_checkpoint(), tmp_path / "gen.png", num_images=10, batch_size=4, seed=0
    )

    # The ragged last chunk is drawn at its own size, not padded up to 4.
    assert [call["num_samples"] for call in sampler_calls] == [4, 4, 2]


def test_the_whole_draw_is_one_chunk_by_default(make_checkpoint, tmp_path, sampler_calls):
    sample_from_checkpoint(make_checkpoint(), tmp_path / "gen.png", num_images=6, seed=0)

    assert [call["num_samples"] for call in sampler_calls] == [6]


def test_a_batch_size_larger_than_the_draw_is_not_padded(make_checkpoint, tmp_path, sampler_calls):
    sample_from_checkpoint(
        make_checkpoint(), tmp_path / "gen.png", num_images=3, batch_size=99, seed=0
    )

    assert [call["num_samples"] for call in sampler_calls] == [3]


def test_the_chunks_carry_the_latents_the_unbatched_draw_would_have(
    make_checkpoint, tmp_path, sampler_calls
):
    """Batch-invariance, at the seam where it could break: the latents."""
    checkpoint = make_checkpoint()
    sample_from_checkpoint(checkpoint, tmp_path / "one.png", num_images=6, seed=7)
    whole = sampler_calls[0]["noise"]

    sampler_calls.clear()
    sample_from_checkpoint(checkpoint, tmp_path / "many.png", num_images=6, batch_size=2, seed=7)
    split = torch.cat([call["noise"] for call in sampler_calls])

    assert torch.equal(whole, split)


def test_labels_stay_with_their_images_across_a_split(make_checkpoint, tmp_path, sampler_calls):
    """A chunk taking the whole label vector would relabel every image after the first."""
    sample_from_checkpoint(
        make_checkpoint(CONDITIONAL),
        tmp_path / "gen.png",
        num_images=6,
        batch_size=2,
        labels=[0, 1, 2],
        seed=0,
    )

    assert [call["labels"] for call in sampler_calls] == [[0, 1], [2, 0], [1, 2]]


def test_batching_does_not_change_the_images(make_checkpoint, tmp_path):
    """The real chain, end to end: same seed and same images however it is split."""
    checkpoint = make_checkpoint()
    whole = sample_from_checkpoint(checkpoint, tmp_path / "one.png", num_images=6, seed=3)
    split = sample_from_checkpoint(
        checkpoint, tmp_path / "many.png", num_images=6, batch_size=2, seed=3
    )

    assert whole.read_bytes() == split.read_bytes()


def test_save_individual_writes_one_file_per_image(make_checkpoint, tmp_path):
    out = sample_from_checkpoint(
        make_checkpoint(), tmp_path / "gen.png", num_images=3, save_individual=True, seed=0
    )

    assert out.exists()
    written = sorted(path.name for path in tmp_path.glob("gen_*.png"))
    assert written == ["gen_0000.png", "gen_0001.png", "gen_0002.png"]


def test_individual_files_are_not_written_unless_asked(make_checkpoint, tmp_path):
    sample_from_checkpoint(make_checkpoint(), tmp_path / "gen.png", num_images=3, seed=0)

    assert not list(tmp_path.glob("gen_*.png"))


@pytest.mark.parametrize(
    ("num_images", "num_classes", "labels", "expected"),
    [
        (16, None, None, 8),  # unconditional: the usual eight per row
        (16, 4, None, 4),  # a default class cycle: one class per column
        (16, 4, [1], 8),  # labels asked for: no column meaning to preserve
        (2, 4, None, 2),  # never wider than the grid itself
    ],
)
def test_grid_width(num_images, num_classes, labels, expected):
    assert sampling.grid_width(num_images, num_classes, labels) == expected
