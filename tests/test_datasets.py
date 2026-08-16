import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler

from tinydiffusion.data import (
    DATASETS,
    dataset_names,
    dataset_spec,
    denormalize,
    image_transform,
)
from tinydiffusion.data import datasets as datasets_module

MNIST = dataset_spec("mnist")
CIFAR = dataset_spec("cifar10")


def _fake_digit(size=MNIST.native_size):
    return Image.new("L", (size, size), color=255)


def test_every_registered_spec_is_keyed_under_its_own_name():
    assert all(name == spec.name for name, spec in DATASETS.items())
    assert dataset_names() == tuple(sorted(DATASETS))


def test_an_unregistered_dataset_names_the_ones_that_exist():
    with pytest.raises(ValueError, match=r"unknown dataset 'imagenet'.*mnist"):
        dataset_spec("imagenet")


def test_transform_produces_the_specs_channels_at_the_requested_size():
    out = image_transform(MNIST.channels, 32)(_fake_digit())
    assert out.shape == (1, 32, 32)
    assert out.dtype == torch.float32


def test_a_three_channel_spec_normalises_every_channel():
    rgb = Image.new("RGB", (CIFAR.native_size, CIFAR.native_size), color=(255, 255, 255))
    out = image_transform(CIFAR.channels, 32)(rgb)
    assert out.shape == (3, 32, 32)
    assert torch.allclose(out, torch.ones_like(out))


def test_transform_maps_pixels_into_the_model_range():
    white = image_transform(1, 32)(_fake_digit())
    black = image_transform(1, 32)(Image.new("L", (MNIST.native_size,) * 2, color=0))
    assert torch.allclose(white, torch.ones_like(white))
    assert torch.allclose(black, -torch.ones_like(black))


def test_a_flip_is_only_added_when_asked_for():
    # The flip draws from the global RNG, so a scored split must never get one.
    plain = image_transform(3, 32)
    flipped = image_transform(3, 32, hflip=True)
    assert len(flipped.transforms) == len(plain.transforms) + 1


def test_the_digit_sets_opt_out_of_horizontal_flips():
    # A mirrored 2 is not a 2, so the augmentation would teach the wrong label.
    assert not dataset_spec("mnist").hflip
    assert dataset_spec("cifar10").hflip


@pytest.fixture
def fake_dataset(monkeypatch):
    """Stand in for the real dataset so no download is needed."""
    images = torch.zeros(10, 1, 8, 8)
    monkeypatch.setattr(
        datasets_module, "image_dataset", lambda *a, **k: [(img, 0) for img in images]
    )


def test_the_training_split_shuffles_and_drops_the_ragged_batch(fake_dataset):
    loader = datasets_module.image_dataloader(MNIST, train=True, batch_size=4, num_workers=0)
    assert isinstance(loader.sampler, RandomSampler)
    assert loader.drop_last is True
    assert sum(x.shape[0] for x, _ in loader) == 8


def test_scoring_can_keep_every_image_in_order(fake_dataset):
    # What evaluation needs: dropping the last batch would omit images from
    # the average, and shuffling would make the batching order RNG-dependent.
    loader = datasets_module.image_dataloader(
        MNIST, train=True, batch_size=4, num_workers=0, shuffle=False, drop_last=False
    )
    assert not isinstance(loader.sampler, RandomSampler)
    assert loader.drop_last is False
    assert sum(x.shape[0] for x, _ in loader) == 10


def test_augmentation_reaches_the_dataset_only_when_asked(monkeypatch):
    seen = []
    items = [(torch.zeros(3, 8, 8), 0)] * 4

    def record(spec, root, **kwargs):
        seen.append(kwargs["augment"])
        return items

    monkeypatch.setattr(datasets_module, "image_dataset", record)
    datasets_module.image_dataloader(CIFAR, train=True, num_workers=0, augment=True)
    datasets_module.image_dataloader(CIFAR, train=False, num_workers=0)
    assert seen == [True, False]


def test_denormalize_inverts_the_transform():
    x = torch.linspace(-1.0, 1.0, 16)
    assert torch.allclose(denormalize(x), (x + 1) / 2)


def test_denormalize_clamps_out_of_range_samples():
    out = denormalize(torch.tensor([-4.0, 4.0]))
    assert torch.equal(out, torch.tensor([0.0, 1.0]))
