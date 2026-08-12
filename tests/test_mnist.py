import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler

from tinydiffusion.data import MNIST_NATIVE_SIZE, denormalize, mnist_transform
from tinydiffusion.data import mnist as mnist_module


def _fake_digit():
    return Image.new("L", (MNIST_NATIVE_SIZE, MNIST_NATIVE_SIZE), color=255)


def test_transform_produces_a_single_channel_at_the_requested_size():
    out = mnist_transform(32)(_fake_digit())
    assert out.shape == (1, 32, 32)
    assert out.dtype == torch.float32


def test_transform_maps_pixels_into_the_model_range():
    white = mnist_transform(32)(_fake_digit())
    black = mnist_transform(32)(Image.new("L", (MNIST_NATIVE_SIZE, MNIST_NATIVE_SIZE), color=0))
    assert torch.allclose(white, torch.ones_like(white))
    assert torch.allclose(black, -torch.ones_like(black))


def test_denormalize_inverts_the_transform():
    x = torch.linspace(-1.0, 1.0, 16)
    assert torch.allclose(denormalize(x), (x + 1) / 2)


def test_denormalize_clamps_out_of_range_samples():
    out = denormalize(torch.tensor([-4.0, 4.0]))
    assert torch.equal(out, torch.tensor([0.0, 1.0]))


@pytest.fixture
def fake_mnist(monkeypatch):
    """Stand in for the real dataset so no download is needed."""
    images = torch.zeros(10, 1, 8, 8)
    monkeypatch.setattr(mnist_module, "mnist_dataset", lambda *a, **k: [(img, 0) for img in images])


def test_the_training_split_shuffles_and_drops_the_ragged_batch(fake_mnist):
    loader = mnist_module.mnist_dataloader(train=True, batch_size=4, num_workers=0)
    assert isinstance(loader.sampler, RandomSampler)
    assert loader.drop_last is True
    assert sum(x.shape[0] for x, _ in loader) == 8


def test_scoring_can_keep_every_image_in_order(fake_mnist):
    # What evaluation needs: dropping the last batch would omit images from
    # the average, and shuffling would make the batching order RNG-dependent.
    loader = mnist_module.mnist_dataloader(
        train=True, batch_size=4, num_workers=0, shuffle=False, drop_last=False
    )
    assert not isinstance(loader.sampler, RandomSampler)
    assert loader.drop_last is False
    assert sum(x.shape[0] for x, _ in loader) == 10
