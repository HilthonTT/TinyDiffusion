import torch
from PIL import Image

from tinydiffusion.data import MNIST_NATIVE_SIZE, denormalize, mnist_transform


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
