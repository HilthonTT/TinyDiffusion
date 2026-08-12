import pytest
import torch

from tinydiffusion.models.unet import UNet


def build(image_size: int, channel_mult: tuple[int, ...]) -> UNet:
    return UNet(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_mult=channel_mult,
        num_res_blocks=1,
        attn_resolutions=(),
        dropout=0.0,
        image_size=image_size,
        num_heads=2,
    )


@pytest.mark.parametrize(("image_size", "channel_mult"), [(32, (1, 2, 2)), (28, (1, 2)), (8, (1,))])
def test_forward_preserves_the_input_geometry(image_size, channel_mult):
    net = build(image_size, channel_mult)
    x = torch.randn(2, 1, image_size, image_size)
    assert net(x, torch.tensor([0, 5])).shape == x.shape


@pytest.mark.parametrize(("image_size", "channel_mult"), [(30, (1, 2, 2)), (8, (1, 2, 2, 2, 2))])
def test_geometry_that_cannot_be_halved_is_rejected(image_size, channel_mult):
    # Previously these died inside forward() with a torch.cat size mismatch or
    # a GroupNorm error about a 1x1 bottleneck.
    with pytest.raises(ValueError, match="image_size"):
        build(image_size, channel_mult)


def test_attention_is_placed_at_the_requested_resolution():
    net = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(8,),
        image_size=16,
        num_heads=2,
    )
    x = torch.randn(1, 1, 16, 16)
    assert net(x, torch.tensor([3])).shape == x.shape


def test_the_output_layer_starts_at_zero():
    # zero_module on the final conv: an untrained net predicts no noise at all,
    # which is what makes the initial loss land on E[eps^2] = 1.
    net = build(16, (1, 2))
    assert torch.equal(net(torch.randn(1, 1, 16, 16), torch.tensor([0])), torch.zeros(1, 1, 16, 16))
