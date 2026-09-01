import pytest
import torch

from tinydiffusion.models.blocks import ResBlock
from tinydiffusion.models.unet import UNet


def build(image_size: int, channel_mult: tuple[int, ...], num_classes: int | None = None) -> UNet:
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
        num_classes=num_classes,
    )


@pytest.mark.parametrize(("image_size", "channel_mult"), [(32, (1, 2, 2)), (28, (1, 2)), (8, (1,))])
def test_forward_preserves_the_input_geometry(image_size, channel_mult):
    net = build(image_size, channel_mult)
    x = torch.randn(2, 1, image_size, image_size)
    assert net(x, torch.tensor([0, 5])).shape == x.shape


@pytest.mark.parametrize(("image_size", "channel_mult"), [(30, (1, 2, 2)), (8, (1, 2, 2, 2, 2))])
def test_geometry_that_cannot_be_halved_is_rejected(image_size, channel_mult):
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


def test_labels_change_the_prediction(wake):
    net = wake(build(16, (1, 2), num_classes=4))
    x = torch.randn(1, 1, 16, 16)
    t = torch.tensor([3])

    assert not torch.allclose(net(x, t, torch.tensor([0])), net(x, t, torch.tensor([1])))


def test_an_omitted_label_means_the_null_token(wake):
    net = wake(build(16, (1, 2), num_classes=4))
    x = torch.randn(1, 1, 16, 16)
    t = torch.tensor([3])

    assert torch.equal(net(x, t), net(x, t, torch.tensor([4])))


def test_an_unconditional_net_rejects_labels():
    net = build(16, (1, 2))
    with pytest.raises(ValueError, match="no labels"):
        net(torch.randn(1, 1, 16, 16), torch.tensor([0]), torch.tensor([1]))


def test_a_class_count_of_zero_is_rejected():
    with pytest.raises(ValueError, match="num_classes"):
        build(16, (1, 2), num_classes=0)


def test_the_output_layer_starts_at_zero():
    net = build(16, (1, 2))
    assert torch.equal(net(torch.randn(1, 1, 16, 16), torch.tensor([0])), torch.zeros(1, 1, 16, 16))


def _checkpointing_pair(wake, dropout: float) -> tuple[UNet, UNet]:
    """Two identically weighted nets, one checkpointed and one not.

    Woken, because zero_module leaves the output conv at zero: an untrained net
    predicts zeros, every gradient behind it is zero too, and a comparison of
    two zero gradients holds whatever the checkpointing does.
    """
    torch.manual_seed(0)
    plain = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(8,),
        dropout=dropout,
        image_size=16,
        num_heads=2,
    )
    checkpointed = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(8,),
        dropout=dropout,
        image_size=16,
        num_heads=2,
        use_checkpoint=True,
    )
    wake(plain)
    checkpointed.load_state_dict(plain.state_dict())
    return plain, checkpointed


def _grads(net: UNet, seed: int) -> torch.Tensor:
    net.zero_grad()
    torch.manual_seed(seed)
    x = torch.randn(2, 1, 16, 16)
    net(x, torch.tensor([3, 7])).square().mean().backward()
    grads = torch.cat([p.grad.reshape(-1) for p in net.parameters()])
    assert grads.norm() > 0
    return grads


def test_checkpointing_leaves_the_forward_pass_unchanged(wake):
    plain, checkpointed = _checkpointing_pair(wake, dropout=0.0)
    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([3, 7])
    assert torch.allclose(plain(x, t), checkpointed(x, t), atol=1e-6)


@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_checkpointing_leaves_the_gradients_unchanged(wake, dropout):
    plain, checkpointed = _checkpointing_pair(wake, dropout=dropout)
    assert torch.allclose(_grads(plain, seed=1), _grads(checkpointed, seed=1), atol=1e-6)


def test_checkpointing_recomputes_each_block_in_the_backward_pass(wake):
    plain, checkpointed = _checkpointing_pair(wake, dropout=0.0)

    def count_forwards(net):
        block = next(m for m in net.modules() if isinstance(m, ResBlock))
        calls = []
        body = block._forward
        block._forward = lambda *args: (calls.append(None), body(*args))[1]
        _grads(net, seed=1)
        return len(calls)

    assert count_forwards(plain) == 1
    assert count_forwards(checkpointed) == 2


def test_checkpointing_is_inert_without_a_backward_pass(wake):
    _, checkpointed = _checkpointing_pair(wake, dropout=0.0)
    with torch.no_grad():
        out = checkpointed(torch.randn(2, 1, 16, 16), torch.tensor([3, 7]))
    assert not out.requires_grad


def test_a_checkpointed_net_loads_an_uncheckpointed_state_dict(wake):
    plain, checkpointed = _checkpointing_pair(wake, dropout=0.0)
    assert plain.state_dict().keys() == checkpointed.state_dict().keys()
