import dataclasses

import pytest
import torch
import torch.nn as nn

from tinydiffusion.models.blocks import Float32GroupNorm, group_norm
from tinydiffusion.models.unet import UNet
from tinydiffusion.training import train as train_module
from tinydiffusion.training.checkpoints import read_checkpoint
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.ema import EMA
from tinydiffusion.utils.fp16 import (
    convert_module_to_f16,
    convert_module_to_f32,
    make_master_params,
    master_params_to_model_params,
    master_params_to_state_dict,
    model_grads_to_master_grads,
    model_params_to_master_params,
    unflatten_master_params,
    zero_grad,
)


def tiny_unet(**kwargs):
    defaults = {
        "in_channels": 1,
        "out_channels": 1,
        "base_channels": 8,
        "channel_mult": (1, 2),
        "num_res_blocks": 1,
        "attn_resolutions": (8,),
        "dropout": 0.0,
        "image_size": 16,
    }
    return UNet(**{**defaults, **kwargs})


# --- converting modules ----------------------------------------------------


def test_conversion_moves_convolution_weights_and_biases():
    conv = nn.Conv2d(3, 4, 3)

    convert_module_to_f16(conv)
    assert conv.weight.dtype is torch.float16
    assert conv.bias.dtype is torch.float16

    convert_module_to_f32(conv)
    assert conv.weight.dtype is torch.float32
    assert conv.bias.dtype is torch.float32


def test_conversion_survives_a_convolution_without_a_bias():
    # bias=False is ordinary inside a normalised stack, and reaching for
    # `.bias.data` unconditionally is an AttributeError on it.
    conv = nn.Conv2d(3, 4, 3, bias=False)

    convert_module_to_f16(conv)

    assert conv.weight.dtype is torch.float16
    assert conv.bias is None


def test_conversion_leaves_everything_that_is_not_a_convolution_alone():
    # Normalisation and the embedding MLPs are where half precision costs
    # accuracy, and they are a rounding error in the parameter count.
    block = nn.Sequential(nn.Linear(4, 4), nn.GroupNorm(2, 4), nn.Embedding(3, 4))

    block.apply(convert_module_to_f16)

    assert {p.dtype for p in block.parameters()} == {torch.float32}


# --- master parameters -----------------------------------------------------


def test_master_parameters_are_one_flat_float32_copy():
    net = tiny_unet()
    net.convert_to_fp16()

    master = make_master_params(net.parameters())

    assert len(master) == 1
    assert master[0].dtype is torch.float32
    assert master[0].numel() == sum(p.numel() for p in net.parameters())
    assert master[0].requires_grad


def test_building_master_parameters_from_nothing_is_an_error():
    # An empty flat tensor is a parameter the optimiser steps forever without
    # ever complaining that there is nothing there.
    with pytest.raises(ValueError, match="no parameters"):
        make_master_params([])


def test_master_parameters_round_trip_through_the_model():
    net = tiny_unet()
    master = make_master_params(net.parameters())
    net.convert_to_fp16()

    with torch.no_grad():
        master[0].add_(0.25)
    master_params_to_model_params(net.parameters(), master)

    values = unflatten_master_params(list(net.parameters()), master)
    for param, value in zip(net.parameters(), values, strict=True):
        assert torch.allclose(param.float(), value, atol=1e-3)


def test_model_parameters_can_be_copied_back_onto_the_master():
    # What a --resume needs: the checkpoint is loaded into the float32 network
    # and the master copy has to pick it up before the fp16 conversion.
    net = tiny_unet()
    master = make_master_params(net.parameters())
    with torch.no_grad():
        for param in net.parameters():
            param.fill_(0.5)

    model_params_to_master_params(net.parameters(), master)

    assert torch.allclose(master[0], torch.full_like(master[0], 0.5))


def test_gradients_reach_the_master_copy_in_float32():
    net = tiny_unet()
    net.convert_to_fp16()
    master = make_master_params(net.parameters())
    net(torch.randn(2, 1, 16, 16), torch.randint(0, 10, (2,))).sum().backward()

    model_grads_to_master_grads(list(net.parameters()), master)

    assert master[0].grad is not None
    assert master[0].grad.dtype is torch.float32
    assert master[0].grad.numel() == master[0].numel()


def test_a_parameter_with_no_gradient_becomes_zeros_rather_than_a_hole():
    # The flat copy is positional: dropping one parameter's gradient would
    # shift every later one onto the wrong weights.
    params = [torch.ones(2, 2, requires_grad=True), torch.ones(3, requires_grad=True)]
    params[1].grad = torch.full((3,), 7.0)
    master = make_master_params(params)

    model_grads_to_master_grads(params, master)

    assert torch.equal(master[0].grad, torch.tensor([0.0, 0.0, 0.0, 0.0, 7.0, 7.0, 7.0]))


def test_zeroing_gradients_keeps_the_tensors_around():
    param = torch.ones(3, requires_grad=True)
    param.grad = torch.full((3,), 5.0)

    zero_grad([param])

    assert param.grad is not None
    assert not param.grad.any()


# --- the state dict a run in this mode checkpoints -------------------------


def test_the_master_state_dict_is_float32_even_though_the_model_is_not():
    net = tiny_unet()
    master = make_master_params(net.parameters())
    net.convert_to_fp16()

    state = master_params_to_state_dict(net, master)

    assert set(state) == set(net.state_dict())
    assert {value.dtype for value in state.values()} == {torch.float32}


def test_the_master_state_dict_holds_copies_not_views_of_the_flat_tensor():
    # torch.save writes a view's whole underlying storage, so saving the
    # unflattened views would write the entire network once per parameter.
    net = tiny_unet()
    master = make_master_params(net.parameters())

    state = master_params_to_state_dict(net, master)

    stored = sum(value.untyped_storage().nbytes() for value in state.values())
    assert stored == sum(value.numel() * value.element_size() for value in state.values())


# --- the network -----------------------------------------------------------


def test_group_norm_normalises_in_float32_whatever_it_is_handed():
    # F.group_norm refuses a half input against a float32 weight on CUDA, so
    # this is what lets fp16 convolutions and fp32 norms share a network.
    norm = group_norm(4)
    assert isinstance(norm, Float32GroupNorm)

    out = norm(torch.randn(2, 4, 8, 8, dtype=torch.float16))

    assert out.dtype is torch.float16


def test_converting_the_unet_halves_the_convolutions_and_spares_the_rest():
    net = tiny_unet()

    net.convert_to_fp16()

    assert net.dtype is torch.float16
    assert net.init_conv.weight.dtype is torch.float16
    # The output head, every norm, and the timestep MLP stay full precision.
    assert net.out[2].weight.dtype is torch.float32
    assert net.out[0].weight.dtype is torch.float32
    assert net.time_embed.mlp[0].weight.dtype is torch.float32


def test_a_converted_unet_takes_and_returns_float32():
    # The caller never sees the half precision: the loss, the schedule and
    # every sampler go on working in float32.
    net = tiny_unet()
    net.convert_to_fp16()

    out = net(torch.randn(2, 1, 16, 16), torch.randint(0, 10, (2,)))

    assert out.dtype is torch.float32
    assert out.shape == (2, 1, 16, 16)


def test_converting_back_restores_an_ordinary_float32_network():
    net = tiny_unet()
    net.convert_to_fp16()

    net.convert_to_fp32()

    assert net.dtype is torch.float32
    assert {p.dtype for p in net.parameters()} == {torch.float32}


def test_a_converted_unet_predicts_what_the_float32_one_does():
    net = tiny_unet(num_classes=10)
    for param in net.parameters():
        nn.init.normal_(param, std=0.05)
    x, t, y = torch.randn(2, 1, 16, 16), torch.randint(0, 10, (2,)), torch.randint(0, 10, (2,))
    expected = net(x, t, y)

    net.convert_to_fp16()

    assert torch.allclose(net(x, t, y), expected, atol=2e-3)


# --- the EMA ---------------------------------------------------------------


def test_the_ema_averages_the_master_copy_when_it_is_given_one():
    net = tiny_unet()
    ema = EMA(net, decay=0.5, warmup=0)
    before = next(ema.module.parameters()).clone()
    master = make_master_params(net.parameters())
    with torch.no_grad():
        master[0].fill_(1.0)
    net.convert_to_fp16()

    ema.update(net, unflatten_master_params(list(net.parameters()), master))

    assert {p.dtype for p in ema.module.parameters()} == {torch.float32}
    # Half the old average, half the master value it was handed — and float32
    # throughout, which is the whole point: the network itself is half now.
    assert torch.allclose(next(ema.module.parameters()), 0.5 * (before + 1.0), atol=1e-6)


def test_the_ema_says_so_rather_than_silently_failing_on_half_weights():
    # torch._foreach_lerp_ rejects the mixed pair several frames down without
    # naming a tensor, and folding a 1e-4 increment into a float16 weight would
    # not move it anyway.
    net = tiny_unet()
    ema = EMA(net, decay=0.9999, warmup=0)
    net.convert_to_fp16()

    with pytest.raises(ValueError, match="full-precision weights as `params`"):
        ema.update(net)


# --- the config ------------------------------------------------------------


def test_full_fp16_needs_amp_on():
    with pytest.raises(ValueError, match="full_fp16 is a mixed-precision mode"):
        TrainConfig(full_fp16=True, amp=False)


def test_full_fp16_has_nothing_to_do_with_bf16():
    with pytest.raises(ValueError, match="has nothing to"):
        TrainConfig(full_fp16=True, amp_dtype="bf16")


def test_full_fp16_round_trips_through_a_config_mapping():
    cfg = TrainConfig.from_mapping({"full_fp16": True})
    assert cfg.full_fp16


# --- the training loop -----------------------------------------------------


@pytest.fixture
def fp16_cfg(tmp_path) -> TrainConfig:
    return TrainConfig(
        image_size=16,
        batch_size=4,
        num_workers=0,
        base_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=(),
        num_timesteps=10,
        num_epochs=1,
        ema_warmup=0,
        lr_warmup=0,
        full_fp16=True,
        sample_every=0,
        val_every=0,
        num_samples=2,
        sample_steps=5,
        out_dir=tmp_path / "contents",
        ckpt_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "runs",
    )


@pytest.fixture
def fake_loader(monkeypatch):
    batches = [
        (torch.randn(4, 1, 16, 16), torch.arange(4, dtype=torch.long) % 10) for _ in range(2)
    ]
    monkeypatch.setattr(train_module, "image_dataloader", lambda *a, **k: batches)


def test_full_fp16_falls_back_to_float32_off_cuda(fp16_cfg, fake_loader, capsys):
    # Half precision on a CPU is emulated, so it is slower than float32 rather
    # than faster, and the run says which one it actually used.
    train_module.train(dataclasses.replace(fp16_cfg, device="cpu"))

    out = capsys.readouterr().out
    assert "full_fp16 needs a CUDA device" in out
    assert "fp16 weights" not in out


@pytest.mark.gpu
def test_a_bf16_run_falls_back_to_fp16_where_the_card_only_emulates_it(
    fp16_cfg, fake_loader, monkeypatch, capsys
):
    # torch reports emulated bf16 as supported, and emulating it costs nearly
    # five times what the fp16 it was chosen over does. The run has to notice.
    monkeypatch.setattr(train_module, "bf16_supported", lambda: False)
    cfg = dataclasses.replace(fp16_cfg, device="cuda", full_fp16=False, amp=True, amp_dtype="bf16")

    train_module.train(cfg)

    out = capsys.readouterr().out
    assert "emulates bfloat16" in out
    # And the plan line reports what it actually ran, not what was asked for.
    assert "amp fp16" in out
    assert "amp bf16" not in out


@pytest.mark.gpu
def test_a_full_fp16_run_trains_and_hands_back_a_float32_model(fp16_cfg, fake_loader):
    diffusion = train_module.train(dataclasses.replace(fp16_cfg, device="cuda"))

    assert {p.dtype for p in diffusion.parameters()} == {torch.float32}


@pytest.mark.gpu
def test_a_full_fp16_run_writes_an_ordinary_float32_checkpoint(fp16_cfg, fake_loader):
    # The float16 weights are a rounded copy of the master ones, so writing
    # them would ship a checkpoint slightly worse than the run actually had.
    train_module.train(dataclasses.replace(fp16_cfg, device="cuda"))

    ckpt = read_checkpoint(fp16_cfg.ckpt_dir / "last.pt")

    assert {v.dtype for v in ckpt["model"].values()} == {torch.float32}
    assert {v.dtype for v in ckpt["ema"].values()} == {torch.float32}


@pytest.mark.gpu
def test_a_full_fp16_run_scales_its_gradients(fp16_cfg, fake_loader):
    # Without the scaler there is no unscaled path at all here, and diffusion
    # gradients sit close enough to float16's floor to flush to zero.
    import json

    from tinydiffusion.utils.tracking import METRICS_FILENAME

    train_module.train(dataclasses.replace(fp16_cfg, device="cuda"))

    record = json.loads((fp16_cfg.log_dir / METRICS_FILENAME).read_text().splitlines()[0])
    assert record["train/amp_scale"] > 1.0


@pytest.mark.gpu
def test_resuming_across_the_full_fp16_boundary_keeps_the_weights(fp16_cfg, fake_loader, capsys):
    # AdamW's moments cannot cross — one flat tensor on one side, a few hundred
    # on the other — but the weights are the same float32 either way.
    train_module.train(dataclasses.replace(fp16_cfg, device="cuda"))
    resume = fp16_cfg.ckpt_dir / "last.pt"
    before = read_checkpoint(resume)["model"]

    plain = dataclasses.replace(fp16_cfg, device="cuda", full_fp16=False, num_epochs=1)
    train_module.train(plain, resume=resume)

    out = capsys.readouterr().out
    assert "fresh AdamW moments" in out
    # num_epochs is already covered by the checkpoint, so nothing trained and
    # the weights that came back out are the ones that went in.
    after = read_checkpoint(resume)["model"]
    assert all(torch.equal(before[k], after[k]) for k in before)
