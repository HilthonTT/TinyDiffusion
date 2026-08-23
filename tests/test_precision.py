import pytest
import torch
import torch.nn as nn

from tinydiffusion.utils import precision as precision_module
from tinydiffusion.utils.precision import (
    DEFAULT_PRECISION,
    PRECISIONS,
    Autocast,
    apply_precision,
    resolve_precision,
)


class Spy(nn.Module):
    """A network standing in for the UNet: records its call, returns a feature map."""

    def __init__(self, out_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(1, out_channels, 3, padding=1)
        self.seen = []

    def forward(self, x, t, y=None):
        self.seen.append((x.dtype, x.is_contiguous(memory_format=torch.channels_last), y))
        return self.conv(x)


def image(batch=2):
    return torch.randn(batch, 1, 8, 8)


def test_the_default_is_float32():
    assert DEFAULT_PRECISION == "fp32"
    assert DEFAULT_PRECISION in PRECISIONS


@pytest.mark.parametrize("name", PRECISIONS)
def test_every_advertised_precision_resolves(name):
    assert resolve_precision(name, "cpu") in PRECISIONS


def test_an_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="unknown precision"):
        resolve_precision("float8", "cpu")
    with pytest.raises(ValueError, match="unknown precision"):
        apply_precision(Spy(), "float8", "cpu")


def test_float32_is_left_alone_wherever_it_runs():
    net = Spy()
    assert resolve_precision("fp32", "cpu") == "fp32"
    assert resolve_precision("fp32", "cuda") == "fp32"
    # Not wrapped, not relaid out: fp32 has to be exactly what it was before
    # this module existed, or a score taken today is not comparable with one
    # taken before it.
    assert apply_precision(net, "fp32", "cpu") is net


@pytest.mark.parametrize("name", ["tf32", "fp16", "bf16"])
def test_anything_but_float32_falls_back_off_cuda(name, capsys):
    assert resolve_precision(name, "cpu") == "fp32"
    assert "CUDA" in capsys.readouterr().out


def test_bf16_without_hardware_falls_back_to_fp16(monkeypatch, capsys):
    # torch reports the emulation path as supported, so the check has to be
    # bf16_supported rather than torch's own answer.
    monkeypatch.setattr(precision_module, "bf16_supported", lambda: False)
    assert resolve_precision("bf16", "cuda") == "fp16"
    assert "emulates bfloat16" in capsys.readouterr().out


def test_bf16_survives_where_the_hardware_has_it(monkeypatch):
    monkeypatch.setattr(precision_module, "bf16_supported", lambda: True)
    assert resolve_precision("bf16", "cuda") == "bf16"


def test_a_fallback_can_be_kept_quiet(capsys):
    assert resolve_precision("fp16", "cpu", verbose=False) == "fp32"
    assert capsys.readouterr().out == ""


def test_tf32_sets_the_backend_flag_rather_than_wrapping(monkeypatch):
    called = []
    monkeypatch.setattr(precision_module, "enable_tf32", lambda: called.append(True))
    net = Spy()
    # A flag, not a wrapper: the network is handed back untouched, so the
    # dtype and the layout are both still float32/NCHW.
    assert apply_precision(net, "tf32", "cuda") is net
    assert called == [True]


@pytest.mark.parametrize("name", ["fp16", "bf16"])
def test_half_precision_wraps_the_network(name):
    net = Spy()
    wrapped = apply_precision(net, name, "cpu")
    assert isinstance(wrapped, Autocast)
    assert wrapped.net is net


def test_the_wrapper_hands_back_float32_contiguous():
    net = Spy()
    wrapped = Autocast(net, "cpu", torch.bfloat16)
    out = wrapped(image(), torch.zeros(2, dtype=torch.long))
    # Everything downstream — guidance, the schedule, the solver update — is
    # written for float32 NCHW, and gains nothing from half precision.
    assert out.dtype is torch.float32
    assert out.is_contiguous()


def test_the_wrapper_feeds_the_network_channels_last():
    net = Spy()
    wrapped = Autocast(net, "cpu", torch.bfloat16)
    wrapped(image(), torch.zeros(2, dtype=torch.long))
    _, was_channels_last, _ = net.seen[0]
    # The whole point of the layout: NCHW half precision transposes per
    # convolution instead of using the tensor cores.
    assert was_channels_last


def test_the_wrapper_passes_labels_through():
    net = Spy()
    wrapped = Autocast(net, "cpu", torch.bfloat16)
    labels = torch.tensor([3, 4])
    wrapped(image(), torch.zeros(2, dtype=torch.long), labels)
    assert net.seen[0][2] is labels


def test_the_wrapper_adopts_the_networks_mode():
    # nn.Module defaults to training=True, so a wrapper that did not ask would
    # put an eval-mode network back into training and re-enable its dropout.
    net = Spy().eval()
    assert not Autocast(net, "cpu", torch.float16).training
    assert Autocast(Spy().train(), "cpu", torch.float16).training


def test_the_wrapper_handles_a_doubled_output_width():
    # A learned variance emits 2C channels; nothing here may assume otherwise.
    net = Spy(out_channels=2)
    out = Autocast(net, "cpu", torch.bfloat16)(image(), torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 2, 8, 8)


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["tf32", "fp16", "bf16"])
def test_a_real_chain_stays_finite_in_half_precision(name, wake):
    """A whole DDIM chain at each precision, on the hardware that runs them.

    Nothing else covers this: every other test here resolves to fp32 on a CPU
    runner, and the failure half precision actually has — an overflow part way
    down the chain that turns the latent to NaN and never recovers — only shows
    up once the kernels are real.
    """
    from tinydiffusion.diffusion.guidance import conditioned, cycled_labels
    from tinydiffusion.diffusion.samplers import get_sampler
    from tinydiffusion.training.config import TrainConfig
    from tinydiffusion.training.model import build_model

    cfg = TrainConfig(
        base_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(16,),
        num_classes=10,
        num_timesteps=100,
        image_size=32,
        device="cuda",
    )
    diffusion = build_model(cfg).to("cuda")
    wake(diffusion.net)

    net = apply_precision(diffusion.net, resolve_precision(name, "cuda"), "cuda")
    y = cycled_labels(4, 10, "cuda")
    out = get_sampler("ddim")(
        diffusion,
        4,
        (1, 32, 32),
        "cuda",
        num_steps=5,
        model=conditioned(net, y, num_classes=10, scale=2.0),
    )

    assert out.shape == (4, 1, 32, 32)
    # Back in float32 for the caller, whatever the network ran in.
    assert out.dtype is torch.float32
    assert torch.isfinite(out).all()
    # clip_denoised holds every step to the model's range, so a chain that
    # overflowed shows up here as well as in the NaN check.
    assert out.abs().max() <= 1.0 + 1e-4
