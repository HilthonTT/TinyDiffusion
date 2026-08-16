import copy

import pytest
import torch
import torch.nn as nn

from tinydiffusion.training.ema import EMA


def _net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.GroupNorm(1, 2), nn.Conv2d(2, 1, 1))


def test_the_fused_update_matches_a_lerp_per_parameter():
    # torch._foreach_lerp_ has to be the same arithmetic as the loop it
    # replaced, not merely close: the average is folded in thousands of times.
    net = _net()
    ema = EMA(net, decay=0.9, warmup=0)
    before = [p.clone() for p in ema.module.parameters()]

    # Move the live weights so the average has somewhere to travel.
    for p in net.parameters():
        nn.init.normal_(p, std=0.5)
    ema.update(net)

    for averaged, start, live in zip(
        ema.module.parameters(), before, net.parameters(), strict=True
    ):
        assert torch.equal(averaged, start.lerp_(live.detach(), 0.1))


def test_buffers_are_copied_rather_than_averaged():
    net = _net()
    ema = EMA(net, decay=0.9, warmup=0)
    net_with_buffer = copy.deepcopy(net)
    net_with_buffer.register_buffer("counter", torch.tensor(7))
    ema_with_buffer = EMA(net_with_buffer, decay=0.9, warmup=0)

    net_with_buffer.counter.fill_(9)
    ema_with_buffer.update(net_with_buffer)

    assert ema_with_buffer.module.counter.item() == 9
    assert not list(ema.module.buffers())


def test_averaging_a_differently_shaped_model_is_refused():
    ema = EMA(_net(), decay=0.9, warmup=0)
    with pytest.raises(ValueError, match="parameter tensors"):
        ema.update(nn.Conv2d(1, 2, 3))


def test_the_decay_ramps_in_over_the_warmup():
    ema = EMA(_net(), decay=0.999, warmup=10)
    assert ema.current_decay == pytest.approx(2 / 11)
    for _ in range(20):
        ema.update(_net())
    assert ema.current_decay == 0.999
