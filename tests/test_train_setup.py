"""The decisions a run makes before its first batch.

Everything here is settled by the time training starts, so it is checked
directly rather than by training an epoch and reading what came out.
"""

import dataclasses

import pytest
import torch

from tinydiffusion.data.datasets import dataset_spec
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.distributed import Distributed
from tinydiffusion.training.model import build_model
from tinydiffusion.training.plan import describe_plan
from tinydiffusion.training.setup import parameter_sets, resolve_precision


@pytest.fixture
def cfg() -> TrainConfig:
    return TrainConfig(
        image_size=16,
        batch_size=4,
        base_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=(),
        num_timesteps=10,
        num_epochs=4,
        sample_steps=5,
        val_steps=5,
        device="cpu",
    )


@pytest.fixture
def said() -> list[str]:
    return []


def test_half_precision_is_refused_on_a_cpu_and_said_so(cfg, said):
    precision = resolve_precision(dataclasses.replace(cfg, full_fp16=True), said.append)

    assert not precision.full_fp16
    assert not precision.amp
    assert precision.label == "amp off"
    assert said == ["full_fp16 needs a CUDA device; training in float32 instead"]


def test_amp_is_off_on_a_cpu_however_loudly_the_config_asks(cfg, said):
    precision = resolve_precision(dataclasses.replace(cfg, amp=True), said.append)

    assert not precision.amp
    assert said == []


def test_bfloat16_falls_back_to_fp16_where_the_gpu_only_emulates_it(cfg, monkeypatch, said):
    monkeypatch.setattr("tinydiffusion.training.setup.bf16_supported", lambda: False)
    real_device = torch.device
    monkeypatch.setattr(torch, "device", lambda _: real_device("cuda"))

    precision = resolve_precision(
        dataclasses.replace(cfg, device="cuda", amp=True, amp_dtype="bf16"), said.append
    )

    assert precision.amp_dtype is torch.float16
    assert precision.label == "amp fp16"
    assert said == ["this GPU emulates bfloat16 rather than running it, falling back to fp16"]


def test_bfloat16_survives_where_the_gpu_runs_it(cfg, monkeypatch, said):
    monkeypatch.setattr("tinydiffusion.training.setup.bf16_supported", lambda: True)
    real_device = torch.device
    monkeypatch.setattr(torch, "device", lambda _: real_device("cuda"))

    precision = resolve_precision(
        dataclasses.replace(cfg, device="cuda", amp=True, amp_dtype="bf16"), said.append
    )

    assert precision.amp_dtype is torch.bfloat16
    assert precision.label == "amp bf16"
    assert said == []


def test_the_label_carries_the_modifiers_the_config_asked_for(cfg):
    precision = resolve_precision(
        dataclasses.replace(cfg, compile=True, channels_last=True), lambda _: None
    )

    assert precision.label == "amp off | compiled | channels_last"


def test_the_scaler_is_enabled_only_where_it_earns_its_keep(cfg):
    off = resolve_precision(cfg, lambda _: None)

    assert not off.grad_scaler().is_enabled()


def test_without_a_master_copy_the_optimiser_steps_the_network_itself(cfg):
    diffusion = build_model(cfg)

    model_params, master_params, step_params = parameter_sets(diffusion, full_fp16=False)

    assert master_params is None
    assert step_params is model_params
    assert model_params == list(diffusion.parameters())


def test_a_master_copy_is_what_the_optimiser_steps_instead(cfg):
    diffusion = build_model(cfg)

    model_params, master_params, step_params = parameter_sets(diffusion, full_fp16=True)

    assert master_params is not None
    assert step_params == [*master_params]
    assert len(master_params) < len(model_params)
    assert sum(p.numel() for p in master_params) == sum(p.numel() for p in model_params)


def plan(cfg, group=None, **kwargs) -> str:
    defaults = {
        "n_params": 1_000_000,
        "precision": "amp off",
        "start_epoch": 0,
        "steps_per_epoch": 10,
        "validation_images": 0,
    }
    return describe_plan(
        cfg, dataset_spec(cfg.dataset), group or Distributed(), **(defaults | kwargs)
    )


def test_a_fresh_run_reports_the_whole_span_of_epochs(cfg):
    assert "4 epochs" in plan(cfg)


def test_a_resumed_run_reports_the_epochs_that_are_left(cfg):
    assert "epochs 3-4 (2 to go)" in plan(cfg, start_epoch=2)


def test_a_checkpoint_past_the_end_does_not_render_a_backwards_range(cfg):
    assert "nothing to run (checkpoint is at 4 epochs)" in plan(cfg, start_epoch=4)


def test_one_epoch_is_singular(cfg):
    assert "1 epoch |" in plan(dataclasses.replace(cfg, num_epochs=1))


def test_an_unconditional_run_says_so(cfg):
    assert "unconditional" in plan(cfg)


def test_a_conditional_run_reports_its_classes_and_dropout(cfg):
    line = plan(dataclasses.replace(cfg, num_classes=10, class_dropout=0.1))

    assert "10 classes, 0.1 label dropout" in line


def test_a_run_without_validation_says_so_rather_than_reporting_zero(cfg):
    assert "no validation" in plan(cfg)


def test_validation_is_reported_as_images_per_interval(cfg):
    line = plan(dataclasses.replace(cfg, val_every=2), validation_images=512)

    assert "512 held-out images every 2 epochs" in line


def test_a_plain_run_reports_steps_without_an_effective_batch(cfg):
    assert "10 steps/epoch |" in plan(cfg)


def test_accumulation_and_ranks_are_named_separately_in_the_effective_batch(cfg):
    line = plan(
        dataclasses.replace(cfg, batch_size=4, grad_accum=2),
        group=Distributed(enabled=True, rank=0, local_rank=0, world_size=4),
    )

    assert "10 steps/epoch (x2 accumulated, x4 ranks, 32 effective)" in line


def test_ranks_alone_do_not_report_a_meaningless_accumulation_factor(cfg):
    line = plan(cfg, group=Distributed(enabled=True, rank=0, local_rank=0, world_size=2))

    assert "x2 ranks, 8 effective" in line
    assert "accumulated" not in line
