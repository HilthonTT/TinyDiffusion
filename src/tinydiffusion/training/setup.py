"""The decisions a run makes before its first batch.

A config asks for a precision, a device, a compiled network; what it gets
depends on the hardware it landed on. Resolving all of that here, ahead of the
loop, is what lets :mod:`~tinydiffusion.training.loop` assume every choice has
already been made -- and what lets the choices be checked without training an
epoch to observe them.
"""

import importlib.util
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler

from tinydiffusion.diffusion.gaussian_diffusion import Diffusion
from tinydiffusion.diffusion.timesteps import LossSecondMomentResampler
from tinydiffusion.training.checkpoints import restore_checkpoint, restore_rng_state
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.distributed import Distributed, all_gather_cat
from tinydiffusion.training.ema import EMA
from tinydiffusion.training.model import build_model
from tinydiffusion.training.plan import epochs_phrase
from tinydiffusion.utils.device import bf16_supported, enable_tf32
from tinydiffusion.utils.fp16 import make_master_params

__all__ = [
    "Precision",
    "build_network",
    "can_compile",
    "configure_backends",
    "parameter_sets",
    "resolve_precision",
    "restore_run",
    "share_timestep_sampler",
    "wrap_network",
]


def can_compile(device_type: str) -> bool:
    """Whether :func:`torch.compile` can actually build kernels here.

    Inductor needs Triton for its CUDA backend, and Triton is not part of the
    Windows PyTorch wheels. Without this check the failure lands on the first
    batch, several frames inside dynamo, long after the run has downloaded a
    dataset and printed its plan — and it is a hard stop rather than something
    the run can carry on without.

    Args:
        device_type: ``"cuda"``, ``"cpu"``, or whatever the run resolved to.

    Returns:
        True when compiling is worth attempting. The CPU backend generates C++
        rather than Triton, so it is not subject to the same check.
    """
    if device_type != "cuda":
        return True
    return importlib.util.find_spec("triton") is not None


def share_timestep_sampler(diffusion: Diffusion, group: Distributed) -> None:
    """Let an adaptive timestep proposal see the whole group's losses.

    :class:`~tinydiffusion.diffusion.timesteps.LossSecondMomentResampler`
    builds its proposal from the losses it is shown, and each rank is shown
    only its own shard of the global batch. Left alone, an eight-way run warms
    eight private proposals on an eighth of the evidence each — which is not
    wrong, since the importance weights still make every rank's own estimator
    unbiased, but it is eight times the variance the group paid for.

    Handing it a gather is what makes the history the group's rather than the
    rank's. Nothing happens for the uniform sampler, which has no history, or
    outside a group, where the collective would have nothing to collect.

    Args:
        diffusion: the process this run is training.
        group: the training group.
    """
    sampler = getattr(diffusion, "timestep_sampler", None)
    if group.enabled and isinstance(sampler, LossSecondMomentResampler):
        sampler.gather = partial(all_gather_cat, group=group)


@dataclass(frozen=True)
class Precision:
    """How a run carries its numbers, once the request has met the hardware.

    A config asks for a precision; what it gets depends on the device it landed
    on. Resolving that once, here, is what keeps the decision, the notices
    explaining it and the plan line reporting it from drifting apart:
    :attr:`label` is built from the same three fields that :meth:`autocast` and
    :meth:`grad_scaler` read, so a run cannot announce ``amp bf16`` and then go
    on to run float16.

    Attributes:
        device_type: ``"cuda"``, ``"cpu"``, or whatever the run resolved to.
        full_fp16: float16 weights with a float32 master copy. CUDA only, and
            never true alongside `amp`.
        amp: autocast is on.
        amp_dtype: the dtype autocast will actually use, which is not always
            the one the config asked for; see :func:`resolve_precision`.
        label: how the plan line describes all of the above.
    """

    device_type: str
    full_fp16: bool
    amp: bool
    amp_dtype: torch.dtype
    label: str

    def autocast(self) -> AbstractContextManager[None]:
        """The context the forward pass runs in.

        Returns:
            An autocast context, or one that does nothing when this run is not
            using autocast.
        """
        return torch.amp.autocast(
            self.device_type, dtype=self.amp_dtype if self.amp else None, enabled=self.amp
        )

    def grad_scaler(self) -> torch.amp.GradScaler:
        """Build the gradient scaler this precision needs.

        bf16 carries fp32's exponent range, so there is nothing to scale
        against and no step to skip when the scale would have overflowed.
        `full_fp16` is the opposite case and needs it most: with the weights
        themselves in half precision there is no unscaled path at all, and
        diffusion gradients sit close enough to float16's floor that an
        unscaled backward pass flushes a good share of them to zero.

        Returns:
            A scaler, enabled only where one earns its keep.
        """
        return torch.amp.GradScaler(
            self.device_type,
            enabled=self.full_fp16 or (self.amp and self.amp_dtype is torch.float16),
        )


def resolve_precision(cfg: TrainConfig, say: Callable[[str], None] = print) -> Precision:
    """Resolve a config's precision request against the device it will run on.

    Two half-precision strategies, and at most one of them runs. ``full_fp16``
    puts float16 in the weights themselves and keeps a float32 master copy for
    the optimiser; autocast leaves the weights alone and casts per kernel. Both
    want CUDA — half precision on a CPU is emulated, so it is slower than
    float32 rather than faster.

    Every downgrade is said rather than applied quietly: a run that asked for
    bf16 and was given fp16 will not reach the throughput its config implies,
    and an unexplained slowdown is the worst way to find that out.

    Args:
        cfg: run configuration.
        say: where the downgrade notices go. Defaults to printing them.

    Returns:
        The resolved precision, including the label the plan line reports.
    """
    device_type = torch.device(cfg.device).type
    full_fp16 = cfg.full_fp16 and device_type == "cuda"
    if cfg.full_fp16 and not full_fp16:
        say("full_fp16 needs a CUDA device; training in float32 instead")
    amp = cfg.amp and device_type == "cuda" and not full_fp16
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    if amp and amp_dtype is torch.bfloat16 and not bf16_supported():
        say("this GPU emulates bfloat16 rather than running it, falling back to fp16")
        amp_dtype = torch.float16

    if full_fp16:
        label = "fp16 weights (float32 master)"
    elif amp:
        label = f"amp {'bf16' if amp_dtype is torch.bfloat16 else 'fp16'}"
    else:
        label = "amp off"
    if cfg.compile:
        label += " | compiled"
    if cfg.channels_last:
        label += " | channels_last"

    return Precision(
        device_type=device_type, full_fp16=full_fp16, amp=amp, amp_dtype=amp_dtype, label=label
    )


def configure_backends(cfg: TrainConfig, device_type: str) -> None:
    """Turn on the CUDA fast paths a run with fixed input shapes can use.

    Args:
        cfg: run configuration. ``deterministic`` is what holds the autotuner
            back.
        device_type: the resolved device type. Nothing happens off CUDA.
    """
    if device_type != "cuda":
        return
    torch.backends.cudnn.benchmark = not cfg.deterministic
    enable_tf32()


def build_network(cfg: TrainConfig, group: Distributed) -> tuple[Diffusion, EMA]:
    """Build the process and its EMA.

    ``diffusion.net`` is the canonical network — what the checkpoint, the EMA
    and every sampler read. The module the loop actually calls comes from
    :func:`wrap_network`, and only once the weights are in their final form.

    Args:
        cfg: run configuration.
        group: the training group, for the per-rank seed.

    Returns:
        ``(diffusion, ema)``, sharing one set of parameters.
    """
    diffusion = build_model(cfg).to(cfg.device)
    share_timestep_sampler(diffusion, group)
    if group.enabled:
        torch.manual_seed(cfg.seed + group.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed + group.rank)
    if cfg.channels_last:
        diffusion.net.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    ema = EMA(diffusion.net, decay=cfg.ema_decay, warmup=cfg.ema_warmup)
    return diffusion, ema


def wrap_network(
    net: nn.Module,
    cfg: TrainConfig,
    group: Distributed,
    precision: Precision,
    say: Callable[[str], None],
) -> tuple[nn.Module, DistributedDataParallel | None]:
    """Wrap the network in whatever the training step calls.

    That may be a DDP wrapper, a compiled wrapper, or both. Keeping them apart
    from the canonical network is what stops a compiled or distributed run
    writing checkpoints whose keys all carry a ``_orig_mod.`` or ``module.``
    prefix.

    Call this after the checkpoint is restored and after any float16
    conversion: DDP sizes and types its gradient buckets from the parameters
    it is handed, so a network converted to half *after* wrapping produces
    gradients the reducer rejects. Wrapping last also means DDP's own
    parameter broadcast carries the restored weights to every rank.

    Args:
        net: the canonical network.
        cfg: run configuration.
        group: the training group. Outside one, no DDP wrapper is built.
        precision: the resolved precision, for the device index DDP wants.
        say: where the compile fallback notice goes.

    Returns:
        ``(train_net, ddp)``. `ddp` is None outside a group, and comes back
        separately from `train_net` because ``no_sync`` is a DDP method that
        compiling would put a wrapper in front of.
    """
    device_type = precision.device_type
    train_net: nn.Module = net
    ddp: DistributedDataParallel | None = None
    if group.enabled:
        ddp = DistributedDataParallel(
            net,
            device_ids=[group.local_rank] if device_type == "cuda" else None,
            output_device=group.local_rank if device_type == "cuda" else None,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        train_net = ddp
    if cfg.compile and can_compile(device_type):
        train_net = cast(nn.Module, torch.compile(train_net))
    elif cfg.compile:
        say(
            "compile is on but Triton is not installed, so the CUDA backend cannot build "
            "kernels; training eagerly instead (pip install triton-windows on Windows)"
        )
    return train_net, ddp


def parameter_sets(
    diffusion: Diffusion, *, full_fp16: bool
) -> tuple[list[torch.Tensor], list[nn.Parameter] | None, list[torch.Tensor]]:
    """Split the network's parameters into the copies a run steps and stores.

    Args:
        diffusion: the model whose parameters these are.
        full_fp16: whether this run keeps a float32 master copy.

    Returns:
        ``(model_params, master_params, step_params)``: the network's own
        parameters, the float32 master copy or None without one, and whichever
        of the two the optimiser and the gradient clip act on.
    """
    model_params: list[torch.Tensor] = list(diffusion.parameters())
    master_params = make_master_params(model_params) if full_fp16 else None
    step_params: list[torch.Tensor] = model_params
    if master_params is not None:
        step_params = [*master_params]
    return model_params, master_params, step_params


def restore_run(
    ckpt: dict[str, Any],
    *,
    resume: Path | None,
    diffusion: Diffusion,
    ema: EMA,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sched: LRScheduler,
    full_fp16: bool,
    say: Callable[[str], None],
    restore_rng: bool = True,
) -> tuple[int, float | None]:
    """Put a checkpoint's state back into a run that is ready to receive it.

    Called after the optimiser, schedule and scaler exist, and before the
    network is converted to float16: a resumed ``full_fp16`` run continues from
    the checkpoint's own float32 weights rather than from those weights rounded
    to half.

    Args:
        ckpt: the checkpoint, already read and checked for compatibility.
        resume: where it came from, for the message. None only reflects the
            caller's own type; a checkpoint always came from somewhere.
        diffusion: the model to restore into.
        ema: the shadow weights to restore.
        optim: the optimiser whose moments to restore, where they still fit.
        scaler: AMP gradient scaler to restore.
        sched: LR schedule to restore.
        full_fp16: whether *this* run keeps a float32 master copy. A checkpoint
            that disagrees cannot hand its optimiser moments over.
        say: where the resume notices go.
        restore_rng: whether to put the checkpoint's RNG state back. The state
            is the main rank's, so a distributed run passes False on every
            other rank and leaves those on their own per-rank streams; copying
            one rank's state to all of them would make every rank draw the
            same noise and timesteps for different images.

    Returns:
        ``(start_epoch, best_val)``: the epoch to resume at, and the lowest
        held-out loss seen so far, or None if the checkpoint records none.
    """
    was_full_fp16 = bool((ckpt.get("config") or {}).get("full_fp16", False))
    moments_fit = was_full_fp16 == full_fp16
    start_epoch = restore_checkpoint(
        ckpt,
        diffusion=diffusion,
        ema=ema,
        optim=optim if moments_fit else None,
        scaler=scaler,
        sched=sched,
    )
    if not moments_fit:
        say(
            f"this checkpoint was trained with full_fp16={was_full_fp16}, which lays the "
            f"optimiser state out differently; resuming with fresh AdamW moments"
        )
    # The schedule's step count is restored, but only the optimiser's own
    # state carries the learning rate it had reached. Without that state the
    # param groups still hold the schedule's step-0 value — zero under any
    # warmup — until the first ``sched.step()``, which comes after the first
    # optimiser step. Put the rate the schedule is at into the groups now.
    for param_group, lr in zip(optim.param_groups, sched.get_last_lr(), strict=True):
        param_group["lr"] = lr
    best_val: float | None = ckpt.get("best_val")
    replayed = restore_rng_state(ckpt) if restore_rng else True
    say(f"resumed from {resume}, {epochs_phrase(start_epoch)} already done")
    if not replayed:
        say("this checkpoint stores no RNG state, so the random stream restarts at the seed")
    return start_epoch, best_val
