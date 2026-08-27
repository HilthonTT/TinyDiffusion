"""The training loop for TinyDiffusion.

What a run trains on follows from its config's ``dataset``; see
:mod:`tinydiffusion.data.datasets` for the registry it names.

The pieces a run needs but a loop is not the natural home for live next door:
:mod:`~tinydiffusion.training.model` builds the process,
:mod:`~tinydiffusion.training.checkpoints` reads and writes its state, and
:mod:`~tinydiffusion.training.lr` is the LR schedule.
"""

import importlib.util
import shutil
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler
from torchvision.utils import save_image
from tqdm import tqdm

from tinydiffusion.data.datasets import denormalize, image_dataloader, set_loader_epoch
from tinydiffusion.diffusion.gaussian_diffusion import Diffusion
from tinydiffusion.diffusion.guidance import Conditioned, conditioned, drop_labels
from tinydiffusion.diffusion.samplers import get_sampler
from tinydiffusion.diffusion.timesteps import LossSecondMomentResampler
from tinydiffusion.models.unet import UNet
from tinydiffusion.training.checkpoints import (
    BEST_CHECKPOINT,
    INTERRUPTED_CHECKPOINT,
    LAST_CHECKPOINT,
    check_resume_compatible,
    read_checkpoint,
    restore_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.distributed import (
    Distributed,
    all_gather_cat,
    all_reduce_mean,
    all_reduce_sum,
    any_rank,
    barrier,
    broadcast_object,
)
from tinydiffusion.training.distributed import (
    setup as distributed_setup,
)
from tinydiffusion.training.distributed import (
    shutdown as distributed_shutdown,
)
from tinydiffusion.training.ema import EMA
from tinydiffusion.training.interrupt import InterruptChoice, interrupt_guard
from tinydiffusion.training.lr import lr_factor
from tinydiffusion.training.model import build_model
from tinydiffusion.training.observer import BatchProgress, TrainObserver, TrainPlan
from tinydiffusion.training.validation import validation_loss
from tinydiffusion.utils.device import (
    bf16_supported,
    describe_device,
    enable_tf32,
)
from tinydiffusion.utils.fp16 import (
    make_master_params,
    master_params_to_model_params,
    master_params_to_state_dict,
    model_grads_to_master_grads,
    model_params_to_master_params,
    unflatten_master_params,
    zero_grad,
)
from tinydiffusion.utils.seed import seed_everything
from tinydiffusion.utils.tracking import (
    LoggerBackend,
    RunLogger,
    quartile_means,
    timestep_quartile_totals,
)

__all__ = [
    "DRAIN_EVERY",
    "QUARTILE_EVERY",
    "epoch_seed",
    "reference_batch",
    "save_samples",
    "train",
    "validation_batches",
]


QUARTILE_EVERY = 8
"""Batches between timestep-quartile samples.

Bucketing the loss by timestep is a handful of extra kernels over a tensor the
loop already holds — cheap, but not free, and nothing reads the result until the
epoch ends. Every batch draws its timesteps independently, so one batch in eight
estimates the same four numbers at an eighth of the cost.

The totals are summed on the device across the whole epoch and read back once,
by :func:`~tinydiffusion.utils.tracking.quartile_means`, for the reason
:data:`DRAIN_EVERY` gives. Summing before dividing also makes each quartile's
figure a mean over the samples that landed in it, rather than an average of
per-batch means that counts a batch contributing two samples as heavily as one
contributing fifty.
"""

DRAIN_EVERY = 8
"""Batches between host reads of the per-batch metrics.

The loop hands the device work and moves on without waiting. Reading any value
back with ``.item()`` reverses that: it blocks the CPU until the queue drains,
so the loop stops queueing the next batch while the current one is still
running — the cost is not the copy but the pipeline bubble behind it.

Nothing needs those values *at* the batch that produced them. They are logged as
an epoch mean and displayed as a smoothed average, so they are buffered on the
device and fetched a run at a time by :func:`_drain_metrics`, which turns a
sync per batch into one per eight. The numbers are unchanged — the same values
in the same order, read later.

The progress bar's loss therefore updates every eighth batch rather than every
batch, which is the whole of the visible difference.
"""


def _silent(message: str) -> None:
    """Swallow a run's messages.

    What ``say`` becomes on the non-main ranks of a distributed run: the plan
    line, the resume notice and the new-best line are all worth printing once,
    and printing them once per GPU is how a four-way run turns its own output
    into noise.

    Args:
        message: the line that is not going anywhere.
    """


def _drain_metrics(
    pending: list[dict[str, torch.Tensor | float]],
    logger: RunLogger,
    loss_ema: float | None,
) -> float | None:
    """Read a run of buffered per-batch metrics back to the host, in one transfer.

    Every device tensor across every buffered batch is stacked and copied in a
    single operation, so the whole run costs one synchronisation rather than
    one per value. The batches are then replayed into the logger in the order
    they were produced, which is what keeps the smoothed loss identical to the
    one an unbuffered loop would have computed.

    Args:
        pending: buffered metrics, oldest first. Values may be device tensors
            or plain floats; the list is emptied.
        logger: where the resolved metrics are accumulated.
        loss_ema: the smoothed loss so far, or None before the first batch.

    Returns:
        The smoothed loss after replaying every buffered batch, or `loss_ema`
        unchanged if there was nothing buffered.
    """
    if not pending:
        return loss_ema

    tensors = [
        value for batch in pending for value in batch.values() if isinstance(value, torch.Tensor)
    ]
    # One stack, one copy. Built in the same order the substitution below walks,
    # so the values land back on the keys they came from.
    values = iter(torch.stack(tensors).tolist() if tensors else ())

    for batch in pending:
        resolved = {
            key: next(values) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        loss = resolved["train/loss"]
        loss_ema = loss if loss_ema is None else 0.9 * loss_ema + 0.1 * loss
        logger.accumulate(**resolved)

    pending.clear()
    return loss_ema


class _ObserverBackend:
    """Feeds an observer the epoch metrics through the ordinary backend fan-out.

    A :class:`~tinydiffusion.utils.tracking.LoggerBackend` already exists for
    exactly this shape of thing, so an observer is registered as one rather
    than given a second route to the same numbers.

    Args:
        observer: the watcher to forward to.
    """

    def __init__(self, observer: TrainObserver) -> None:
        self._observer = observer

    def write(self, metrics: Mapping[str, float], step: int) -> None:
        """Forward one epoch's metrics.

        Args:
            metrics: metric name to value.
            step: the epoch index.
        """
        self._observer.on_epoch(step, metrics)

    def close(self) -> None:
        """Nothing to release: the observer outlives the run."""


def _can_compile(device_type: str) -> bool:
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


def _share_timestep_sampler(diffusion: Diffusion, group: Distributed) -> None:
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


def epoch_seed(seed: int, epoch: int) -> int:
    """Seed for one epoch's shuffle order.

    A function of ``(seed, epoch)`` alone, deliberately: seeding the loader
    once at startup makes the order depend on how many epochs have already run
    in *this process*, so a run resumed at epoch 5 replays the ordering a fresh
    run used for epoch 0, and every later epoch follows suit. Deriving it here
    means epoch 5 draws epoch 5's batches whether it was reached by resuming or
    by training straight through.

    Args:
        seed: the run's seed.
        epoch: zero-based epoch index.

    Returns:
        A seed inside the range ``torch.Generator.manual_seed`` accepts.
    """
    # An odd multiplier so consecutive epochs land far apart in the stream,
    # masked to 63 bits because manual_seed rejects anything wider.
    return (seed * 1_000_003 + epoch) & ((1 << 63) - 1)


def _epochs(count: int) -> str:
    """Render an epoch count, pluralised.

    Args:
        count: number of epochs.

    Returns:
        e.g. ``"1 epoch"`` or ``"30 epochs"``.
    """
    return f"{count} epoch" if count == 1 else f"{count} epochs"


def validation_batches(cfg: TrainConfig) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialise the held-out slice the run is scored on each epoch.

    Read once and kept in host memory rather than reloaded per epoch: the slice
    is small, and it has to be *the same images every time* for the epoch-to-
    epoch comparison — and so for ``best.pt`` — to mean anything.

    Args:
        cfg: run configuration. ``val_batches`` bounds the slice; 0 takes the
            whole test split.

    Returns:
        ``(images, labels)`` pairs on the CPU, empty if ``val_every`` is off.
    """
    if cfg.val_every <= 0:
        return []

    loader = image_dataloader(
        cfg.dataset_spec(),
        cfg.data_root,
        batch_size=cfg.batch_size,
        train=False,
        image_size=cfg.image_size,
        # Read once, so worker processes would cost more to spawn than they save.
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, (x, y) in enumerate(loader):
        if cfg.val_batches and index >= cfg.val_batches:
            break
        batches.append((x, y))
    return batches


def reference_batch(cfg: TrainConfig) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Materialise the real strip every sample grid is compared against.

    Read from the front of the *unshuffled* training split rather than taken
    from whichever batch the loop happens to see first. The shuffle order is a
    function of the epoch index, so a batch picked off the loop would differ
    between a straight run and a ``--resume`` — and for a conditional run those
    labels are what the generated half is drawn from, so the grids would stop
    being a flipbook of one set of images at exactly the point the run was
    picked up again. This depends on the dataset alone, which is the same
    property :data:`~tinydiffusion.training.train.train`'s fixed ``x_T`` has.

    Unaugmented, for the same reason it is unshuffled: a flip that lands
    differently per read would move the real strip on its own.

    Args:
        cfg: run configuration. ``num_samples`` bounds the strip, and
            ``sample_every`` of 0 skips the read entirely.

    Returns:
        ``(images, labels)`` on the CPU. Labels are None for an unconditional
        run, and both are None when no grid will ever be drawn.
    """
    if cfg.sample_every <= 0:
        return None, None

    loader = image_dataloader(
        cfg.dataset_spec(),
        cfg.data_root,
        batch_size=cfg.num_samples,
        train=True,
        image_size=cfg.image_size,
        # One batch, so worker processes would cost more to spawn than they save.
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )
    for x, y in loader:
        return x[: cfg.num_samples], y[: cfg.num_samples] if cfg.num_classes is not None else None
    return None, None


def _snapshot_epoch(ckpt_dir: Path, source: Path, *, epoch: int, keep: int) -> None:
    """Keep a numbered copy of this epoch's checkpoint, pruning old ones.

    A copy of the file just written rather than a second :func:`save_checkpoint`
    call: the bytes are identical by construction, and serialising a U-Net
    twice per epoch is not free.

    Args:
        ckpt_dir: directory holding the run's checkpoints.
        source: the checkpoint just written, copied under an epoch name.
        epoch: zero-based epoch index. The filename is one-based.
        keep: how many snapshots to retain, newest first.
    """
    if keep <= 0 or not source.is_file():
        return
    shutil.copy2(source, ckpt_dir / f"epoch_{epoch + 1:04d}.pt")
    # Zero-padded, so lexical order is epoch order.
    snapshots = sorted(ckpt_dir.glob("epoch_*.pt"))
    for stale in snapshots[: max(len(snapshots) - keep, 0)]:
        stale.unlink(missing_ok=True)


@torch.no_grad()
def save_samples(
    diffusion: Diffusion,
    ema: EMA,
    real: torch.Tensor,
    cfg: TrainConfig,
    epoch: int,
    labels: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> Path:
    """Render a grid of EMA samples above a strip of real images.

    Putting real data in the same grid makes contrast and stroke weight
    directly comparable, which is what tells you whether the model has
    learned the data distribution or merely something digit-shaped.

    A conditional run generates on the *real* strip's own labels, so the two
    halves line up column by column and the comparison becomes per class: a
    generated 4 sits directly above a real 4.

    Args:
        diffusion: the diffusion model, used for its schedule.
        ema: EMA weights to sample from.
        real: a batch of real images in [-1, 1] to show alongside.
        cfg: run configuration.
        epoch: zero-based epoch index. The filename is one-based, matching the
            progress bar.
        labels: the real strip's class labels, or None when unconditional.
        noise: the starting x_T to redraw from, or None for a fresh draw. Held
            fixed across a run, it makes the grids a flipbook of one set of
            images sharpening rather than an unrelated sample each epoch.

    Returns:
        The path written, so a caller watching the run can pick the grid up
        without reconstructing the filename.
    """
    shape = (cfg.dataset_spec().channels, cfg.image_size, cfg.image_size)
    if labels is not None:
        # A batch smaller than num_samples leaves the strip short; repeat it so
        # there is a label per generated image either way.
        index = torch.arange(cfg.num_samples) % labels.shape[0]
        labels = labels[index].to(cfg.device)

    fake = get_sampler(cfg.sampler)(
        diffusion,
        cfg.num_samples,
        shape,
        cfg.device,
        num_steps=cfg.sample_steps,
        eta=0.0,
        model=conditioned(
            ema.module,
            labels,
            num_classes=cfg.num_classes,
            scale=cfg.guidance,
            rescale=cfg.guidance_rescale,
        ),
        noise=noise,
        spacing=cfg.sample_spacing,
    )
    reference = real[: cfg.num_samples].to(cfg.device)
    grid = torch.cat([denormalize(fake), denormalize(reference)], dim=0)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.out_dir / f"sample_{epoch + 1:04d}.png"
    save_image(grid, path, nrow=min(8, cfg.num_samples))
    return path


def _model_state(
    diffusion: Diffusion, master_params: list[nn.Parameter] | None
) -> dict[str, torch.Tensor] | None:
    """The weights to checkpoint, in float32 whichever precision the run uses.

    Args:
        diffusion: the diffusion model.
        master_params: the run's float32 master copy, or None when the network
            holds its own weights at full precision.

    Returns:
        A float32 state dict when there is a master copy to read, and None to
        leave :func:`~tinydiffusion.training.checkpoints.save_checkpoint` with
        its default of the network's own.
    """
    if master_params is None:
        return None
    return master_params_to_state_dict(diffusion.net, master_params)


def _save_and_report(
    cfg: TrainConfig,
    *,
    epoch: int,
    diffusion: Diffusion,
    ema: EMA,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sched: LRScheduler | None = None,
    best_val: float | None = None,
    model_state: dict[str, torch.Tensor] | None = None,
    say: Callable[[str], None] = print,
) -> None:
    """Checkpoint a cancelled run and tell the user how to pick it up again.

    Written to :data:`INTERRUPTED_CHECKPOINT`, never over ``last.pt``; see that
    constant for why.

    Args:
        cfg: run configuration.
        epoch: index of the last fully completed epoch, or -1 if none.
        diffusion: the diffusion model.
        ema: exponential moving average of the network weights.
        optim: optimiser whose moments should be preserved.
        scaler: AMP gradient scaler.
        sched: LR schedule to restore, or None to skip.
        best_val: lowest held-out loss seen so far, carried through so a resume
            keeps comparing against it.
        model_state: float32 weights to write in place of the network's own;
            see :func:`~tinydiffusion.training.checkpoints.save_checkpoint`.
        say: where the two lines go. Defaults to printing them.
    """
    path = cfg.ckpt_dir / INTERRUPTED_CHECKPOINT
    save_checkpoint(
        path,
        epoch=epoch,
        diffusion=diffusion,
        ema=ema,
        optim=optim,
        scaler=scaler,
        sched=sched,
        cfg=cfg,
        best_val=best_val,
        model_state=model_state,
    )
    done = max(epoch + 1, 0)
    say(f"saved {path} ({_epochs(done)} complete, plus a partial epoch)")
    say(f"resume with: tinydiffusion train --resume {path}")


def train(
    cfg: TrainConfig | None = None,
    resume: Path | None = None,
    observer: TrainObserver | None = None,
) -> Diffusion:
    """Train a diffusion model on the dataset the config names.

    Which process is trained follows from the config too; see
    :func:`~tinydiffusion.training.model.build_model`.

    Ctrl+C does not tear the run down: it is caught at the next batch boundary
    and turned into a confirmation prompt, with the option to checkpoint first
    so ``--resume`` can pick the run up later. That save goes to
    :data:`INTERRUPTED_CHECKPOINT`, leaving ``last.pt`` as the newest *complete*
    epoch.

    Each epoch ends by scoring a fixed held-out slice, and the best-scoring
    epoch is kept as :data:`BEST_CHECKPOINT`. Diffusion runs do not improve
    monotonically, so the last epoch is not reliably the one worth sampling.

    Args:
        cfg: run configuration. Defaults are used when omitted.
        resume: checkpoint to continue from, or None to start fresh.
        observer: something watching the run from outside the loop; see
            :mod:`tinydiffusion.training.observer`. Passing one redirects every
            line this would have printed, replaces the tqdm bar with
            :meth:`~tinydiffusion.training.observer.TrainObserver.on_batch`,
            and lets the watcher stop the run without the Ctrl+C prompt — which
            has nobody to answer it when the terminal belongs to a display.
            None leaves all three exactly as they were.

    Returns:
        The trained model, with EMA weights already swapped in. A run cancelled
        part way through returns the model as it stood at that point.

    Raises:
        ValueError: if `resume` names a checkpoint trained with a different
            model than `cfg` describes.
    """
    cfg = cfg or TrainConfig()
    # Joins the process group when a launcher started one, and is a no-op that
    # resolves the device exactly as before when it did not. Everything below
    # reads `group` rather than asking whether it is distributed: on a single
    # process it is rank 0 of 1, every `is_main` guard is true, and every
    # collective returns its argument untouched.
    group, device = distributed_setup(cfg.device)
    # Every message below goes through this rather than to stdout directly, so
    # a watcher can take them without the loop caring where they end up — and
    # so the non-main ranks of a distributed run say nothing at all, rather
    # than printing the same plan line once per GPU.
    if observer is not None:
        say: Callable[[str], None] = observer.on_message
    else:
        say = print if group.is_main else _silent
    cfg = replace(cfg, device=device)
    spec = cfg.dataset_spec()
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    device_type = torch.device(cfg.device).type
    # Two half-precision strategies, and at most one of them runs. full_fp16
    # puts float16 in the weights themselves and keeps a float32 master copy
    # for the optimiser; autocast leaves the weights alone and casts per
    # kernel. Both want CUDA — half precision on a CPU is emulated, so it is
    # slower than float32 rather than faster.
    use_full_fp16 = cfg.full_fp16 and device_type == "cuda"
    if cfg.full_fp16 and not use_full_fp16:
        say("full_fp16 needs a CUDA device; training in float32 instead")
    use_amp = cfg.amp and device_type == "cuda" and not use_full_fp16
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    if use_amp and amp_dtype is torch.bfloat16 and not bf16_supported():
        # Pre-Ampere. Saying so beats an unexplained slowdown from a dtype the
        # hardware only emulates — and emulated is exactly what torch reports
        # as supported, which is why this asks `bf16_supported` rather than
        # torch directly.
        say("this GPU emulates bfloat16 rather than running it, falling back to fp16")
        amp_dtype = torch.float16
    if device_type == "cuda":
        # Input shapes are fixed for the whole run, so autotuning pays off once
        # — but the autotuner picks whichever kernel wins on the day, so it is
        # exactly what `deterministic` is asking us not to do. seed_everything
        # has already cleared the flag; setting it back here unconditionally
        # would quietly undo the setting that was just applied.
        torch.backends.cudnn.benchmark = not cfg.deterministic
        enable_tf32()

    diffusion = build_model(cfg).to(cfg.device)
    # build_model has no idea whether it is being called inside a group, so the
    # one part of the process that wants to know is told here.
    _share_timestep_sampler(diffusion, group)
    if group.enabled:
        # Deliberately *after* the network is built, and only after. Every rank
        # has to start from identical weights — DDP asserts on it — which is
        # what the shared cfg.seed above gives, so this cannot move earlier.
        #
        # From here on the ranks want to differ: the timesteps and the noise
        # for each batch come off the global RNG, and four ranks drawing the
        # same t for their own shard would turn a four-way run into a noisier
        # estimate of the same gradient rather than a four-times-larger batch.
        torch.manual_seed(cfg.seed + group.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed + group.rank)
    if cfg.channels_last:
        # Before the EMA is taken, so the shadow weights carry the same layout
        # and the copy back at the end of the run does not have to convert.
        # `memory_format` is absent from the shipped `Module.to` overloads.
        diffusion.net.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    ema = EMA(diffusion.net, decay=cfg.ema_decay, warmup=cfg.ema_warmup)

    # Compiled for the training step only, and only as a wrapper: it shares its
    # parameters with the eager module, which is what the checkpoint, the EMA
    # and every sampler go on using. A compiled run therefore writes ordinary
    # checkpoints, rather than ones whose keys all carry a `_orig_mod.` prefix.
    train_net: nn.Module = diffusion.net
    # The same trick, and for the same reason: DDP wraps the network and shares
    # its parameters, so the checkpoint, the EMA and every sampler go on using
    # the eager `diffusion.net` and never see a `module.` prefix in a key.
    #
    # Kept as its own name as well as in `train_net` because `no_sync` below is
    # a DDP method, and compiling would put a wrapper in front of it.
    ddp: DistributedDataParallel | None = None
    if group.enabled:
        ddp = DistributedDataParallel(
            diffusion.net,
            # None on CPU/gloo, where there is no device index to name.
            device_ids=[group.local_rank] if device_type == "cuda" else None,
            output_device=group.local_rank if device_type == "cuda" else None,
            # Every parameter of the U-Net takes part in every forward pass, so
            # the graph traversal that finding unused ones costs is pure
            # overhead — and leaving it off turns a future architecture change
            # that *does* skip a parameter into a loud error rather than a
            # silent hang.
            find_unused_parameters=False,
            # Point each .grad at its slot in the reduction bucket instead of
            # keeping a second copy of it. Saves one model's worth of gradient
            # memory per rank, which is what buys back the headroom DDP's
            # buckets cost in the first place.
            gradient_as_bucket_view=True,
        )
        train_net = ddp
    if cfg.compile and _can_compile(device_type):
        # torch.compile is annotated as returning a bare callable; what it
        # returns is a Module wrapping — and sharing the parameters of — the
        # one it was given.
        #
        # Compiles whatever `train_net` already is, so under a group that is
        # the DDP wrapper rather than the bare network. That order is the
        # supported one: Dynamo recognises DDP and splits the graph at the
        # bucket boundaries so the all-reduces still overlap the backward pass,
        # which compiling the inner module would have hidden from it.
        train_net = cast(nn.Module, torch.compile(train_net))
    elif cfg.compile:
        say(
            "compile is on but Triton is not installed, so the CUDA backend cannot build "
            "kernels; training eagerly instead (pip install triton-windows on Windows)"
        )

    # Held as a list rather than re-walked: order is what every fp16 helper
    # below indexes by, and a generator would be exhausted after one of them.
    model_params: list[torch.Tensor] = list(diffusion.parameters())
    # The float16 network is not what gets optimised — see
    # :mod:`tinydiffusion.utils.fp16` for why an Adam step applied to a
    # float16 weight rounds away to nothing. Built while the weights are still
    # float32, so the copy is exact, and converted below once the checkpoint
    # (if any) has been restored into it.
    master_params = make_master_params(model_params) if use_full_fp16 else None
    # What the optimiser and the gradient clip act on. Spelled over two
    # statements because `list` is invariant: a Parameter is a Tensor, but a
    # list of them is not a list of Tensors until the annotation says so.
    step_params: list[torch.Tensor] = model_params
    if master_params is not None:
        step_params = [*master_params]

    # AdamW rather than Adam, and identical to it at the default
    # weight_decay=0: decoupled decay is what the two differ in.
    optim = torch.optim.AdamW(
        step_params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )

    # Read before the resume so a mismatch names the setting that changed
    # instead of listing every tensor that no longer fits — and before the
    # dataset is touched, so it fails without waiting on a download.
    ckpt: dict[str, Any] | None = None
    if resume is not None:
        ckpt = read_checkpoint(resume, device=cfg.device)
        check_resume_compatible(ckpt, cfg, path=resume)

    # Re-seeded per epoch below, so the batch order is a function of the epoch
    # index rather than of how many epochs this process has already run.
    loader_rng = torch.Generator()
    loader = image_dataloader(
        spec,
        cfg.data_root,
        batch_size=cfg.batch_size,
        train=True,
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
        # Only the training split is augmented, and only where the spec says a
        # flip preserves the label.
        augment=True,
        generator=loader_rng,
        # None outside a group, which leaves the loader exactly as it was.
        # Inside one, this is what makes the run data-parallel: each rank draws
        # a disjoint shard, so an epoch is still one pass over the dataset and
        # not `world_size` passes over all of it.
        num_replicas=group.world_size if group.enabled else None,
        rank=group.rank if group.enabled else None,
        seed=cfg.seed,
    )

    # Optimiser steps, not batches: an accumulated group is one step, and the
    # ragged group a non-dividing accumulation leaves still steps once.
    steps_per_epoch = -(-len(loader) // cfg.grad_accum)
    # Stepped once per *applied* optimiser step below, so the ramp counts real
    # updates rather than batches AMP threw away.
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: lr_factor(
            step,
            warmup=cfg.lr_warmup,
            total=steps_per_epoch * cfg.num_epochs,
            schedule=cfg.lr_schedule,
        ),
    )
    scaler = torch.amp.GradScaler(
        device_type,
        # bf16 carries fp32's exponent range, so there is nothing to scale
        # against and no step to skip when the scale would have overflowed.
        # full_fp16 is the opposite case and needs it most: with the weights
        # themselves in half precision there is no unscaled path at all, and
        # diffusion gradients sit close enough to float16's floor that an
        # unscaled backward pass flushes a good share of them to zero.
        enabled=use_full_fp16 or (use_amp and amp_dtype is torch.float16),
    )

    start_epoch = 0
    best_val: float | None = None
    if ckpt is not None:
        # AdamW's moments are stored per parameter tensor, and full_fp16 gives
        # it one flattened tensor where every other mode gives it a few hundred
        # — so the moments cannot cross that boundary, however well the weights
        # do. Dropping them and saying so beats both a raw size-mismatch dump
        # and refusing a resume whose weights are perfectly good.
        was_full_fp16 = bool((ckpt.get("config") or {}).get("full_fp16", False))
        moments_fit = was_full_fp16 == use_full_fp16
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
        best_val = ckpt.get("best_val")
        # After seed_everything above, so the stored stream wins over the one
        # cfg.seed set up: the point is for a resumed epoch to draw the noise
        # that epoch would have drawn, not the noise epoch 0 draws.
        replayed = restore_rng_state(ckpt)
        say(f"resumed from {resume}, {_epochs(start_epoch)} already done")
        if not replayed:
            say("this checkpoint stores no RNG state, so the random stream restarts at the seed")

    if master_params is not None:
        # Both halves matter, and in this order. The master copy picks up
        # whatever the restore just wrote — which is still float32 at this
        # point, so a resumed run continues from the checkpoint's own weights
        # rather than from the checkpoint rounded to half. Only then does the
        # network go to float16.
        model_params_to_master_params(model_params, master_params)
        # build_model always builds a UNet, and it is the only thing that knows
        # which of its parts can safely hold half precision.
        cast(UNet, diffusion.net).convert_to_fp16()

    held_out = validation_batches(cfg)

    n_params = sum(p.numel() for p in diffusion.net.parameters())
    remaining = cfg.num_epochs - start_epoch
    if start_epoch == 0:
        plan = _epochs(cfg.num_epochs)
    elif remaining > 0:
        plan = f"epochs {start_epoch + 1}-{cfg.num_epochs} ({remaining} to go)"
    else:
        # A checkpoint past num_epochs would otherwise render as "epochs 4-2".
        plan = f"nothing to run (checkpoint is at {_epochs(start_epoch)})"
    conditioning = (
        f"{cfg.num_classes} classes, {cfg.class_dropout:g} label dropout"
        if cfg.num_classes is not None
        else "unconditional"
    )
    scoring = (
        f"{sum(x.shape[0] for x, _ in held_out)} held-out images every {_epochs(cfg.val_every)}"
        if held_out
        else "no validation"
    )
    if use_full_fp16:
        precision = "fp16 weights (float32 master)"
    elif use_amp:
        # The resolved dtype, not the requested one: a pre-Ampere card asks for
        # bf16 and is quietly given fp16 above, and the plan line is where that
        # would otherwise go unmentioned.
        precision = f"amp {'bf16' if amp_dtype is torch.bfloat16 else 'fp16'}"
    else:
        precision = "amp off"
    if cfg.compile:
        precision += " | compiled"
    if cfg.channels_last:
        precision += " | channels_last"
    steps = f"{steps_per_epoch} steps/epoch"
    # What one optimiser step actually averages over, which under a group is
    # neither the config's batch_size nor anything the loop prints elsewhere:
    # every rank contributes its own batch to the same all-reduced gradient.
    effective_batch = cfg.batch_size * cfg.grad_accum * group.world_size
    if effective_batch != cfg.batch_size:
        # Named separately because they are different mechanisms reaching the
        # same number: accumulation trades steps for batch on one device, the
        # world size adds devices. A run using only the second would otherwise
        # report "x1 accumulated", which is true and misleading.
        how = []
        if cfg.grad_accum > 1:
            how.append(f"x{cfg.grad_accum} accumulated")
        if group.world_size > 1:
            how.append(f"x{group.world_size} ranks")
        steps += f" ({', '.join(how)}, {effective_batch} effective)"
    parallelism = f" | {group}" if group.enabled else ""
    say(
        f"{n_params / 1e6:.2f}M parameters | {spec.name} {cfg.image_size}px x{spec.channels} | "
        f"device {describe_device(cfg.device)} | {precision} | {conditioning} | {plan} | "
        f"{steps} | {scoring}{parallelism}"
    )
    if observer is not None:
        # The same facts the line above just said, as data. A display should
        # not have to parse the sentence back apart to fill in a header.
        observer.on_plan(
            TrainPlan(
                dataset=spec.name,
                image_size=cfg.image_size,
                channels=spec.channels,
                device=cfg.device,
                device_description=describe_device(cfg.device),
                parameters=n_params,
                precision=precision,
                num_classes=cfg.num_classes,
                start_epoch=start_epoch,
                num_epochs=cfg.num_epochs,
                steps_per_epoch=steps_per_epoch,
                batch_size=cfg.batch_size,
                grad_accum=cfg.grad_accum,
                validation_images=sum(x.shape[0] for x, _ in held_out),
                log_dir=cfg.log_dir,
            )
        )
    if remaining <= 0:
        say(f"nothing to do: the checkpoint already covers all {_epochs(cfg.num_epochs)}")

    # Fixed real images, read from the front of the split rather than lifted off
    # the loop: the batch order depends on the epoch, so a resumed run would
    # otherwise compare against — and, when conditional, generate on the labels
    # of — a different set of images than the epochs before it. Kept on the CPU
    # so the sample grid does not pin a batch in VRAM.
    reference, reference_labels = reference_batch(cfg)

    # Fixed latents, so each epoch's grid redraws the same x_T and the sequence of
    # PNGs reads as one set of digits sharpening rather than a fresh draw each time.
    # Seeded off cfg.seed rather than the live RNG, so a --resume continues the
    # same grid instead of starting a new one.
    sample_noise = torch.randn(
        cfg.num_samples,
        spec.channels,
        cfg.image_size,
        cfg.image_size,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    cancelled = False

    extra: list[LoggerBackend] = [] if observer is None else [_ObserverBackend(observer)]
    # Rank 0 keeps the record for the whole group. The metrics it writes are
    # already the group's — the losses below are all-reduced before they reach
    # the logger — so a second rank appending to metrics.jsonl would not add
    # information, only interleave with the first and corrupt the file.
    logger = RunLogger.for_run(
        cfg.log_dir,
        console=cfg.log_console and group.is_main,
        jsonl=cfg.log_jsonl and group.is_main,
        tensorboard=cfg.tensorboard and group.is_main,
        wandb=cfg.wandb and group.is_main,
        wandb_project=cfg.wandb_project,
        # The config as written, so the W&B sweep view can group and filter by
        # any field a run was launched with.
        wandb_config=asdict(cfg),
        extra=extra,
    )

    with logger, interrupt_guard() as interrupts:
        for epoch in range(start_epoch, cfg.num_epochs):
            loader_rng.manual_seed(epoch_seed(cfg.seed, epoch))
            # The sharded loader's own version of the line above: its sampler
            # shuffles from (seed, epoch) and has to be told the epoch changed,
            # or every rank re-draws the shard it saw last time. A no-op when
            # the loader is not sharded.
            set_loader_epoch(loader, epoch)
            diffusion.train()
            loss_ema: float | None = None
            epoch_start = time.perf_counter()
            images = 0
            # Metrics for batches whose values are still on the device. Drained
            # a run at a time; see DRAIN_EVERY.
            pending: list[dict[str, torch.Tensor | float]] = []
            # Running quartile totals for the epoch, summed on the device and
            # read back once after the loop; see QUARTILE_EVERY.
            quartile_sums = torch.zeros(4, device=cfg.device)
            quartile_counts = torch.zeros(4, device=cfg.device)

            # A watcher owns the terminal, so the bar would be drawn into the
            # middle of it. on_batch below carries the same information.
            with tqdm(
                loader,
                desc=f"epoch {epoch + 1}/{cfg.num_epochs}",
                # One bar for the group, not one per GPU redrawing over the
                # others. The ranks run in lockstep anyway, so rank 0's bar is
                # an accurate picture of all of them.
                disable=observer is not None or not group.is_main,
            ) as pbar:
                for batch, (x, y) in enumerate(pbar):
                    x = x.to(cfg.device, non_blocking=True)
                    if cfg.channels_last:
                        x = x.contiguous(memory_format=torch.channels_last)

                    model: nn.Module = train_net
                    if cfg.num_classes is not None:
                        # Dropping a fraction of the labels to the null token is
                        # the only thing training does differently: it is what
                        # teaches the one network the unconditional prediction
                        # that guidance extrapolates away from at sample time.
                        labels = drop_labels(
                            y.to(cfg.device, non_blocking=True),
                            cfg.num_classes,
                            cfg.class_dropout,
                        )
                        model = Conditioned(train_net, labels)

                    # Micro-batches are summed into one update. The group is
                    # divided by its own length rather than by grad_accum, so
                    # the ragged group a non-dividing loader leaves is still an
                    # average over the batches it actually holds.
                    group_start = batch - batch % cfg.grad_accum
                    group_size = min(cfg.grad_accum, len(loader) - group_start)
                    applies = batch + 1 == group_start + group_size

                    if batch == group_start:
                        optim.zero_grad(set_to_none=True)
                        if master_params is not None:
                            # Accumulation happens on the network's own float16
                            # gradients, which the optimiser never sees, so
                            # clearing its master gradient is not enough.
                            zero_grad(model_params)
                    with torch.amp.autocast(
                        device_type, dtype=amp_dtype if use_amp else None, enabled=use_amp
                    ):
                        terms = diffusion.loss_terms(x, model=model)
                    loss = terms.loss

                    # DDP all-reduces the gradients *during* the backward pass,
                    # which is what overlaps the communication with the compute
                    # — but it means an accumulated group would pay for
                    # `grad_accum` all-reduces to produce one update. no_sync
                    # suppresses it on every micro-batch except the one that
                    # applies, whose backward then reduces the whole
                    # accumulated gradient in a single pass.
                    sync: AbstractContextManager[None] = nullcontext()
                    if ddp is not None and not applies:
                        sync = ddp.no_sync()
                    with sync:
                        # torch ships `Tensor.backward` unannotated, so now that
                        # the loss is a real Tensor rather than the Any that
                        # `nn.Module.__call__` returns, strict mypy calls this
                        # an untyped call.
                        scaler.scale(loss / group_size).backward()  # type: ignore[no-untyped-call]

                    # Left as a device tensor rather than read here: it is
                    # only ever logged, so it rides the same deferred transfer
                    # the loss does.
                    grad_norm: torch.Tensor | None = None
                    stepped = True
                    if applies:
                        if master_params is not None:
                            # The optimiser steps the master copy, so this is
                            # the last moment the gradients exist anywhere it
                            # can reach them — and it has to be before the
                            # unscale below, which only touches what the
                            # optimiser holds.
                            model_grads_to_master_grads(model_params, master_params)
                        if cfg.grad_clip > 0:
                            # Unscale first, or the clip threshold is applied to scaled grads.
                            scaler.unscale_(optim)
                            # The pre-clip norm comes back for free; it is the first
                            # thing to look at when a loss curve goes flat or spikes.
                            grad_norm = nn.utils.clip_grad_norm_(step_params, cfg.grad_clip)

                        scale_before = scaler.get_scale()
                        scaler.step(optim)
                        scaler.update()
                        # A shrinking scale means inf/NaN grads and a skipped optimiser
                        # step; folding the unchanged weights in would still burn a step
                        # of the EMA warmup.
                        stepped = scaler.get_scale() >= scale_before
                        if stepped:
                            if master_params is not None:
                                # Only on a step that landed: a skipped one
                                # leaves the master copy alone, so the network
                                # already agrees with it.
                                master_params_to_model_params(model_params, master_params)
                                # Averaged from the master copy rather than the
                                # network — at a decay of 0.9999 the increment
                                # is far below what float16 can represent, and
                                # an EMA fed half-precision weights would stop
                                # moving. Buffers still come from the module.
                                ema.update(
                                    diffusion.net,
                                    unflatten_master_params(model_params, master_params),
                                )
                            else:
                                ema.update(diffusion.net)
                            sched.step()

                    images += x.shape[0]
                    # Detached and normalised to float32 so the whole buffer
                    # stacks: under autocast the loss can come back as float16
                    # while the gradient norm, taken on float32 parameters,
                    # does not.
                    loss_metric = loss.detach().float()
                    if group.enabled:
                        # So the logged loss is the whole global batch's rather
                        # than this rank's shard of it. Cloned first: the
                        # reduction is in place, and `.float()` on a tensor
                        # that is already float32 hands back the same storage
                        # the backward pass just read.
                        #
                        # This is a device-side collective, not a host read —
                        # it queues behind the batch like everything else, so
                        # it does not undo what DRAIN_EVERY is buying.
                        loss_metric = all_reduce_mean(loss_metric.clone(), group)
                    batch_metrics: dict[str, torch.Tensor | float] = {"train/loss": loss_metric}
                    if applies:
                        # Only meaningful where a step was attempted; recording
                        # them per micro-batch would dilute the rate by
                        # grad_accum and log a stale norm alongside it.
                        batch_metrics["train/skipped_step"] = float(not stepped)
                    if grad_norm is not None:
                        batch_metrics["train/grad_norm"] = grad_norm.detach().float()
                    pending.append(batch_metrics)

                    if batch % QUARTILE_EVERY == 0:
                        # Added into the epoch's running totals rather than
                        # averaged here: reading four bucket means off the
                        # device per sampled batch is the same synchronisation
                        # the buffer above exists to avoid.
                        batch_sums, batch_counts = timestep_quartile_totals(
                            terms.per_sample.float(), terms.timesteps, cfg.num_timesteps
                        )
                        quartile_sums += batch_sums
                        quartile_counts += batch_counts

                    if len(pending) >= DRAIN_EVERY:
                        loss_ema = _drain_metrics(pending, logger, loss_ema)
                        if loss_ema is not None:
                            pbar.set_postfix(loss=f"{loss_ema:.4f}")
                        if observer is not None:
                            # Here rather than per batch, and deliberately: the
                            # numbers only exist on the host once the drain has
                            # read them back.
                            observer.on_batch(
                                BatchProgress(
                                    epoch=epoch,
                                    num_epochs=cfg.num_epochs,
                                    batch=batch,
                                    num_batches=len(loader),
                                    loss=loss_ema,
                                    images=images,
                                    seconds=time.perf_counter() - epoch_start,
                                )
                            )

                    # A watcher that has asked to stop has already decided, so
                    # it is not asked again: the prompt below has nobody to
                    # answer it when a display owns the terminal. Stopping and
                    # saving is what an unattended Ctrl+C already resolves to.
                    if observer is not None and observer.stop_requested():
                        _save_and_report(
                            cfg,
                            epoch=epoch - 1,
                            diffusion=diffusion,
                            ema=ema,
                            optim=optim,
                            scaler=scaler,
                            best_val=best_val,
                            sched=sched,
                            model_state=_model_state(diffusion, master_params),
                            say=say,
                        )
                        cancelled = True
                        break

                    # Under a group the flag is not this rank's to act on. The
                    # launcher delivers the signal to each process
                    # independently, so rank 0 can see it a batch or two before
                    # rank 3 does — and a rank that left the loop early strands
                    # the others at the next gradient all-reduce until the
                    # timeout expires. Both collectives below are therefore
                    # asked at a batch index every rank agrees on rather than
                    # wherever the signal happened to land.
                    interrupted = interrupts.requested
                    if group.enabled:
                        aligned = batch % DRAIN_EVERY == 0
                        interrupted = (
                            any_rank(interrupted, group, device=cfg.device) if aligned else False
                        )

                    if interrupted:
                        # Batch boundary: model, optimiser and EMA all agree, so
                        # a checkpoint written here resumes cleanly.
                        #
                        # Only rank 0 has a terminal to prompt in — the others
                        # would read EOF and resolve to the unattended default
                        # on their own — so it decides for the group and says
                        # what it decided.
                        choice: InterruptChoice | None = None
                        if group.is_main:
                            with tqdm.external_write_mode():
                                choice = interrupts.resolve()
                        # broadcast_object is typed Any in and Any out, since
                        # what crosses it is whatever the caller sent.
                        choice = cast(InterruptChoice, broadcast_object(choice, group))
                        if not choice.stop:
                            continue
                        if choice.save:
                            # The last *completed* epoch is the one before this
                            # partial one, so resuming replays it in full.
                            if group.is_main:
                                _save_and_report(
                                    cfg,
                                    epoch=epoch - 1,
                                    diffusion=diffusion,
                                    ema=ema,
                                    optim=optim,
                                    scaler=scaler,
                                    best_val=best_val,
                                    sched=sched,
                                    model_state=_model_state(diffusion, master_params),
                                    say=say,
                                )
                        else:
                            say("cancelled without saving")
                        cancelled = True
                        break

            # Covers both a completed epoch and the partial one a Ctrl+C ends
            # on, so the last few batches reach the flush below either way.
            loss_ema = _drain_metrics(pending, logger, loss_ema)

            elapsed = time.perf_counter() - epoch_start
            if group.enabled:
                # Each rank bucketed only its own shard, so rank 0's totals
                # describe an eighth of the epoch. Summed rather than averaged
                # — quartile_means divides by the counts, and both sides of
                # that division have to cover the same images. One collective
                # per epoch, against one per batch had the buckets been
                # reduced where they were filled.
                #
                # torch.distributed has no all-reduce that takes two tensors,
                # and stacking them to save a call would cost the copy it
                # saved, so this is two.
                all_reduce_sum(quartile_sums, group)
                all_reduce_sum(quartile_counts, group)
            # Set rather than accumulated: the mean over the epoch has already
            # been formed from the totals, and this is the one read back.
            logger.set(
                **{
                    f"train/{name}": value
                    for name, value in quartile_means(quartile_sums, quartile_counts).items()
                }
            )
            logger.set(
                **{
                    "train/lr": float(optim.param_groups[0]["lr"]),
                    "train/ema_decay": ema.current_decay,
                    # Asked of the scaler rather than of the config: it is
                    # disabled under bf16 and off CUDA, and enabled for
                    # full_fp16, none of which `amp` alone distinguishes.
                    "train/amp_scale": float(scaler.get_scale()) if scaler.is_enabled() else 1.0,
                    "time/epoch_seconds": elapsed,
                    # Scaled to the whole group rather than reduced: the ranks
                    # run the same number of batches at the same batch size,
                    # and this is the number worth comparing between a one-GPU
                    # run and a four-GPU one.
                    "time/images_per_second": (
                        images * group.world_size / elapsed if elapsed > 0 else 0.0
                    ),
                }
            )
            new_best: float | None = None
            if held_out and not cancelled and (epoch + 1) % cfg.val_every == 0:
                # The EMA weights, because they are what the sample grids and
                # every downstream command draw from. Scoring the live weights
                # would pick a "best" epoch nobody ever samples.
                val = validation_loss(
                    diffusion,
                    held_out,
                    model=ema.module,
                    num_classes=cfg.num_classes,
                    device=cfg.device,
                    num_steps=cfg.val_steps,
                    seed=cfg.seed,
                )
                logger.set(**{"val/loss": val})
                if best_val is None or val < best_val:
                    best_val = new_best = val
                logger.set(**{"val/best_loss": best_val})

            # Flushed even for the partial epoch a Ctrl+C ends on: those batches
            # were still work, and the record explains where the run stopped.
            logger.flush(step=epoch)

            if cancelled:
                break

            # Everything from here to the end of the epoch is rank 0's alone:
            # the weights are identical on every rank, so the other three would
            # spend the time redrawing the same grid and racing each other to
            # write the same two files. No collective is skipped by the guard,
            # so the ranks that go straight on to the next epoch stay in step.
            if group.is_main:
                if (
                    cfg.sample_every > 0
                    and (epoch + 1) % cfg.sample_every == 0
                    and reference is not None
                ):
                    grid = save_samples(
                        diffusion,
                        ema,
                        reference,
                        cfg,
                        epoch,
                        labels=reference_labels,
                        noise=sample_noise,
                    )
                    if observer is not None:
                        observer.on_sample(grid)

                last = cfg.ckpt_dir / LAST_CHECKPOINT
                save_checkpoint(
                    last,
                    epoch=epoch,
                    diffusion=diffusion,
                    ema=ema,
                    optim=optim,
                    scaler=scaler,
                    sched=sched,
                    cfg=cfg,
                    best_val=best_val,
                    model_state=_model_state(diffusion, master_params),
                )
                _snapshot_epoch(cfg.ckpt_dir, last, epoch=epoch, keep=cfg.keep_last)

                if new_best is not None and cfg.keep_best:
                    best = cfg.ckpt_dir / BEST_CHECKPOINT
                    # Copied rather than re-serialised: identical bytes, half the I/O.
                    shutil.copy2(last, best)
                    say(f"val/loss {new_best:.5f} is a new best; wrote {best}")

    if master_params is not None:
        # Back to an ordinary float32 network before it leaves this function.
        # Nothing downstream — the samplers, FID, the server — knows about the
        # master copy, and the EMA weights loaded in below are float32 anyway.
        cast(UNet, diffusion.net).convert_to_fp32()

    # Ship the EMA weights: they are what the sample grids were drawn from.
    diffusion.net.load_state_dict(ema.module.state_dict())
    # The barrier is what makes the teardown orderly: rank 0 is still writing
    # the final checkpoint and sample grid while the others are already here,
    # and tearing a NCCL communicator down underneath a rank that has not
    # reached it is how a clean run ends in a warning about an aborted
    # communicator.
    barrier(group)
    distributed_shutdown()
    return diffusion


if __name__ == "__main__":
    train()
