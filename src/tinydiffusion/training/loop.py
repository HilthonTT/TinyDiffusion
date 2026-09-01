"""The batch and the epoch: everything a run repeats.

:class:`Run` is the state a step needs that outlives the batch it runs on,
assembled once by :func:`~tinydiffusion.training.train.train` after
:mod:`~tinydiffusion.training.setup` has resolved every choice. Everything else
here takes one of those and does a batch, an epoch, or the bookkeeping that
follows one.
"""

import shutil
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import NamedTuple, cast

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tinydiffusion.diffusion.ddpm import LossTerms
from tinydiffusion.diffusion.gaussian_diffusion import Diffusion
from tinydiffusion.diffusion.guidance import Conditioned, drop_labels
from tinydiffusion.training.artifacts import (
    checkpoint_state,
    save_and_report,
    save_samples,
    snapshot_epoch,
)
from tinydiffusion.training.checkpoints import (
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    save_checkpoint,
)
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.distributed import (
    Distributed,
    all_reduce_mean,
    all_reduce_sum,
    any_rank,
    broadcast_object,
)
from tinydiffusion.training.ema import EMA
from tinydiffusion.training.interrupt import InterruptChoice, InterruptGuard
from tinydiffusion.training.observer import BatchProgress, TrainObserver
from tinydiffusion.training.reporting import DRAIN_EVERY, QUARTILE_EVERY, drain_metrics
from tinydiffusion.training.setup import Precision
from tinydiffusion.training.validation import validation_loss
from tinydiffusion.utils.fp16 import (
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from tinydiffusion.utils.tracking import RunLogger, quartile_means, timestep_quartile_totals

__all__ = [
    "EpochOutcome",
    "Run",
    "StepResult",
    "finish_epoch",
    "log_epoch",
    "run_epoch",
    "score_epoch",
    "train_step",
]


@dataclass(frozen=True)
class Run:
    """Everything a training step needs that outlives the batch it runs on.

    Assembled once, after the setup above has resolved every choice, so the
    per-batch and per-epoch helpers take one argument rather than the dozen
    they would otherwise thread through. Frozen because nothing here is ever
    rebound during a run — the objects themselves are what mutate.

    Attributes:
        cfg: run configuration, with its device already resolved.
        group: the training group. Rank 0 of 1 outside a distributed run.
        precision: how this run carries its numbers.
        diffusion: the process being trained. Its ``net`` is the canonical
            network, whatever wrappers `train_net` adds.
        ema: exponential moving average of the network weights.
        train_net: the module the forward pass calls, DDP and compile wrappers
            included.
        ddp: the DDP wrapper alone, for ``no_sync``, or None outside a group.
        optim: the optimiser.
        sched: the LR schedule, stepped once per *applied* optimiser step.
        scaler: AMP gradient scaler.
        model_params: the network's own parameters, in the order every fp16
            helper indexes by.
        master_params: the float32 master copy, or None without one.
        step_params: whichever of the two the optimiser steps.
        say: where a run's messages go.
    """

    cfg: TrainConfig
    group: Distributed
    precision: Precision
    diffusion: Diffusion
    ema: EMA
    train_net: nn.Module
    ddp: DistributedDataParallel | None
    optim: torch.optim.Optimizer
    sched: LRScheduler
    scaler: torch.amp.GradScaler
    model_params: list[torch.Tensor]
    master_params: list[nn.Parameter] | None
    step_params: list[torch.Tensor]
    say: Callable[[str], None]


class StepResult(NamedTuple):
    """What one micro-batch leaves behind for the loop to fold in.

    Attributes:
        metrics: this batch's metrics, values still on the device that produced
            them; see :data:`~tinydiffusion.training.reporting.DRAIN_EVERY`.
        terms: the loss terms, kept so the caller can bucket them by timestep
            without a second forward pass.
        images: how many images the batch held.
    """

    metrics: dict[str, torch.Tensor | float]
    terms: LossTerms
    images: int


def train_step(
    run: Run, x: torch.Tensor, y: torch.Tensor, *, batch: int, num_batches: int
) -> StepResult:
    """Run one micro-batch, and the optimiser step it completes if it does.

    Every micro-batch does a forward and a backward pass; only the last of an
    accumulated group unscales, clips, steps the optimiser and moves the EMA.
    Which one that is follows from the batch index alone, so nothing has to be
    carried between calls.

    Args:
        run: the assembled run state.
        x: the batch's images, on any device.
        y: the batch's labels. Ignored by an unconditional run.
        batch: index of this batch within the epoch.
        num_batches: batches in the epoch, for sizing the ragged last group.

    Returns:
        The batch's metrics, its loss terms and its image count.
    """
    cfg = run.cfg
    x = x.to(cfg.device, non_blocking=True)
    if cfg.channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    model: nn.Module = run.train_net
    if cfg.num_classes is not None:
        labels = drop_labels(
            y.to(cfg.device, non_blocking=True), cfg.num_classes, cfg.class_dropout
        )
        model = Conditioned(run.train_net, labels)

    group_start = batch - batch % cfg.grad_accum
    group_size = min(cfg.grad_accum, num_batches - group_start)
    applies = batch + 1 == group_start + group_size

    if batch == group_start:
        run.optim.zero_grad(set_to_none=True)
        if run.master_params is not None:
            zero_grad(run.model_params)
    with run.precision.autocast():
        terms = run.diffusion.loss_terms(x, model=model)
    loss = terms.loss

    sync: AbstractContextManager[None] = nullcontext()
    if run.ddp is not None and not applies:
        sync = run.ddp.no_sync()
    with sync:
        run.scaler.scale(loss / group_size).backward()  # type: ignore[no-untyped-call]

    grad_norm: torch.Tensor | None = None
    stepped = True
    if applies:
        if run.master_params is not None:
            model_grads_to_master_grads(run.model_params, run.master_params)
        if cfg.grad_clip > 0:
            run.scaler.unscale_(run.optim)
            grad_norm = nn.utils.clip_grad_norm_(run.step_params, cfg.grad_clip)

        scale_before = run.scaler.get_scale()
        run.scaler.step(run.optim)
        run.scaler.update()
        stepped = run.scaler.get_scale() >= scale_before
        if stepped:
            if run.master_params is not None:
                master_params_to_model_params(run.model_params, run.master_params)
                run.ema.update(
                    run.diffusion.net,
                    unflatten_master_params(run.model_params, run.master_params),
                )
            else:
                run.ema.update(run.diffusion.net)
            run.sched.step()

    loss_metric = loss.detach().float()
    if run.group.enabled:
        loss_metric = all_reduce_mean(loss_metric.clone(), run.group)
    metrics: dict[str, torch.Tensor | float] = {"train/loss": loss_metric}
    if applies:
        metrics["train/skipped_step"] = float(not stepped)
    if grad_norm is not None:
        metrics["train/grad_norm"] = grad_norm.detach().float()

    return StepResult(metrics=metrics, terms=terms, images=x.shape[0])


class EpochOutcome(NamedTuple):
    """What one epoch leaves for the bookkeeping that follows it.

    Attributes:
        cancelled: the epoch ended early, at a batch boundary, because a Ctrl+C
            or a watcher asked it to. The checkpoint was already written by
            whichever of the two it was.
        images: images this rank saw.
        elapsed: wall-clock seconds the epoch took.
        quartile_sums: per-quartile loss totals, still on the device and still
            this rank's own; :func:`log_epoch` reduces them.
        quartile_counts: how many samples each of those totals covers.
    """

    cancelled: bool
    images: int
    elapsed: float
    quartile_sums: torch.Tensor
    quartile_counts: torch.Tensor


def run_epoch(
    run: Run,
    loader: DataLoader[tuple[torch.Tensor, int]],
    logger: RunLogger,
    *,
    epoch: int,
    interrupts: InterruptGuard,
    observer: TrainObserver | None,
    best_val: float | None,
) -> EpochOutcome:
    """Train one pass over the loader, stopping early if asked to.

    The two ways out are both taken at a batch boundary, where the model, the
    optimiser and the EMA agree and a checkpoint written from them resumes
    cleanly: a watcher that has asked to stop, and a Ctrl+C. Neither tears the
    run down.

    Args:
        run: the assembled run state.
        loader: this epoch's batches.
        logger: where the per-batch metrics accumulate.
        epoch: zero-based index of the epoch being run.
        interrupts: the Ctrl+C guard, asked once per batch.
        observer: a watcher to report progress to, or None for a tqdm bar.
        best_val: the lowest held-out loss so far, carried only so that a
            checkpoint written on the way out keeps comparing against it.

    Returns:
        What the epoch produced, and whether it finished.
    """
    cfg = run.cfg
    run.diffusion.train()
    loss_ema: float | None = None
    epoch_start = time.perf_counter()
    images = 0
    cancelled = False
    pending: list[dict[str, torch.Tensor | float]] = []
    quartile_sums = torch.zeros(4, device=cfg.device)
    quartile_counts = torch.zeros(4, device=cfg.device)

    with tqdm(
        loader,
        desc=f"epoch {epoch + 1}/{cfg.num_epochs}",
        disable=observer is not None or not run.group.is_main,
    ) as pbar:
        for batch, (x, y) in enumerate(pbar):
            step = train_step(run, x, y, batch=batch, num_batches=len(loader))
            images += step.images
            pending.append(step.metrics)

            if batch % QUARTILE_EVERY == 0:
                batch_sums, batch_counts = timestep_quartile_totals(
                    step.terms.per_sample.float(), step.terms.timesteps, cfg.num_timesteps
                )
                quartile_sums += batch_sums
                quartile_counts += batch_counts

            if len(pending) >= DRAIN_EVERY:
                loss_ema = drain_metrics(pending, logger, loss_ema)
                if loss_ema is not None:
                    pbar.set_postfix(loss=f"{loss_ema:.4f}")
                if observer is not None:
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

            if observer is not None:
                stopping = observer.stop_requested()
                if run.group.enabled:
                    stopping = (
                        any_rank(stopping, run.group, device=cfg.device)
                        if batch % DRAIN_EVERY == 0
                        else False
                    )
                if stopping:
                    if run.group.is_main:
                        save_and_report(
                            cfg,
                            epoch=epoch - 1,
                            diffusion=run.diffusion,
                            ema=run.ema,
                            optim=run.optim,
                            scaler=run.scaler,
                            best_val=best_val,
                            sched=run.sched,
                            model_state=checkpoint_state(run.diffusion, run.master_params),
                            say=run.say,
                        )
                    cancelled = True
                    break

            interrupted = interrupts.requested
            if run.group.enabled:
                aligned = batch % DRAIN_EVERY == 0
                interrupted = (
                    any_rank(interrupted, run.group, device=cfg.device) if aligned else False
                )

            if interrupted:
                choice: InterruptChoice | None = None
                if run.group.is_main:
                    with tqdm.external_write_mode():
                        choice = interrupts.resolve()
                choice = cast(InterruptChoice, broadcast_object(choice, run.group))
                if not choice.stop:
                    continue
                if choice.save:
                    if run.group.is_main:
                        save_and_report(
                            cfg,
                            epoch=epoch - 1,
                            diffusion=run.diffusion,
                            ema=run.ema,
                            optim=run.optim,
                            scaler=run.scaler,
                            best_val=best_val,
                            sched=run.sched,
                            model_state=checkpoint_state(run.diffusion, run.master_params),
                            say=run.say,
                        )
                else:
                    run.say("cancelled without saving")
                cancelled = True
                break

    drain_metrics(pending, logger, loss_ema)

    return EpochOutcome(
        cancelled=cancelled,
        images=images,
        elapsed=time.perf_counter() - epoch_start,
        quartile_sums=quartile_sums,
        quartile_counts=quartile_counts,
    )


def log_epoch(
    run: Run,
    logger: RunLogger,
    *,
    quartile_sums: torch.Tensor,
    quartile_counts: torch.Tensor,
    elapsed: float,
    images: int,
) -> None:
    """Fold an epoch's own numbers — as opposed to its batches' — into the log.

    Args:
        run: the assembled run state.
        logger: where the epoch's record is kept.
        quartile_sums: per-quartile loss totals for the epoch, on the device.
        quartile_counts: how many samples each quartile total covers.
        elapsed: wall-clock seconds the epoch took.
        images: images this rank saw.
    """
    if run.group.enabled:
        all_reduce_sum(quartile_sums, run.group)
        all_reduce_sum(quartile_counts, run.group)
    logger.set(
        **{
            f"train/{name}": value
            for name, value in quartile_means(quartile_sums, quartile_counts).items()
        }
    )
    logger.set(
        **{
            "train/lr": float(run.optim.param_groups[0]["lr"]),
            "train/ema_decay": run.ema.current_decay,
            "train/amp_scale": float(run.scaler.get_scale()) if run.scaler.is_enabled() else 1.0,
            "time/epoch_seconds": elapsed,
            "time/images_per_second": (
                images * run.group.world_size / elapsed if elapsed > 0 else 0.0
            ),
        }
    )


def score_epoch(
    run: Run,
    logger: RunLogger,
    held_out: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    best_val: float | None,
) -> tuple[float | None, float | None]:
    """Score the held-out slice, and record whether it is the best so far.

    Args:
        run: the assembled run state.
        logger: where the score is recorded.
        held_out: the fixed slice to score on.
        best_val: the lowest score seen so far, or None before the first.

    Returns:
        ``(best_val, new_best)``: the running best, and that same number again
        when this epoch set it — which is what tells the caller to write
        ``best.pt`` — or None when it did not.
    """
    val = validation_loss(
        run.diffusion,
        held_out,
        model=run.ema.module,
        num_classes=run.cfg.num_classes,
        device=run.cfg.device,
        num_steps=run.cfg.val_steps,
        seed=run.cfg.seed,
    )
    logger.set(**{"val/loss": val})
    new_best: float | None = None
    if best_val is None or val < best_val:
        best_val = new_best = val
    logger.set(**{"val/best_loss": best_val})
    return best_val, new_best


def finish_epoch(
    run: Run,
    *,
    epoch: int,
    best_val: float | None,
    new_best: float | None,
    reference: torch.Tensor | None,
    reference_labels: torch.Tensor | None,
    sample_noise: torch.Tensor,
    observer: TrainObserver | None,
) -> None:
    """Draw the epoch's grid and write its checkpoints. Rank 0's work alone.

    The weights are identical on every rank, so the others would spend the time
    redrawing the same grid and racing each other to write the same two files.
    No collective is skipped by the guard, so the ranks that go straight on to
    the next epoch stay in step.

    Args:
        run: the assembled run state.
        epoch: zero-based index of the epoch that just finished.
        best_val: the lowest held-out loss seen so far, carried into the
            checkpoint so a resume keeps comparing against it.
        new_best: that same number when this epoch set it, or None. Not None is
            what also makes this epoch's checkpoint ``best.pt``.
        reference: the fixed real strip each grid is drawn against, or None
            when no grid will ever be drawn.
        reference_labels: that strip's labels, or None when unconditional.
        sample_noise: the fixed x_T every grid redraws from.
        observer: a watcher to hand the written grid to, or None.
    """
    if not run.group.is_main:
        return
    cfg = run.cfg
    if cfg.sample_every > 0 and (epoch + 1) % cfg.sample_every == 0 and reference is not None:
        grid = save_samples(
            run.diffusion,
            run.ema,
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
        diffusion=run.diffusion,
        ema=run.ema,
        optim=run.optim,
        scaler=run.scaler,
        sched=run.sched,
        cfg=cfg,
        best_val=best_val,
        model_state=checkpoint_state(run.diffusion, run.master_params),
    )
    snapshot_epoch(cfg.ckpt_dir, last, epoch=epoch, keep=cfg.keep_last)

    if new_best is not None and cfg.keep_best:
        best = cfg.ckpt_dir / BEST_CHECKPOINT
        shutil.copy2(last, best)
        run.say(f"val/loss {new_best:.5f} is a new best; wrote {best}")
