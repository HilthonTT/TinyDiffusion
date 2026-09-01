"""The line a run prints before its first batch.

Everything reported here is settled by the time it is called, so
:func:`describe_plan` is pure: what the line says can be checked without
training an epoch to produce it. :func:`announce_plan` is the impure half,
saying it to the terminal and to any watcher.
"""

from collections.abc import Callable

from tinydiffusion.data.datasets import DatasetSpec
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.distributed import Distributed
from tinydiffusion.training.observer import TrainObserver, TrainPlan
from tinydiffusion.utils.device import describe_device

__all__ = [
    "announce_plan",
    "describe_plan",
    "epochs_phrase",
]


def epochs_phrase(count: int) -> str:
    """Render an epoch count, pluralised.

    Args:
        count: number of epochs.

    Returns:
        e.g. ``"1 epoch"`` or ``"30 epochs"``.
    """
    return f"{count} epoch" if count == 1 else f"{count} epochs"


def describe_plan(
    cfg: TrainConfig,
    spec: DatasetSpec,
    group: Distributed,
    *,
    n_params: int,
    precision: str,
    start_epoch: int,
    steps_per_epoch: int,
    validation_images: int,
) -> str:
    """Render the single line a run prints before its first batch.

    Pure, and deliberately so: everything it reports is settled by the time it
    is called, so what the line says can be checked without training an epoch
    to produce it.

    Args:
        cfg: run configuration.
        spec: the dataset being trained on.
        group: the training group, for the rank count and its own description.
        n_params: parameter count of the network.
        precision: :attr:`Precision.label`, already resolved against the
            hardware the run landed on.
        start_epoch: the epoch this run begins at; non-zero when resuming.
        steps_per_epoch: optimiser steps per epoch, after accumulation.
        validation_images: held-out images each score covers, 0 for a run with
            no validation.

    Returns:
        The plan line, with no trailing newline.
    """
    remaining = cfg.num_epochs - start_epoch
    if start_epoch == 0:
        plan = epochs_phrase(cfg.num_epochs)
    elif remaining > 0:
        plan = f"epochs {start_epoch + 1}-{cfg.num_epochs} ({remaining} to go)"
    else:
        plan = f"nothing to run (checkpoint is at {epochs_phrase(start_epoch)})"
    conditioning = (
        f"{cfg.num_classes} classes, {cfg.class_dropout:g} label dropout"
        if cfg.num_classes is not None
        else "unconditional"
    )
    scoring = (
        f"{validation_images} held-out images every {epochs_phrase(cfg.val_every)}"
        if validation_images
        else "no validation"
    )
    steps = f"{steps_per_epoch} steps/epoch"
    effective_batch = cfg.batch_size * cfg.grad_accum * group.world_size
    if effective_batch != cfg.batch_size:
        how = []
        if cfg.grad_accum > 1:
            how.append(f"x{cfg.grad_accum} accumulated")
        if group.world_size > 1:
            how.append(f"x{group.world_size} ranks")
        steps += f" ({', '.join(how)}, {effective_batch} effective)"
    parallelism = f" | {group}" if group.enabled else ""
    return (
        f"{n_params / 1e6:.2f}M parameters | {spec.name} {cfg.image_size}px x{spec.channels} | "
        f"device {describe_device(cfg.device)} | {precision} | {conditioning} | {plan} | "
        f"{steps} | {scoring}{parallelism}"
    )


def announce_plan(
    cfg: TrainConfig,
    spec: DatasetSpec,
    group: Distributed,
    *,
    observer: TrainObserver | None,
    say: Callable[[str], None],
    n_params: int,
    precision: str,
    start_epoch: int,
    steps_per_epoch: int,
    validation_images: int,
) -> None:
    """Say what the run settled on, to the terminal and to any watcher.

    Args:
        cfg: run configuration.
        spec: the dataset being trained on.
        group: the training group.
        observer: a watcher to hand the same facts to as data, or None.
        say: where the line goes.
        n_params: parameter count of the network.
        precision: :attr:`Precision.label`.
        start_epoch: the epoch this run begins at.
        steps_per_epoch: optimiser steps per epoch, after accumulation.
        validation_images: held-out images each score covers.
    """
    say(
        describe_plan(
            cfg,
            spec,
            group,
            n_params=n_params,
            precision=precision,
            start_epoch=start_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_images=validation_images,
        )
    )
    if observer is not None:
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
                validation_images=validation_images,
                log_dir=cfg.log_dir,
            )
        )
    if cfg.num_epochs - start_epoch <= 0:
        say(f"nothing to do: the checkpoint already covers all {epochs_phrase(cfg.num_epochs)}")
