"""Watching a training run from outside the loop.

:func:`~tinydiffusion.training.train.train` reports itself to a terminal:
``print`` for anything worth saying once, a tqdm bar for the batch it is on,
and a :class:`~tinydiffusion.utils.tracking.RunLogger` for the numbers. That is
the right set for a run in a shell, and the wrong one for a run inside a
:mod:`tinydiffusion.tui` — where stdout belongs to the display, a progress bar
drawn with carriage returns lands in the middle of a widget, and the Ctrl+C
prompt would block the worker thread on an ``input`` nobody can answer.

An observer is the seam. Pass one and the loop reports through it instead of to
the terminal; pass nothing and every one of those behaviours is exactly what it
was. It is deliberately a small protocol of plain data — no widgets, no event
loop, nothing importable only with the ``tui`` extra — so the training loop
stays unaware of what is watching it, and a test double is a class with six
short methods.

Epoch metrics do *not* come through here. They already have a fan-out built for
this in :class:`~tinydiffusion.utils.tracking.LoggerBackend`, and a second path
to the same numbers would be one to keep in step for no gain: a watcher that
wants them registers a backend.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = ["BatchProgress", "TrainObserver", "TrainPlan"]


@dataclass(frozen=True, slots=True)
class TrainPlan:
    """What a run settled on, once, before the first batch.

    The same facts the plan line prints, as data rather than as a sentence —
    everything a display needs to describe the run it is about to show, without
    parsing the sentence back apart.

    Attributes:
        dataset: the registered dataset name being trained on.
        image_size: resolution in pixels.
        channels: image channels.
        device: the resolved device string.
        device_description: the device with its GPU model, where it names one.
        parameters: total parameter count of the network.
        precision: how the run is training — ``"amp fp16"``, ``"fp16 weights
            (float32 master)"``, ``"amp off"`` — including the modifiers that
            follow it, so this is the whole of what the plan line said.
        num_classes: class count for a conditional run, or None.
        start_epoch: the epoch this run begins at; non-zero when resuming.
        num_epochs: the epoch it stops at.
        steps_per_epoch: optimiser steps per epoch, after accumulation.
        batch_size: micro-batch size.
        grad_accum: micro-batches per optimiser step.
        validation_images: how many held-out images each score covers, 0 for a
            run with no validation.
        log_dir: where the run writes its metrics.
    """

    dataset: str
    image_size: int
    channels: int
    device: str
    device_description: str
    parameters: int
    precision: str
    num_classes: int | None
    start_epoch: int
    num_epochs: int
    steps_per_epoch: int
    batch_size: int
    grad_accum: int
    validation_images: int
    log_dir: Path


@dataclass(frozen=True, slots=True)
class BatchProgress:
    """Where a run has got to, reported as it goes.

    Emitted at the same point the tqdm bar's postfix was updated — once per
    :data:`~tinydiffusion.training.train.DRAIN_EVERY` batches, not once per
    batch. That is not a limitation to work around: the metrics behind it live
    on the device until then, and reading them any sooner is the
    synchronisation the buffer exists to avoid.

    Attributes:
        epoch: zero-based epoch index.
        num_epochs: the epoch the run stops at.
        batch: zero-based index of the batch just finished.
        num_batches: batches in this epoch.
        loss: the smoothed training loss, on the same EMA the progress bar
            showed. None before the first drain.
        images: images seen so far this epoch.
        seconds: seconds elapsed in this epoch.
    """

    epoch: int
    num_epochs: int
    batch: int
    num_batches: int
    loss: float | None
    images: int
    seconds: float

    @property
    def epoch_fraction(self) -> float:
        """How far through the current epoch this is, in ``[0, 1]``."""
        if self.num_batches <= 0:
            return 0.0
        return min((self.batch + 1) / self.num_batches, 1.0)

    @property
    def images_per_second(self) -> float:
        """Throughput so far this epoch, or 0 before any time has passed."""
        return self.images / self.seconds if self.seconds > 0 else 0.0


@runtime_checkable
class TrainObserver(Protocol):
    """What a training loop reports to when it is not reporting to a terminal.

    Every method is called from the training thread, so an implementation that
    belongs to a UI has to hand the work over rather than touch its widgets
    here; :class:`~tinydiffusion.tui.app.TuiObserver` posts messages.
    """

    def on_plan(self, plan: TrainPlan) -> None:
        """The run is about to start, and has settled what it is doing.

        Args:
            plan: the resolved settings.
        """
        ...

    def on_message(self, text: str) -> None:
        """One line the run would otherwise have printed.

        Args:
            text: the message, without a trailing newline.
        """
        ...

    def on_batch(self, progress: BatchProgress) -> None:
        """A run of batches has finished and its metrics have been read back.

        Args:
            progress: where the run has got to.
        """
        ...

    def on_epoch(self, step: int, metrics: Mapping[str, float]) -> None:
        """An epoch's metrics have been flushed.

        The same mapping every :class:`~tinydiffusion.utils.tracking.LoggerBackend`
        receives; an observer is registered as one so it does not need to be
        passed separately.

        Args:
            step: the epoch index.
            metrics: metric name to value.
        """
        ...

    def on_sample(self, path: Path) -> None:
        """A sample grid has been written.

        Args:
            path: the PNG just saved.
        """
        ...

    def stop_requested(self) -> bool:
        """Whether the run should stop at the next batch boundary.

        Polled where a Ctrl+C would be noticed, and answered the same way: the
        run stops after writing a resumable checkpoint. Nothing prompts, since
        an observer asking to stop has already decided.

        Returns:
            True to stop.
        """
        ...
