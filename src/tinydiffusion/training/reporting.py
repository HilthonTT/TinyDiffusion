"""Where a run's numbers go, and how often they leave the device.

The loop produces metrics far faster than anything reads them. This module
holds the two intervals that decide when they are read back
(:data:`DRAIN_EVERY`, :data:`QUARTILE_EVERY`), the transfer that does it
(:func:`drain_metrics`), and the adapters that carry the result out to a
watcher or to nowhere at all.
"""

from collections.abc import Mapping

import torch

from tinydiffusion.training.observer import TrainObserver
from tinydiffusion.utils.tracking import RunLogger

__all__ = [
    "DRAIN_EVERY",
    "QUARTILE_EVERY",
    "ObserverBackend",
    "drain_metrics",
    "silent",
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
device and fetched a run at a time by :func:`drain_metrics`, which turns a
sync per batch into one per eight. The numbers are unchanged — the same values
in the same order, read later.

The progress bar's loss therefore updates every eighth batch rather than every
batch, which is the whole of the visible difference.
"""


def silent(message: str) -> None:
    """Swallow a run's messages.

    What ``say`` becomes on the non-main ranks of a distributed run: the plan
    line, the resume notice and the new-best line are all worth printing once,
    and printing them once per GPU is how a four-way run turns its own output
    into noise.

    Args:
        message: the line that is not going anywhere.
    """


def drain_metrics(
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


class ObserverBackend:
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
