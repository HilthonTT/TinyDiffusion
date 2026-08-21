"""Metric logging: console, JSONL on disk, and optionally TensorBoard.

A training run produces two kinds of number. Some are worth watching live, in
the progress bar; all of them are worth keeping so a finished run can be
compared against the next one. :class:`RunLogger` fans a single ``log`` call
out to every configured backend so the loop does not grow a branch per sink.

Values are buffered by :meth:`RunLogger.accumulate` and flushed as a mean at
the end of each epoch. Logging every batch would be both noisy and slow; a
per-epoch mean of a quantity that moves as much as the diffusion loss is what
actually shows a trend.
"""

import json
import math
import time
from collections import defaultdict
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable

import torch

__all__ = [
    "ConsoleBackend",
    "JsonlBackend",
    "LoggerBackend",
    "RunLogger",
    "TensorBoardBackend",
    "null_logger",
    "quartile_means",
    "timestep_quartile_losses",
    "timestep_quartile_totals",
]

METRICS_FILENAME = "metrics.jsonl"
"""Name of the JSONL file written inside a run's log directory."""


def _write_line(text: str) -> None:
    """Print `text` without corrupting an active tqdm progress bar.

    Args:
        text: the text to emit.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        print(text)
        return
    with tqdm.external_write_mode():
        print(text)


def _jsonable(value: float) -> float | None:
    """Coerce a metric to something strict JSON can hold.

    A diverged run is the one whose log matters most, and it is also the one
    that logs NaN. :mod:`json` would write the bare token ``NaN``, which is a
    Python extension rather than JSON: ``jq``, ``pandas.read_json`` and most
    other readers reject the whole file over it. Stored as ``null`` instead, so
    the step is still there and the gap in it stays visible.

    Args:
        value: the metric value.

    Returns:
        `value`, or None if it is NaN or an infinity.
    """
    return value if math.isfinite(value) else None


@runtime_checkable
class LoggerBackend(Protocol):
    """A sink for scalar metrics.

    Backends are duck-typed rather than subclassed so a caller can pass any
    object with these two methods, including a test double.
    """

    def write(self, metrics: Mapping[str, float], step: int) -> None:
        """Record one flat mapping of metrics at `step`.

        Args:
            metrics: metric name to value. Names use ``group/name`` so
                TensorBoard groups them into panes.
            step: monotonically increasing step index, usually the epoch.
        """
        ...

    def close(self) -> None:
        """Release any file handles or writers held by this backend."""
        ...


class ConsoleBackend:
    """Print metrics as an aligned table, without breaking a tqdm bar.

    The column widths are recomputed per write, so a run whose metric names
    change part way through still lines up.
    """

    def __init__(self, *, precision: int = 4) -> None:
        self._precision = precision

    def write(self, metrics: Mapping[str, float], step: int) -> None:
        """Print one table of metrics.

        Args:
            metrics: metric name to value.
            step: step index, shown in the header.
        """
        if not metrics:
            return

        rendered = {key: self._format(value) for key, value in sorted(metrics.items())}
        key_width = max(len(key) for key in rendered)
        value_width = max(len(value) for value in rendered.values())
        rule = "-" * (key_width + value_width + 7)

        lines = [rule, f"| step {step:<{key_width + value_width - 3}} |", rule]
        lines += [
            f"| {key:<{key_width}} | {value:>{value_width}} |" for key, value in rendered.items()
        ]
        lines.append(rule)
        _write_line("\n".join(lines))

    def _format(self, value: float) -> str:
        """Render a value compactly, switching to exponent form when tiny.

        Fixed-point rendering of a learning rate like ``2.5e-4`` would round to
        ``0.0003`` and lose the digits that distinguish one run from the next,
        so anything that small switches to exponent form.

        Args:
            value: the number to render.

        Returns:
            A short string, e.g. ``"0.0421"`` or ``"2.5e-04"``.
        """
        if value != 0 and abs(value) < 10 ** -(self._precision - 1):
            return f"{value:.1e}"

        return f"{value:.{self._precision}f}"

    def close(self) -> None:
        """No-op: this backend owns no resources."""


class JsonlBackend:
    """Append metrics to a JSON Lines file, one object per step.

    JSONL rather than CSV because a run that starts logging a new metric
    mid-flight just writes a wider object, where a CSV would have to rewrite
    every row already on disk.

    Every record carries the reserved keys ``step`` and ``time``, and they win
    over a metric of the same name: ``step`` is what every reader joins on, so
    a stray metric called ``step`` must not be able to overwrite it.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")

    def write(self, metrics: Mapping[str, float], step: int) -> None:
        """Append one record.

        Non-finite values are stored as ``null``; see :func:`_jsonable`.

        Args:
            metrics: metric name to value. A ``step`` or ``time`` entry is
                dropped in favour of this backend's own.
            step: step index, stored alongside a wall-clock timestamp.
        """
        record: dict[str, Any] = {
            **{key: _jsonable(value) for key, value in metrics.items()},
            "step": step,
            "time": time.time(),
        }
        # allow_nan=False so anything that slipped past _jsonable raises here
        # rather than writing a token no strict parser will read back.
        self._handle.write(json.dumps(record, allow_nan=False) + "\n")
        # Flushed per step so a killed run keeps everything up to the last epoch.
        self._handle.flush()

    def close(self) -> None:
        """Close the underlying file."""
        if not self._handle.closed:
            self._handle.close()


class TensorBoardBackend:
    """Write metrics to TensorBoard event files.

    Uses ``torch.utils.tensorboard``, which ships with PyTorch and needs only
    the ``tensorboard`` package installed — available via the ``tracking``
    extra. It is imported lazily so the module stays importable without it.
    """

    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(log_dir))

    def write(self, metrics: Mapping[str, float], step: int) -> None:
        """Record scalars for one step.

        Args:
            metrics: metric name to value.
            step: step index used as the x-axis.
        """
        for key, value in metrics.items():
            self._writer.add_scalar(key, value, step)
        self._writer.flush()

    def close(self) -> None:
        """Close the event writer."""
        self._writer.close()


class RunLogger:
    """Collect metrics during an epoch and flush their means to every backend.

    Example:
        >>> with RunLogger.for_run(Path("runs/mnist")) as logger:  # doctest: +SKIP
        ...     for epoch in range(epochs):
        ...         for batch in loader:
        ...             logger.accumulate(train_loss=..., train_grad_norm=...)
        ...         logger.flush(step=epoch)
    """

    def __init__(self, backends: list[LoggerBackend]) -> None:
        self._backends = backends
        self._sums: defaultdict[str, float] = defaultdict(float)
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._latest: dict[str, float] = {}

    @classmethod
    def for_run(
        cls,
        log_dir: Path,
        *,
        console: bool = True,
        jsonl: bool = True,
        tensorboard: bool = False,
    ) -> Self:
        """Build a logger with the usual set of backends.

        Args:
            log_dir: directory for ``metrics.jsonl`` and TensorBoard events.
            console: print a table each flush.
            jsonl: append to ``metrics.jsonl``.
            tensorboard: also write TensorBoard events. Requires the
                ``tracking`` extra.

        Returns:
            A logger ready for use as a context manager.

        Raises:
            RuntimeError: if `tensorboard` is requested but not installed.
        """
        backends: list[LoggerBackend] = []
        if console:
            backends.append(ConsoleBackend())
        if jsonl:
            backends.append(JsonlBackend(log_dir / METRICS_FILENAME))
        if tensorboard:
            try:
                backends.append(TensorBoardBackend(log_dir / "tb"))
            except ImportError as exc:
                raise RuntimeError(
                    "tensorboard logging needs the 'tracking' extra: "
                    "pip install 'tinydiffusion[tracking]'"
                ) from exc
        return cls(backends)

    def accumulate(self, **metrics: float) -> None:
        """Buffer per-batch values to be averaged at the next flush.

        Args:
            **metrics: metric name to value for this batch.
        """
        for key, value in metrics.items():
            self._sums[key] += value
            self._counts[key] += 1

    def set(self, **metrics: float) -> None:
        """Record values that should be reported as-is, not averaged.

        Use this for quantities that are already a state rather than a sample:
        the learning rate, the AMP scale, an epoch duration.

        Args:
            **metrics: metric name to value.
        """
        self._latest.update(metrics)

    @property
    def means(self) -> dict[str, float]:
        """Current buffered means, without clearing them.

        Returns:
            Metric name to mean value, including any values set via
            :meth:`set`.
        """
        averaged = {key: self._sums[key] / self._counts[key] for key in self._sums}
        return {**averaged, **self._latest}

    def flush(self, *, step: int) -> dict[str, float]:
        """Write buffered means to every backend and reset the buffers.

        Args:
            step: step index, usually the zero-based epoch.

        Returns:
            The metrics that were written, so a caller can assert on them.
        """
        metrics = self.means
        if metrics:
            for backend in self._backends:
                backend.write(metrics, step)
        self._sums.clear()
        self._counts.clear()
        self._latest.clear()
        return metrics

    def close(self) -> None:
        """Close every backend, even if one of them raises.

        Raises:
            ExceptionGroup: holding whatever the backends raised on close.
        """
        errors: list[Exception] = []
        for backend in self._backends:
            try:
                backend.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("failed to close logger backends", errors)

    def __enter__(self) -> Self:
        """Enter the context, returning this logger.

        Returns:
            This logger.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the backends on the way out.

        Args:
            exc_type: exception class, if the block raised.
            exc: exception instance, if the block raised.
            traceback: traceback, if the block raised.
        """
        self.close()


def timestep_quartile_totals(
    per_sample_loss: torch.Tensor,
    timesteps: torch.Tensor,
    num_timesteps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bucket a batch's losses into quartiles of the diffusion timestep.

    A single averaged loss hides *where* the model is struggling. High-``t``
    error means the denoiser cannot recover structure from near-pure noise;
    low-``t`` error means it cannot clean up the last little bit. They call for
    different fixes, and the mean of the two moves for neither reason.

    Totals rather than means, and tensors rather than floats, so a caller can
    add these into a running pair over a whole epoch and read the result back
    once at the end of it. Nothing here leaves the device, which is the point:
    testing a bucket for emptiness or averaging it on the spot both mean asking
    the device for an answer, and a training loop that does either has
    reintroduced the per-batch synchronisation it went to some trouble to
    remove.

    Args:
        per_sample_loss: shape ``(B,)`` loss for each item in the batch.
        timesteps: shape ``(B,)`` integer timesteps those losses came from.
        num_timesteps: the schedule length, used to size the buckets.

    Returns:
        Tuple of ``(sums, counts)``, both shape ``(4,)``, of the input's dtype
        and on its device.

    Raises:
        ValueError: if the two tensors disagree on batch size.
    """
    if per_sample_loss.shape != timesteps.shape:
        raise ValueError(
            f"loss shape {tuple(per_sample_loss.shape)} does not match "
            f"timestep shape {tuple(timesteps.shape)}"
        )

    # Integer arithmetic rather than a scale-and-truncate through float: with
    # t and num_timesteps both integers this is exact for the whole range,
    # where t * (4 / T) can land a boundary timestep on either side of it.
    quartile = (timesteps * 4 // num_timesteps).clamp(0, 3)
    empty = torch.zeros(4, dtype=per_sample_loss.dtype, device=per_sample_loss.device)
    sums = empty.scatter_add(0, quartile, per_sample_loss)
    counts = empty.scatter_add(0, quartile, torch.ones_like(per_sample_loss))
    return sums, counts


def quartile_means(sums: torch.Tensor, counts: torch.Tensor) -> dict[str, float]:
    """Turn accumulated quartile totals into labelled means.

    This is where the totals come back to the host, in one transfer — which is
    what keeping them as tensors until now was for.

    Args:
        sums: shape ``(4,)`` summed loss per quartile.
        counts: shape ``(4,)`` number of samples that went into each.

    Returns:
        Mapping like ``{"loss_q0": ..., "loss_q3": ...}``. Quartiles that saw
        no samples are omitted rather than reported as a division by zero.
    """
    total, seen = torch.stack([sums.double(), counts.double()]).cpu().tolist()
    return {f"loss_q{index}": total[index] / seen[index] for index in range(4) if seen[index]}


def timestep_quartile_losses(
    per_sample_loss: torch.Tensor,
    timesteps: torch.Tensor,
    num_timesteps: int,
) -> dict[str, float]:
    """Mean loss per timestep quartile, for one batch.

    :func:`timestep_quartile_totals` followed by :func:`quartile_means`. This
    is the convenient form for scoring a single batch; a training loop wants
    the two halves apart, so it can accumulate the totals across an epoch and
    pay for the read back only once.

    Args:
        per_sample_loss: shape ``(B,)`` loss for each item in the batch.
        timesteps: shape ``(B,)`` integer timesteps those losses came from.
        num_timesteps: the schedule length, used to size the buckets.

    Returns:
        Mapping like ``{"loss_q0": ..., "loss_q3": ...}`` holding the mean loss
        per quartile. Quartiles with no samples in this batch are omitted.

    Raises:
        ValueError: if the two tensors disagree on batch size.
    """
    return quartile_means(*timestep_quartile_totals(per_sample_loss, timesteps, num_timesteps))


@contextmanager
def null_logger() -> Generator[RunLogger]:
    """A logger with no backends, for tests and library callers.

    Yields:
        A :class:`RunLogger` that discards everything written to it.
    """
    logger = RunLogger([])
    try:
        yield logger
    finally:
        logger.close()
