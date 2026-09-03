"""Metric logging: console, JSONL on disk, and optionally TensorBoard or W&B.

A training run produces two kinds of number. Some are worth watching live, in
the progress bar; all of them are worth keeping so a finished run can be
compared against the next one. :class:`RunLogger` fans a single ``log`` call
out to every configured backend so the loop does not grow a branch per sink.

Values are buffered by :meth:`RunLogger.accumulate` and flushed as a mean at
the end of each epoch. Logging every batch would be both noisy and slow; a
per-epoch mean of a quantity that moves as much as the diffusion loss is what
actually shows a trend.
"""

import contextlib
import json
import math
import time
import warnings
from collections import defaultdict
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable

import torch

__all__ = [
    "DEFAULT_WANDB_PROJECT",
    "ConsoleBackend",
    "JsonlBackend",
    "LoggerBackend",
    "RunLogger",
    "TensorBoardBackend",
    "WandbBackend",
    "null_logger",
    "quartile_means",
    "read_metrics",
    "timestep_quartile_losses",
    "timestep_quartile_totals",
]

METRICS_FILENAME = "metrics.jsonl"
"""Name of the JSONL file written inside a run's log directory."""

DEFAULT_WANDB_PROJECT = "tinydiffusion"
"""W&B project a run logs into when the config does not name one."""


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


def _records(path: Path) -> list[dict[str, Any]]:
    """Parse a metrics file, skipping anything unreadable.

    A run killed mid-write can leave a truncated final line, and that half a
    record is not a reason to refuse to read the epochs before it.

    Args:
        path: the JSONL file. A missing file reads as empty.

    Returns:
        One dict per parsable line, in file order.
    """
    if not path.exists():
        return []

    parsed = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            parsed.append(record)
    return parsed


def _next_session(path: Path) -> int:
    """The session number a backend about to append to `path` should stamp.

    Args:
        path: the JSONL file that is about to be opened for appending.

    Returns:
        One past the highest session already in the file, or 0 for a new one.
    """
    sessions = [
        record["session"] for record in _records(path) if isinstance(record.get("session"), int)
    ]
    return max(sessions) + 1 if sessions else 0


def read_metrics(path: Path) -> list[dict[str, Any]]:
    """Read a metrics file back as one record per step, newest session first.

    Resuming appends a second copy of every step it replays, so the raw file
    holds more lines than the run has epochs. Reading it back is where that is
    resolved: the last session to write a step is the one that owns it, and
    the superseded records are dropped.

    Args:
        path: the JSONL file, usually ``log_dir/metrics.jsonl``. A missing
            file reads as empty.

    Returns:
        Records sorted by step, one per step, each the newest written for it.
        Unparsable lines are skipped.
    """
    latest: dict[int, dict[str, Any]] = {}
    for record in _records(path):
        step = record.get("step")
        if not isinstance(step, int):
            continue
        previous = latest.get(step)
        if previous is None or record.get("session", 0) >= previous.get("session", 0):
            latest[step] = record
    return [latest[step] for step in sorted(latest)]


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

    Every record carries the reserved keys ``step``, ``time`` and ``session``,
    and they win over a metric of the same name: ``step`` is what every reader
    joins on, so a stray metric called ``step`` must not be able to overwrite
    it.

    ``session`` counts how many times this file has been opened for writing,
    which is what makes a resumed run readable. Resuming from epoch 5 replays
    steps 5 onwards into a file that already holds them; without a session the
    two are indistinguishable, and a plot of the column shows the run doubling
    back on itself. :func:`read_metrics` keeps the newest session per step.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._session = _next_session(path)
        self._handle = path.open("a", encoding="utf-8")

    def write(self, metrics: Mapping[str, float], step: int) -> None:
        """Append one record.

        Non-finite values are stored as ``null``; see :func:`_jsonable`.

        Args:
            metrics: metric name to value. A ``step``, ``time`` or ``session``
                entry is dropped in favour of this backend's own.
            step: step index, stored alongside a wall-clock timestamp.
        """
        record: dict[str, Any] = {
            **{key: _jsonable(value) for key, value in metrics.items()},
            "step": step,
            "time": time.time(),
            "session": self._session,
        }
        self._handle.write(json.dumps(record, allow_nan=False) + "\n")
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


class WandbBackend:
    """Stream metrics to a Weights & Biases run.

    The one backend here that talks to a machine that is not this one, which
    is the whole reason to want it — a run on a remote box is watchable from a
    laptop, and several runs land on shared axes without anyone copying
    ``metrics.jsonl`` around. It is also the reason it is opt-in and the reason
    it never raises during training: a network that drops mid-run must cost the
    run nothing, since ``metrics.jsonl`` already holds everything this is
    sending.

    ``wandb`` is imported lazily so the module stays importable without it, and
    is available via the ``tracking`` extra. Authentication is wandb's own —
    ``wandb login``, or ``WANDB_API_KEY`` in the environment. Set
    ``WANDB_MODE=offline`` to record locally and sync later, which is what a
    training box with no outbound network wants.

    The run's config is sent once at creation, so the sweep view can group and
    filter by hyperparameter. Nothing else about the run leaves the machine:
    no images, no checkpoints, no dataset.
    """

    def __init__(
        self,
        log_dir: Path,
        *,
        project: str,
        name: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        """Start a W&B run.

        Args:
            log_dir: the run's local directory. Handed to wandb as its own
                working directory, so its cache sits beside the run it belongs
                to rather than in the current directory.
            project: W&B project to log into.
            name: display name for the run, or None to let wandb generate one.
            config: hyperparameters to record alongside the metrics. Values
                that are not JSON-native — :class:`~pathlib.Path`, tuples —
                are stringified by wandb itself.

        Raises:
            ImportError: if wandb is not installed.
        """
        import wandb

        log_dir.mkdir(parents=True, exist_ok=True)
        self._run = wandb.init(
            project=project,
            name=name,
            dir=str(log_dir),
            config=dict(config) if config is not None else None,
        )

    def write(self, metrics: Mapping[str, float], step: int) -> None:
        """Send one step's metrics.

        A failure here is warned about and swallowed: the numbers are already
        on disk, and a dropped connection is not a reason to lose an epoch of
        training.

        Args:
            metrics: metric name to value.
            step: step index used as the x-axis.
        """
        try:
            self._run.log(dict(metrics), step=step)
        except Exception as exc:
            warnings.warn(f"wandb logging failed, continuing: {exc}", stacklevel=2)

    def close(self) -> None:
        """Finish the run, so the last step is flushed before the process exits."""
        self._run.finish()


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
        wandb: bool = False,
        wandb_project: str = DEFAULT_WANDB_PROJECT,
        wandb_config: Mapping[str, Any] | None = None,
        extra: Sequence[LoggerBackend] = (),
    ) -> Self:
        """Build a logger with the usual set of backends.

        Args:
            log_dir: directory for ``metrics.jsonl`` and TensorBoard events.
            console: print a table each flush.
            jsonl: append to ``metrics.jsonl``.
            tensorboard: also write TensorBoard events. Requires the
                ``tracking`` extra.
            wandb: also stream to Weights & Biases. Requires the ``tracking``
                extra and an authenticated wandb; see :class:`WandbBackend`.
            wandb_project: W&B project to log into. Ignored unless `wandb`.
            wandb_config: hyperparameters to record with the W&B run, normally
                the training config. Ignored unless `wandb`.
            extra: further backends to fan out to, appended after the built-in
                ones. This is how something outside the loop — a display, a
                test — receives the epoch metrics without the loop growing a
                second path to the same numbers.

        Returns:
            A logger ready for use as a context manager.

        Raises:
            RuntimeError: if `tensorboard` or `wandb` is requested but the
                package behind it is not installed.
        """
        backends: list[LoggerBackend] = []
        try:
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
            if wandb:
                try:
                    backends.append(
                        WandbBackend(
                            log_dir,
                            project=wandb_project,
                            name=log_dir.name,
                            config=wandb_config,
                        )
                    )
                except ImportError as exc:
                    raise RuntimeError(
                        "wandb logging needs the 'tracking' extra: "
                        "pip install 'tinydiffusion[tracking]'"
                    ) from exc
        except BaseException:
            # A backend that fails to build must not leak the ones already
            # open: the JSONL file handle and the TensorBoard writer.
            for backend in backends:
                with contextlib.suppress(Exception):
                    backend.close()
            raise
        backends.extend(extra)
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

        A close that fails while the block is already unwinding is warned
        about rather than raised. Training is a long block to lose: raising
        here would replace whatever the loop failed on — the exception worth
        reading — with a bookkeeping error about a file handle, and demote the
        real one to a ``__context__`` few people look at. With no exception in
        flight there is nothing to shadow, so it propagates as usual.

        Args:
            exc_type: exception class, if the block raised.
            exc: exception instance, if the block raised.
            traceback: traceback, if the block raised.
        """
        try:
            self.close()
        except ExceptionGroup as group:
            if exc_type is None:
                raise
            warnings.warn(
                f"{group.message}: {'; '.join(map(str, group.exceptions))}",
                stacklevel=2,
            )


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
