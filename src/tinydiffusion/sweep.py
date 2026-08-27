"""Run one config over a grid of hyperparameters, one directory per point.

A sweep is the thing ``--set`` was always for, written down. Looping the shell
over ``--set lr=1e-4 --set lr=2e-4`` works, and then every point writes to the
same ``log_dir``, the same ``ckpt_dir`` and the same ``contents/``, so the
second run overwrites the first's record of itself and the comparison the sweep
was for is gone. This module is that loop with the bookkeeping done: each point
gets its own directory under a root, named after the values that distinguish
it, and the whole root is what ``plot`` takes to draw them on shared axes.

    tinydiffusion sweep --config configs/mnist.toml --axis lr=1e-4,2e-4,4e-4
    tinydiffusion plot runs/sweep/*

Every combination is run: two axes of three and two values are six points, and
the cost is six training runs. ``--dry-run`` prints the grid without training
anything, which is the cheap way to find out that a sweep is larger than it
looked.

Points are independent, so one that fails does not stop the rest — a bad
combination of settings is a fact about that point, and losing five good runs to
it would be the wrong trade. What each point did is in the summary at the end,
and a sweep with any failure in it exits non-zero.
"""

import dataclasses
import itertools
import traceback
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tinydiffusion.training.config import TrainConfig
from tinydiffusion.utils.tracking import METRICS_FILENAME, read_metrics

__all__ = [
    "DIRECTORY_FIELDS",
    "SweepPoint",
    "SweepRun",
    "point_name",
    "run_sweep",
    "sweep_points",
    "sweep_summary",
]

DIRECTORY_FIELDS = ("log_dir", "ckpt_dir", "out_dir")
"""Config fields the sweep sets itself, and so cannot be swept over.

Each point's directories are what keep it from overwriting its neighbours, so
an axis over one of them would defeat the only thing this module does that a
shell loop does not.
"""

_SAFE_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-=")
"""What survives into a directory name. Everything else becomes a dash."""


class TrainFn(Protocol):
    """The training entry point a sweep drives, as it needs it.

    Narrower than :func:`~tinydiffusion.training.train.train`'s real signature
    on purpose: a sweep passes a config and nothing else, and typing it this
    way is what lets a test hand in a recorder instead.
    """

    def __call__(self, cfg: TrainConfig, /) -> Any:
        """Train one run to completion."""
        ...


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One combination of axis values, and the run it describes.

    Attributes:
        name: the point's directory name, derived from `overrides`.
        overrides: the axis values that distinguish this point from its
            neighbours. Constants shared by every point are in `config` but not
            here, so the name stays as short as the grid allows.
        config: the fully resolved config, with the directories already pointed
            at this point's own.
    """

    name: str
    overrides: Mapping[str, Any]
    config: TrainConfig

    @property
    def log_dir(self) -> Path:
        """Where this point's ``metrics.jsonl`` lands."""
        return self.config.log_dir


@dataclass(slots=True)
class SweepRun:
    """What became of one point.

    Attributes:
        point: the point that was run.
        error: the exception it failed with, or None if it finished.
        best_val_loss: the lowest ``val/loss`` in its metrics, or None if it
            logged none — which is what an unvalidated run
            (``val_every = 0``) and a failed one both look like.
        epochs: how many epochs it recorded.
    """

    point: SweepPoint
    error: BaseException | None = None
    best_val_loss: float | None = None
    epochs: int = 0

    @property
    def ok(self) -> bool:
        """Whether the point trained to completion."""
        return self.error is None


def point_name(overrides: Mapping[str, Any]) -> str:
    """Name a point after the values that distinguish it.

    Args:
        overrides: this point's axis values, in axis order.

    Returns:
        A directory name like ``lr=0.0001_batch_size=64``. Characters a
        filesystem would object to are replaced with a dash, so the name is
        usable on Windows and POSIX alike; that makes it lossy in principle,
        and unambiguous in practice for the scalars a grid is normally over.
        An empty mapping — a sweep with no axes, which is one ordinary run —
        names itself ``base``.
    """
    if not overrides:
        return "base"
    parts = []
    for field, value in overrides.items():
        rendered = "".join(ch if ch in _SAFE_CHARS else "-" for ch in str(value))
        parts.append(f"{field}={rendered}")
    return "_".join(parts)


def sweep_points(
    base: TrainConfig,
    axes: Sequence[tuple[str, Sequence[Any]]],
    root: Path,
) -> list[SweepPoint]:
    """Expand the axes into every combination, each with its own directories.

    Args:
        base: the config every point starts from, usually a file.
        axes: ``(field, values)`` pairs. The product is taken in this order, so
            the first axis varies slowest and the directory listing groups by
            it.
        root: directory the points are created under.

    Returns:
        One point per combination, in product order.

    Raises:
        ValueError: if an axis names one of :data:`DIRECTORY_FIELDS`, names no
            config field, offers no values, or repeats a field; or if a
            combination is not a valid config.
    """
    seen: set[str] = set()
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    for field, values in axes:
        if field in DIRECTORY_FIELDS:
            raise ValueError(
                f"cannot sweep over {field!r}: the sweep sets it per point, which is "
                "what keeps the runs from overwriting each other"
            )
        if field not in known:
            raise ValueError(f"unknown config field {field!r} on the left of an axis")
        if not values:
            raise ValueError(f"axis {field!r} has no values to sweep over")
        if field in seen:
            raise ValueError(f"axis {field!r} given twice; put every value in one axis")
        seen.add(field)

    fields = [field for field, _ in axes]
    points: list[SweepPoint] = []
    for combination in itertools.product(*(values for _, values in axes)):
        overrides = dict(zip(fields, combination, strict=True))
        name = point_name(overrides)
        directory = root / name
        # from_mapping rather than replace, for the reason the CLI uses it: an
        # axis value arrives as whatever TOML made of it, and this is what
        # coerces a string to a Path or a list to a tuple. It also validates,
        # so a combination that cannot be trained is rejected here rather than
        # after the points before it have already run.
        config = TrainConfig.from_mapping(
            {
                **dataclasses.asdict(base),
                **overrides,
                "log_dir": directory,
                "ckpt_dir": directory / "checkpoints",
                "out_dir": directory / "contents",
            }
        )
        points.append(SweepPoint(name=name, overrides=overrides, config=config))
    return points


def _best_val_loss(log_dir: Path) -> tuple[float | None, int]:
    """Read a finished point's best validation loss out of its metrics.

    Args:
        log_dir: the point's log directory.

    Returns:
        Tuple of ``(best_val_loss, epochs)``. The loss is None where the run
        logged none, which covers both a run with validation turned off and one
        that died before its first epoch ended.
    """
    records = read_metrics(log_dir / METRICS_FILENAME)
    losses = [
        record["val/loss"] for record in records if isinstance(record.get("val/loss"), int | float)
    ]
    return (min(losses) if losses else None), len(records)


def run_sweep(
    points: Sequence[SweepPoint],
    *,
    train: TrainFn,
    skip_existing: bool = False,
    say: Any = print,
) -> Iterator[SweepRun]:
    """Train every point, yielding each result as it lands.

    Yielding rather than returning a list is what lets a caller print a running
    summary: a sweep is hours long, and a report that only exists at the end is
    one an interrupted sweep never gets.

    Args:
        points: the grid, from :func:`sweep_points`.
        train: what to run for each point. Injected rather than imported so a
            test need not train anything.
        skip_existing: leave a point alone if its log directory already holds
            metrics. This is what makes a sweep resumable after an interrupt,
            and it reads the existing numbers rather than pretending the point
            did not happen.
        say: where progress lines go.

    Yields:
        One :class:`SweepRun` per point, in order.

    Raises:
        KeyboardInterrupt: propagated rather than recorded. A failing point is
            a fact about that point; an interrupt is an instruction about the
            sweep.
    """
    for index, point in enumerate(points, start=1):
        prefix = f"[{index}/{len(points)}] {point.name}"
        if skip_existing and (point.log_dir / METRICS_FILENAME).exists():
            say(f"{prefix}: already run, skipping")
            best, epochs = _best_val_loss(point.log_dir)
            yield SweepRun(point=point, best_val_loss=best, epochs=epochs)
            continue

        say(f"{prefix}: training")
        try:
            train(point.config)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            say(f"{prefix}: failed: {exc}")
            say(traceback.format_exc())
            yield SweepRun(point=point, error=exc)
            continue

        best, epochs = _best_val_loss(point.log_dir)
        yield SweepRun(point=point, best_val_loss=best, epochs=epochs)


def sweep_summary(runs: Sequence[SweepRun]) -> str:
    """Render the finished sweep as a table, best first.

    Args:
        runs: the results, in the order they were run.

    Returns:
        A multi-line string: one row per point, sorted by validation loss with
        the points that logged none — failures included — at the bottom in the
        order they ran.
    """
    if not runs:
        return "no points to run"

    def sort_key(run: SweepRun) -> tuple[int, float]:
        # Two keys rather than one: a missing loss is not "infinitely bad", it
        # is unranked, and sorting it as a number would put a failed point and
        # an unvalidated one in a meaningful-looking position.
        return (1, 0.0) if run.best_val_loss is None else (0, run.best_val_loss)

    ordered = sorted(runs, key=sort_key)
    width = max(len(run.point.name) for run in runs)
    lines = [f"{'point':<{width}}   epochs   best val/loss"]
    for run in ordered:
        if not run.ok:
            status = "failed"
        elif run.best_val_loss is None:
            status = "-"
        else:
            status = f"{run.best_val_loss:.5f}"
        lines.append(f"{run.point.name:<{width}}   {run.epochs:>6}   {status:>13}")

    failed = [run for run in runs if not run.ok]
    if failed:
        lines += ["", f"{len(failed)} of {len(runs)} points failed:"]
        lines += [f"  {run.point.name}: {run.error}" for run in failed]
    return "\n".join(lines)
