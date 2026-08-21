"""Turn a run's ``metrics.jsonl`` into a figure.

Training already writes every number it measured, and until now nothing read
them back. A table per epoch is the wrong shape for the question the numbers
answer — whether the loss is still going down, whether the held-out score has
turned back up, and which quarter of the diffusion schedule the error is
sitting in. Those are shapes, and a shape wants a picture.

The panels are chosen from what a run actually logged, so an unconditional run
with no validation split does not get an empty axis where ``val/loss`` would
have been. Several runs on one figure share every axis, which is what makes a
sweep comparable: pass more than one and each panel holds one line per run.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tinydiffusion.utils.tracking import METRICS_FILENAME, read_metrics

__all__ = ["PANELS", "metrics_path", "plot_runs"]

PANELS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("loss", ("train/loss", "val/loss", "val/best_loss"), True),
    (
        "timestep quartiles",
        ("train/loss_q0", "train/loss_q1", "train/loss_q2", "train/loss_q3"),
        True,
    ),
    ("learning rate", ("train/lr",), False),
    ("gradient norm", ("train/grad_norm", "train/skipped_step"), False),
    ("throughput", ("time/images_per_second",), False),
)
"""Panel title, the metrics it draws, and whether its y-axis is logarithmic.

A panel with nothing logged for it is dropped rather than drawn empty. The
grouping is the point: `train/loss` against `val/loss` is the overfitting
question, and the four quartiles against each other is the one a single loss
cannot answer.
"""


def metrics_path(target: Path) -> Path:
    """Resolve a run directory or a file to the metrics file to read.

    Args:
        target: either a run's log directory or the JSONL file itself.

    Returns:
        The file to read. A directory resolves to the ``metrics.jsonl`` inside
        it; anything else is returned unchanged.
    """
    return target / METRICS_FILENAME if target.is_dir() else target


def _series(records: Sequence[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    """The steps and values one metric was recorded at.

    Records missing the key are skipped rather than plotted as a gap at zero:
    `val/loss` is written every ``val_every`` epochs, and a metric that started
    part way through a run has no value before it.

    Args:
        records: the run's records, from
            :func:`~tinydiffusion.utils.tracking.read_metrics`.
        key: the metric to pull out.

    Returns:
        Parallel lists of step and value, in step order.
    """
    points = [
        (record["step"], record[key])
        for record in records
        if isinstance(record.get(key), int | float)
    ]
    return [step for step, _ in points], [value for _, value in points]


def plot_runs(
    targets: Sequence[Path],
    out: Path,
    *,
    panels: Sequence[tuple[str, tuple[str, ...], bool]] = PANELS,
    dpi: int = 120,
) -> Path:
    """Draw one figure covering every run given.

    Args:
        targets: run directories or ``metrics.jsonl`` files. More than one puts
            them on shared axes, labelled by the run directory's name.
        out: image file to write. The extension picks the format, so ``.svg``
            works as well as ``.png``.
        panels: what to draw; see :data:`PANELS`. Panels no run logged
            anything for are skipped.
        dpi: resolution for raster formats.

    Returns:
        The file written.

    Raises:
        RuntimeError: if matplotlib is not installed.
        ValueError: if a metrics file is missing or empty, or nothing in any of
            them matches any panel.
    """
    try:
        import matplotlib

        # Chosen before pyplot is imported: a machine with no display would
        # otherwise fail inside a GUI toolkit, which is a confusing way to
        # learn that a headless box cannot open a window it did not want.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plotting needs the 'plots' extra: pip install 'tinydiffusion[plots]'"
        ) from exc

    if not targets:
        raise ValueError("no runs to plot")

    runs: list[tuple[str, list[dict[str, Any]]]] = []
    for target in targets:
        path = metrics_path(target)
        records = read_metrics(path)
        if not records:
            raise ValueError(f"no metrics in {path}")
        # The run directory rather than the filename, which is metrics.jsonl
        # for every run and so labels nothing.
        runs.append((path.parent.name or str(path), records))

    drawn = [
        (title, [key for key in keys if any(_series(rec, key)[0] for _, rec in runs)], log)
        for title, keys, log in panels
    ]
    drawn = [(title, keys, log) for title, keys, log in drawn if keys]
    if not drawn:
        raise ValueError("none of the runs logged anything this can plot")

    figure, axes = plt.subplots(
        len(drawn), 1, figsize=(9, 3 * len(drawn)), sharex=True, squeeze=False
    )
    for axis, (title, keys, log) in zip(axes[:, 0], drawn, strict=True):
        for key in keys:
            for run_name, records in runs:
                steps, values = _series(records, key)
                if not steps:
                    continue
                label = key if len(runs) == 1 else f"{run_name} {key}"
                axis.plot(steps, values, marker="." if len(steps) < 40 else None, label=label)
        axis.set_title(title, loc="left", fontsize=10)
        positive = all(
            value > 0 for key in keys for _, records in runs for value in _series(records, key)[1]
        )
        if log and positive:
            # Only where every point of every run is positive: a log axis drops
            # a non-positive point silently, so asking of the first run alone
            # would quietly delete the second run's zero rather than fall back
            # to a linear axis for it.
            axis.set_yscale("log")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1, 0].set_xlabel("epoch")

    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=dpi)
    plt.close(figure)
    return out
