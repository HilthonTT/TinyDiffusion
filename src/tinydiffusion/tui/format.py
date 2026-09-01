"""Numbers into the strings the dashboard shows.

All pure, and none of it touches a widget: what a panel says can be checked
without standing an app up to say it.
"""

from __future__ import annotations

from collections.abc import Mapping

from rich.text import Text

from tinydiffusion.training.observer import BatchProgress

__all__ = [
    "duration",
    "epoch_summary",
    "run_eta",
    "two_columns",
]


def two_columns(rows: list[tuple[str, str]]) -> Text:
    """Render label/value pairs as an aligned two-column block.

    Args:
        rows: the pairs, in the order they should appear.

    Returns:
        The block, labels dimmed and values plain.
    """
    if not rows:
        return Text()
    width = max(len(label) for label, _ in rows)
    text = Text()
    for index, (label, value) in enumerate(rows):
        if index:
            text.append("\n")
        text.append(f"{label:<{width}}  ", style="dim")
        text.append(value)
    return text


def epoch_summary(metrics: Mapping[str, float]) -> str:
    """One line describing an epoch, for the log.

    Args:
        metrics: the epoch's metrics.

    Returns:
        A compact summary of the few that matter, or a note that there were
        none.
    """
    parts = [
        f"{label} {value:.4g}"
        for key, label in (
            ("train/loss", "loss"),
            ("val/loss", "val"),
            ("time/images_per_second", "img/s"),
        )
        if (value := metrics.get(key)) is not None
    ]
    return "  ".join(parts) or "no metrics"


def duration(seconds: float) -> str:
    """Render a number of seconds as ``1h02m``, ``3m20s`` or ``12s``.

    Args:
        seconds: the duration. Negatives read as zero.

    Returns:
        A compact string, coarsening as it grows so the width stays steady.
    """
    whole = max(int(seconds), 0)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def run_eta(progress: BatchProgress) -> float | None:
    """Estimate the seconds left in the whole run.

    Extrapolated from the current epoch's rate alone, which is the only rate a
    run that has not finished an epoch yet has. It is therefore optimistic on
    the first epoch of a run that also validates and samples at the end of one.

    Args:
        progress: the latest batch report.

    Returns:
        Seconds remaining, or None before there is anything to extrapolate
        from.
    """
    done = progress.epoch_fraction
    if done <= 0 or progress.seconds <= 0:
        return None
    per_epoch = progress.seconds / done
    remaining = (progress.num_epochs - progress.epoch) - done
    return max(per_epoch * remaining, 0.0)
