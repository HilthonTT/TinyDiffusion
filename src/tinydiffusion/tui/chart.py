"""Drawing line charts in a terminal, at four times a cell's resolution.

A sparkline says whether a loss is going down. It cannot say *how far* down,
cannot put two runs of numbers on the same axis, and cannot be read against a
value — which is most of what anyone actually wants from a loss curve. Braille
can: one character cell holds a 2x4 grid of dots, so an eight-row chart is
thirty-two pixels tall and two series can share it.

Like :mod:`tinydiffusion.tui.preview`, this is deliberately free of both Textual
and Rich. It returns characters and the index of the series that owns each one,
never markup, so the arithmetic can be tested on an install with neither, and
the widget on top is left with nothing to do but choose colours.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "BLANK",
    "Cell",
    "braille_cells",
    "format_tick",
    "nice_bounds",
    "tick_values",
]

BLANK = " "
"""What an empty cell holds. A space, not U+2800, so the panel shows through."""

Cell = tuple[str, int | None]
"""One rendered cell: its character, and which series drew it (None if empty)."""

_BASE = 0x2800
"""The braille block's first code point; the dot bits are added to it."""

_DOT_BITS: tuple[tuple[int, ...], ...] = (
    (0x01, 0x02, 0x04, 0x40),
    (0x08, 0x10, 0x20, 0x80),
)
"""``_DOT_BITS[column][row]`` for the 2x4 dot grid inside one cell.

Braille numbers its dots down the left column then down the right, with the
fourth row added last and out of sequence — hence 0x40 and 0x80 trailing their
columns rather than continuing them.
"""

_CELL_WIDTH = 2
"""Dots across one character cell."""

_CELL_HEIGHT = 4
"""Dots down one character cell."""


def nice_bounds(values: Sequence[float]) -> tuple[float, float]:
    """Choose the axis range to draw `values` against.

    A range of zero — one point, or a series that has not moved — has no scale
    to speak of, so it is opened out around the value rather than left to divide
    by nothing. The result is a flat line across the middle, which is the honest
    picture of a number that is not changing.

    Args:
        values: the finite points to cover. Non-finite entries are ignored.

    Returns:
        ``(low, high)`` with ``high > low`` always.
    """
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return (0.0, 1.0)
    low, high = min(finite), max(finite)
    if high > low:
        margin = (high - low) * 0.06
        bottom = low - margin
        return (max(bottom, 0.0) if low >= 0 else bottom, high + margin)
    spread = abs(low) * 0.1 or 1.0
    return (low - spread, low + spread)


def tick_values(low: float, high: float, rows: int) -> list[float]:
    """The value at the middle of each chart row, top first.

    Args:
        low: the bottom of the axis.
        high: the top of the axis.
        rows: how many character rows the chart occupies.

    Returns:
        One value per row, so a label can be printed beside any of them.
    """
    if rows <= 0:
        return []
    if rows == 1:
        return [(low + high) / 2]
    return [high - (high - low) * index / (rows - 1) for index in range(rows)]


def format_tick(value: float) -> str:
    """Render an axis value in as few characters as it can be read in.

    Args:
        value: the number to label.

    Returns:
        A short string: fixed-point where that is legible, and exponent
        notation for the very large and the very small, which is where a loss
        curve spends its later epochs.
    """
    if not math.isfinite(value):
        return "-"
    magnitude = abs(value)
    if magnitude and (magnitude < 1e-3 or magnitude >= 1e5):
        return f"{value:.1e}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def braille_cells(
    series: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
    low: float | None = None,
    high: float | None = None,
) -> list[list[Cell]]:
    """Plot every series into one grid of braille characters.

    Each series is drawn as a continuous line: consecutive samples are joined
    vertically, so a loss that halves between two epochs reads as a fall rather
    than as two unrelated dots. Where more points arrive than there are dot
    columns, a column covers the whole span of the points that fall in it — a
    spike survives being scaled down, which is the one thing it must do.

    Args:
        series: the series, each oldest point first. Later series draw over
            earlier ones where they share a cell.
        width: the chart's width in character cells.
        height: its height in character cells.
        low: the bottom of the axis, or None to take it from the data.
        high: the top of the axis, or None to take it from the data.

    Returns:
        ``height`` rows of ``width`` cells, top row first. Every cell carries
        its character and the index of the series that drew most of its dots;
        an empty cell is :data:`BLANK` and None.
    """
    if width < 1 or height < 1:
        return []

    blank_row: list[Cell] = [(BLANK, None)] * width
    if low is None or high is None:
        bounds = nice_bounds([value for one in series for value in one])
        low = bounds[0] if low is None else low
        high = bounds[1] if high is None else high
    if high <= low:
        return [list(blank_row) for _ in range(height)]

    dot_width = width * _CELL_WIDTH
    dot_height = height * _CELL_HEIGHT
    owners: list[list[int | None]] = [[None] * dot_width for _ in range(dot_height)]

    for index, values in enumerate(series):
        _draw(owners, values, index, dot_width=dot_width, dot_height=dot_height, low=low, high=high)

    rows: list[list[Cell]] = []
    for cell_y in range(height):
        row: list[Cell] = []
        for cell_x in range(width):
            row.append(_cell(owners, cell_x, cell_y))
        rows.append(row)
    return rows


def _draw(
    owners: list[list[int | None]],
    values: Sequence[float],
    index: int,
    *,
    dot_width: int,
    dot_height: int,
    low: float,
    high: float,
) -> None:
    """Mark the dots one series covers.

    Args:
        owners: the dot grid to write into, modified in place.
        values: the series' points, oldest first.
        index: the series' index, written into every dot it claims.
        dot_width: dot columns available.
        dot_height: dot rows available.
        low: the bottom of the axis.
        high: the top of the axis.
    """
    points = [value for value in values if math.isfinite(value)]
    if not points:
        return
    if len(points) == 1:
        points = points * 2

    spans = _column_spans(points, dot_width=dot_width, dot_height=dot_height, low=low, high=high)

    for column, (top, bottom) in enumerate(spans):
        if column + 1 < dot_width:
            middle = (bottom + spans[column + 1][0]) // 2
            top, bottom = min(top, middle), max(bottom, middle)
        if column:
            middle = (top + spans[column - 1][1]) // 2
            top, bottom = min(top, middle), max(bottom, middle)
        for dot_y in range(max(top, 0), min(bottom, dot_height - 1) + 1):
            owners[dot_y][column] = index


def _column_spans(
    points: Sequence[float],
    *,
    dot_width: int,
    dot_height: int,
    low: float,
    high: float,
) -> list[tuple[int, int]]:
    """Reduce a series to the dot rows it covers in each dot column.

    Every column gets a span, whether or not a sample landed in it: a column
    between two epochs takes the interpolated value, which is what makes the
    line continuous when there are fewer points than columns. A column that
    several samples land in covers all of them, which is what stops a spike
    disappearing when there are more.

    Args:
        points: the finite points, oldest first, at least two of them.
        dot_width: dot columns available.
        dot_height: dot rows available.
        low: the bottom of the axis.
        high: the top of the axis.

    Returns:
        ``(top, bottom)`` dot rows per column, left to right.
    """
    last = len(points) - 1
    buckets: list[list[float]] = [[] for _ in range(dot_width)]
    for sample, value in enumerate(points):
        column = round(sample * (dot_width - 1) / last) if dot_width > 1 else 0
        buckets[column].append(value)

    spans: list[tuple[int, int]] = []
    for column in range(dot_width):
        position: float = column * last / (dot_width - 1) if dot_width > 1 else 0.0
        rows = [
            _row_for(value, dot_height=dot_height, low=low, high=high)
            for value in (_interpolate(points, position), *buckets[column])
        ]
        spans.append((min(rows), max(rows)))
    return spans


def _interpolate(points: Sequence[float], position: float) -> float:
    """Read a series at a fractional index.

    Args:
        points: the points, oldest first.
        position: where to read, in ``[0, len(points) - 1]``.

    Returns:
        The value there, straight-line between the two points either side.
    """
    lower = int(position)
    if lower >= len(points) - 1:
        return points[-1]
    fraction = position - lower
    return points[lower] * (1 - fraction) + points[lower + 1] * fraction


def _row_for(value: float, *, dot_height: int, low: float, high: float) -> int:
    """Map a value to a dot row.

    Args:
        value: the point.
        dot_height: dot rows available.
        low: the bottom of the axis.
        high: the top of the axis.

    Returns:
        The row, counted from the top and clamped into the grid — a value
        outside an explicitly given axis is pinned to its edge rather than
        drawn off the chart.
    """
    fraction = (high - value) / (high - low)
    return max(0, min(dot_height - 1, round(fraction * (dot_height - 1))))


def _cell(owners: Sequence[Sequence[int | None]], cell_x: int, cell_y: int) -> Cell:
    """Fold one cell's 2x4 dots into a character.

    Args:
        owners: the dot grid.
        cell_x: the cell's column.
        cell_y: the cell's row.

    Returns:
        The braille character and the series that drew most of its dots — with
        ties going to the later series, which is the one drawn on top.
    """
    bits = 0
    counts: dict[int, int] = {}
    for dot_x in range(_CELL_WIDTH):
        for dot_y in range(_CELL_HEIGHT):
            owner = owners[cell_y * _CELL_HEIGHT + dot_y][cell_x * _CELL_WIDTH + dot_x]
            if owner is None:
                continue
            bits |= _DOT_BITS[dot_x][dot_y]
            counts[owner] = counts.get(owner, 0) + 1
    if not bits:
        return (BLANK, None)
    winner = max(counts, key=lambda owner: (counts[owner], owner))
    return (chr(_BASE + bits), winner)
