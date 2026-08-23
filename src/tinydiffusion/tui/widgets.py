"""The pieces the dashboard is built from.

Everything here renders and nothing here decides: a widget is handed numbers
and draws them, and the app owns what those numbers mean. That split is what
lets the layout be rearranged — or a panel dropped into a different screen —
without touching the training seam at all.

Colours are read from the live theme rather than written into the widgets, so
all thirty-odd palettes in :mod:`tinydiffusion.tui.themes` reach the charts and
the bars, not just the borders Textual styles for us.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual.color import Color
from textual.reactive import reactive
from textual.widgets import Static

from tinydiffusion.tui.chart import braille_cells, format_tick, nice_bounds, tick_values
from tinydiffusion.tui.preview import HALF_BLOCK, half_block_rows

__all__ = [
    "LossChart",
    "QuartileBars",
    "SamplePreview",
    "StatTile",
    "StatusBar",
    "resolve_colour",
    "theme_colour",
]

_FALLBACK = "#808080"
"""What a colour lookup returns when the app is not there to ask — under test,
or during teardown. Grey reads on every palette and misleads on none."""


def theme_colour(widget: Static, name: str) -> str:
    """Look one of the current theme's colours up, in a form Rich will accept.

    Textual's colour variables are written for its own CSS, and two kinds of
    them are not colours Rich has ever heard of:

    * ``auto 60%`` and friends, which Textual resolves against whatever they
      land on at render time. Here that blend is done up front — the foreground
      faded that far towards the surface behind it.
    * ``ansi_blue``, ``ansi_default``, ``transparent``, which the ANSI themes
      use so that the terminal's own palette shows through. Rich spells the
      same colours without the prefix, so the prefix is dropped; there is
      nothing to blend a terminal-defined colour with, so a percentage on one
      is honoured by giving back the colour itself.

    Args:
        widget: any mounted widget; only its app is used.
        name: a Textual colour variable, such as ``primary`` or ``text-muted``.

    Returns:
        A colour Rich can parse, or a neutral grey if there is no app to ask —
        which happens while a widget is being torn down, and is not worth an
        exception.
    """
    try:
        variables = widget.app.get_css_variables()
    except Exception:  # pragma: no cover - only reachable without a running app
        return _FALLBACK
    value = variables.get(name)
    if not value:
        return _FALLBACK
    return resolve_colour(value, variables)


def resolve_colour(value: str, variables: Mapping[str, str]) -> str:
    """Turn one Textual colour variable into a Rich colour.

    Args:
        value: the variable's raw value, such as ``#88C0D0``, ``auto 60%`` or
            ``ansi_blue``.
        variables: the rest of the theme, for the two values ``auto`` is
            resolved against.

    Returns:
        A hex colour, a colour name Rich knows, or a neutral grey for anything
        that cannot be read as either.
    """
    head, alpha = _split(value)
    if head == "auto":
        head, _ = _split(variables.get("foreground", ""))
    if (named := _ansi_name(head)) is not None:
        # The terminal owns this colour, so there is nothing here to fade it
        # against. Better a legible colour at full strength than an exception.
        return named
    if (colour := _colour(head)) is None:
        return _FALLBACK
    if alpha >= 1.0:
        return colour.hex
    ground = variables.get("surface") or variables.get("background") or ""
    behind = _colour(_split(ground)[0])
    return colour.hex if behind is None else behind.blend(colour, alpha).hex


def _split(value: str) -> tuple[str, float]:
    """Separate a colour from the percentage after it.

    Args:
        value: the variable's raw value.

    Returns:
        The colour, and the fraction it is to be applied at — 1.0 where no
        percentage was given, or where the one given cannot be read.
    """
    head, _, rest = value.strip().partition(" ")
    rest = rest.strip()
    if not rest.endswith("%"):
        return (head, 1.0)
    try:
        return (head, max(0.0, min(1.0, float(rest[:-1]) / 100)))
    except ValueError:  # pragma: no cover - Textual does not emit these
        return (head, 1.0)


def _ansi_name(head: str) -> str | None:
    """Rich's name for one of Textual's ANSI colours.

    Args:
        head: a colour with no percentage after it.

    Returns:
        ``ansi_bright_red`` as ``bright_red`` and ``transparent`` as
        ``default``, since both mean "whatever the terminal is using"; None for
        anything that is a colour in its own right.
    """
    if head == "transparent":
        return "default"
    if head.startswith("ansi_"):
        return head.removeprefix("ansi_") or "default"
    return None


def _colour(head: str) -> Color | None:
    """Parse a colour, without raising over one that cannot be parsed.

    Args:
        head: a colour with no percentage after it.

    Returns:
        The colour, or None.
    """
    if not head:
        return None
    try:
        return Color.parse(head)
    except Exception:
        return None


class StatusBar(Static):
    """The one line that says what the run is doing, above everything else.

    A dashboard's first job is to answer "is it working?" from across the room,
    and a coloured word does that where a number buried in a panel does not.
    """

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        padding: 0 1;
        background: $panel;
        color: $foreground;
    }
    """

    STATES: ClassVar[dict[str, tuple[str, str, str]]] = {
        "idle": ("○", "ready", "text-muted"),
        "running": ("●", "training", "text-success"),
        "stopping": ("◐", "stopping", "text-warning"),
        "done": ("✔", "finished", "text-success"),
        "failed": ("✖", "failed", "text-error"),
    }
    """State to its glyph, its word and the theme colour it is drawn in."""

    state: reactive[str] = reactive("idle")
    """Which of :data:`STATES` the run is in."""

    detail: reactive[str] = reactive("")
    """What the run is, in a phrase: dataset, device, precision."""

    note: reactive[str] = reactive("")
    """The right-hand half: whatever is most worth knowing right now."""

    def __init__(self) -> None:
        super().__init__(id="status-bar")

    def watch_state(self) -> None:
        """Redraw when the run changes state."""
        self.redraw()

    def watch_detail(self) -> None:
        """Redraw when the run is described differently."""
        self.redraw()

    def watch_note(self) -> None:
        """Redraw when the note changes."""
        self.redraw()

    def on_resize(self) -> None:
        """Redraw: the note is right-aligned against the current width."""
        self.redraw()

    def redraw(self) -> None:
        """Render the bar into its current width."""
        glyph, word, colour = self.STATES.get(self.state, self.STATES["idle"])
        text = Text()
        text.append(f"{glyph} ", style=theme_colour(self, colour))
        text.append(word.upper(), style=f"bold {theme_colour(self, colour)}")
        if self.detail:
            text.append("   ")
            text.append(self.detail, style=theme_colour(self, "text-muted"))
        if self.note:
            # Padded out rather than justified, so the note keeps its own
            # styling and the left half is never re-wrapped underneath it.
            gap = self.size.width - len(text.plain) - len(self.note) - 2
            if gap > 1:
                text.append(" " * gap)
                text.append(self.note, style=theme_colour(self, "text-accent"))
        self.update(text)


class StatTile(Static):
    """One number, big enough to read without looking for it.

    Args:
        label: what the number is.
        tile_id: the widget id, so the app can address the tile it wants.
        colour: the theme colour the value is drawn in.
    """

    DEFAULT_CSS = """
    StatTile {
        width: 1fr;
        height: 3;
        border: round $primary 40%;
        border-title-align: left;
        padding: 0 1;
        content-align: left middle;
    }
    """

    value: reactive[str] = reactive("-")
    """The number, already formatted. Tiles do no arithmetic."""

    def __init__(self, label: str, tile_id: str, colour: str = "text-primary") -> None:
        super().__init__(id=tile_id)
        self.border_title = label
        self.colour = colour

    def watch_value(self) -> None:
        """Redraw when the number changes."""
        self.redraw()

    def redraw(self) -> None:
        """Render the held value in the theme's colour."""
        self.update(Text(self.value, style=f"bold {theme_colour(self, self.colour)}"))


class LossChart(Static):
    """Training and validation loss on one braille-drawn axis.

    Two series on shared bounds is the whole point: a validation curve that has
    stopped following the training curve down is the single most useful thing a
    training dashboard can show, and it is invisible on two separate charts
    each normalised to its own range.
    """

    DEFAULT_CSS = """
    LossChart {
        height: 1fr;
        /* Low, deliberately: a minimum taller than the panel it sits in makes
           the widget overflow, and the row that gets clipped is the legend. */
        min-height: 3;
        padding: 0 1;
    }
    """

    LABEL_WIDTH = 9
    """Columns reserved for the axis labels, including the gutter after them."""

    def __init__(self, widget_id: str = "loss-chart") -> None:
        super().__init__("no epochs yet", id=widget_id)
        self._series: list[tuple[str, str, list[float]]] = []

    def show(self, series: Sequence[tuple[str, str, list[float]]]) -> None:
        """Hold a new set of series and draw them.

        Args:
            series: ``(name, theme colour, values)`` per line, drawn in order,
                so a later series sits on top where the two coincide.
        """
        self._series = [(name, colour, list(values)) for name, colour, values in series]
        self.redraw()

    def on_resize(self) -> None:
        """Redraw: the chart is drawn to fit, so a resize redraws it."""
        self.redraw()

    def redraw(self) -> None:
        """Render the held series into this widget's current size."""
        points = [value for _, _, values in self._series for value in values]
        if not points:
            self.update(Text("no epochs yet", style=theme_colour(self, "text-muted")))
            return

        width = self.size.width - self.LABEL_WIDTH - 2
        height = self.size.height - 1  # one row for the legend
        if width < 4 or height < 2:
            # Too small to say anything true; the latest numbers still are.
            self.update(self._legend())
            return

        low, high = nice_bounds(points)
        cells = braille_cells(
            [values for _, _, values in self._series],
            width=width,
            height=height,
            low=low,
            high=high,
        )
        colours = [theme_colour(self, colour) for _, colour, _ in self._series]
        muted = theme_colour(self, "text-muted")
        ticks = tick_values(low, high, height)

        text = Text()
        for index, row in enumerate(cells):
            label = format_tick(ticks[index]) if index in (0, height - 1, height // 2) else ""
            text.append(f"{label:>{self.LABEL_WIDTH - 1}} ", style=muted)
            text.append("│", style=muted)
            for character, owner in row:
                text.append(character, style=colours[owner] if owner is not None else "")
            text.append("\n")
        text.append_text(self._legend())
        self.update(text)

    def _legend(self) -> Text:
        """Name each series and give its latest value.

        Returns:
            One line: a swatch, the series' name and where it currently stands.
        """
        text = Text()
        for index, (name, colour, values) in enumerate(self._series):
            if index:
                text.append("   ")
            style = theme_colour(self, colour)
            text.append("━ ", style=style)
            text.append(name, style=theme_colour(self, "text-muted"))
            if values:
                text.append(f" {values[-1]:.4g}", style=f"bold {style}")
        return text


class SamplePreview(Static):
    """The latest sample grid, drawn two pixels to a character cell."""

    DEFAULT_CSS = """
    SamplePreview {
        height: 1fr;
        min-height: 4;
        content-align: center middle;
    }
    """

    def __init__(self) -> None:
        super().__init__("no samples yet", id="preview")
        self._path: Path | None = None

    def show(self, path: Path) -> None:
        """Draw the grid at `path`, and redraw it on every later resize.

        Args:
            path: the PNG to render.
        """
        self._path = path
        self.redraw()

    def on_resize(self) -> None:
        """Re-render: the cell grid is fitted to the widget, so a resize refits it."""
        self.redraw()

    def redraw(self) -> None:
        """Render the held path into this widget's current size."""
        if self._path is None:
            return
        width, height = self.size.width, self.size.height
        if width < 2 or height < 2:
            return
        try:
            rows = half_block_rows(self._path, max_width=width, max_height=height)
        except OSError as exc:
            # A grid that was half-written when we looked is not worth a crash:
            # the next epoch writes another one.
            self.update(f"could not read {self._path.name}: {exc}")
            return

        text = Text()
        for index, row in enumerate(rows):
            if index:
                text.append("\n")
            for top, bottom in row:
                fore = f"rgb({top[0]},{top[1]},{top[2]})"
                back = f"rgb({bottom[0]},{bottom[1]},{bottom[2]})"
                text.append(HALF_BLOCK, style=f"{fore} on {back}")
        self.update(text)


class QuartileBars(Static):
    """The per-quartile training loss, as four bars sharing one scale.

    Which quarter of the diffusion schedule the error is sitting in is the
    thing a single loss number cannot tell you, and four bars say it at a
    glance. They are drawn light-to-dark down the schedule, so which end is
    which needs no reading.
    """

    DEFAULT_CSS = """
    QuartileBars {
        height: auto;
        min-height: 4;
        padding: 0 1;
    }
    """

    BLOCKS = "▏▎▍▌▋▊▉█"
    """Eighth-width blocks, so a bar has eight times a cell's resolution."""

    SHADES: ClassVar[tuple[str, ...]] = (
        "text-primary",
        "text-secondary",
        "text-accent",
        "text-warning",
    )
    """One theme colour per quartile, lowest timestep bucket first."""

    LABELS: ClassVar[tuple[str, ...]] = ("t 0-25%", "t 25-50%", "t 50-75%", "t 75-100%")
    """What each bar covers, said outright rather than left as ``q0``."""

    def __init__(self) -> None:
        super().__init__("waiting for the first epoch", id="quartiles")
        self._values: list[float | None] = []

    def show(self, values: list[float | None]) -> None:
        """Draw one bar per quartile.

        Args:
            values: the four means, lowest timestep bucket first. None where an
                epoch logged nothing for that bucket.
        """
        if not any(value is not None for value in values):
            return
        self._values = list(values)
        self.redraw()

    def on_resize(self) -> None:
        """Redraw: the bars are scaled to the width available."""
        if self._values:
            self.redraw()

    def redraw(self) -> None:
        """Render the held quartiles into this widget's current size."""
        present = [value for value in self._values if value is not None]
        if not present:
            return
        # One scale across all four, so the bars compare with each other rather
        # than each being normalised into saying nothing.
        top = max(present)
        label_width = max(len(label) for label in self.LABELS)
        width = max(self.size.width - label_width - 12, 4)
        muted = theme_colour(self, "text-muted")

        text = Text()
        for index, value in enumerate(self._values):
            if index:
                text.append("\n")
            label = self.LABELS[index] if index < len(self.LABELS) else f"q{index}"
            text.append(f"{label:<{label_width}} ", style=muted)
            if value is None:
                text.append(f"{'-':<{width}}", style=muted)
                continue
            filled = (value / top) * width if top > 0 else 0.0
            whole = int(filled)
            remainder = filled - whole
            bar = "█" * whole
            if whole < width and remainder > 0:
                bar += self.BLOCKS[min(int(remainder * 8), 7)]
            colour = theme_colour(self, self.SHADES[index % len(self.SHADES)])
            text.append(f"{bar:<{width}}", style=colour)
            text.append(f" {value:.4f}", style="bold")
        self.update(text)
