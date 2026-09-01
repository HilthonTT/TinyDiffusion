"""How the dashboard is laid out, and how it gives way as the terminal shrinks.

The stylesheet lives here rather than on the app class because it is the one
part of the display with no logic in it at all, and because the two breakpoints
it reacts to -- :data:`NARROW` and :data:`VERY_NARROW` -- are decisions about
layout rather than about training.
"""

from __future__ import annotations

from textual.containers import Vertical

__all__ = [
    "APP_CSS",
    "NARROW",
    "VERY_NARROW",
    "Panel",
]


NARROW = 100
"""Columns below which the tiles are dropped and the sidebar tightened."""

VERY_NARROW = 72
"""Columns below which the sidebar goes entirely and the charts take the room."""

APP_CSS = """
    Screen { background: $background; }

    #columns { height: 1fr; }

    #sidebar { width: 38; height: 1fr; padding: 0 0 0 1; scrollbar-size: 1 1; }
    #main { width: 1fr; height: 1fr; padding: 0 1 0 0; }

    Panel {
        border: round $primary 45%;
        border-title-color: $text-accent;
        border-title-style: bold;
        padding: 0 1;
        height: auto;
        background: $surface;
    }
    Panel:focus-within { border: round $primary; }

    #tiles { height: 3; margin-top: 1; }
    StatTile { margin-right: 1; background: $surface; }
    StatTile:last-of-type { margin-right: 0; }

    #chart-panel { height: 1fr; min-height: 5; margin-top: 1; }
    #lower { height: 1fr; min-height: 5; margin-top: 1; }
    #quartile-panel { width: 1fr; height: 1fr; align-vertical: middle; }
    #preview-panel { width: 45%; min-width: 20; height: 1fr; margin-left: 1; }

    #sidebar > Panel { margin-top: 1; width: 1fr; }
    #epoch-bar, #batch-bar { width: 1fr; margin-bottom: 1; }
    #epoch-label, #batch-label { color: $text-muted; }

    #log-panel { height: 10; }
    #log { background: $surface; scrollbar-size: 1 1; }

    .heading { text-style: bold; color: $text-accent; }

    Screen.-narrow #tiles { display: none; }
    Screen.-narrow #sidebar { width: 30; }
    Screen.-narrow #preview-panel { width: 50%; }
    Screen.-narrow #log-panel { height: 8; }
    Screen.-tiny #sidebar { display: none; }
    Screen.-tiny #log-panel { height: 6; }

    Screen.-focus #sidebar { display: none; }
    Screen.-focus #tiles { display: none; }
    Screen.-focus #log-panel { display: none; }
"""


class Panel(Vertical):
    """A bordered box with its heading drawn into the border itself.

    A heading widget inside a box costs a row and repeats what the border is
    already framing; a border title costs nothing and cannot drift away from
    the thing it names.

    Args:
        *children: what goes inside.
        title: the border title.
        id: the widget id.
        classes: any CSS classes.
    """

    def __init__(
        self,
        *children: object,
        title: str,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
    ) -> None:
        super().__init__(*children, id=id, classes=classes)  # type: ignore[arg-type]
        self.border_title = title
