"""The two screens that sit over the dashboard: the theme picker and the help.

Both are modal, both close on Escape, and neither of them can touch the run —
a display that could stop training by accident while someone was reading the
key list would be worse than no display.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList, Static

from tinydiffusion.tui.themes import cycle_order, save_preferred_theme

if TYPE_CHECKING:
    from textual.binding import BindingType

__all__ = ["HelpScreen", "ThemeScreen"]


class ThemeScreen(ModalScreen[None]):
    """Pick a theme, seeing it applied as the highlight moves.

    Previewing on highlight rather than on selection is the whole reason this
    exists next to the cycle key: thirty themes is too many to walk one keypress
    at a time, and a name alone tells you nothing about how a chart will look in
    it. Escape puts back whatever was in use on the way in.
    """

    DEFAULT_CSS = """
    ThemeScreen {
        align: center middle;
    }
    ThemeScreen > Vertical {
        width: 44;
        height: 80%;
        max-height: 30;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }
    ThemeScreen OptionList {
        height: 1fr;
        background: $surface;
    }
    ThemeScreen .hint {
        color: $text-muted;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._original = ""

    def compose(self) -> ComposeResult:
        """Build the picker.

        Yields:
            The dialog's widgets.
        """
        with Vertical():
            yield Label("theme", classes="heading")
            yield OptionList(id="theme-list")
            yield Label("↑↓ preview · enter keep · esc cancel", classes="hint")

    def on_mount(self) -> None:
        """Fill the list, and start it on the theme currently in use."""
        self._original = self.app.theme
        options = cycle_order(list(self.app.available_themes))
        option_list = self.query_one("#theme-list", OptionList)
        option_list.add_options(options)
        if self._original in options:
            option_list.highlighted = options.index(self._original)
        option_list.focus()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Apply the highlighted theme, so the choice is made by eye.

        Args:
            event: carries the option now under the cursor.
        """
        if isinstance(name := event.option.prompt, str):
            self.app.theme = name

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Keep the selected theme and remember it for next time.

        Args:
            event: carries the chosen option.
        """
        if isinstance(name := event.option.prompt, str):
            self.app.theme = name
            save_preferred_theme(name)
            self.app.notify(f"theme: {name}")
        self.dismiss(None)

    def action_cancel(self) -> None:
        """Put back the theme that was in use before the picker opened."""
        self.app.theme = self._original
        self.dismiss(None)


class HelpScreen(ModalScreen[None]):
    """Every key, in one place, for the ones that do not fit in the footer."""

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    HelpScreen > VerticalScroll {
        width: 68;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }
    HelpScreen .heading {
        text-style: bold;
        color: $text-accent;
    }
    HelpScreen .section {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape,question_mark,f1", "dismiss_help", "Close", show=True),
    ]

    SECTIONS: ClassVar[list[tuple[str, list[tuple[str, str]]]]] = [
        (
            "the run",
            [
                ("s", "start training"),
                ("x", "stop at the next batch, checkpointing on the way"),
                ("r", "restart: stop, then start again"),
            ],
        ),
        (
            "the view",
            [
                ("l", "show or hide the log"),
                ("f", "focus mode: the charts and nothing else"),
                ("c", "clear the log"),
                ("ctrl+s", "save an SVG screenshot"),
            ],
        ),
        (
            "appearance",
            [
                ("t", "theme picker, previewing as you move"),
                ("d", "cycle to the next theme"),
                ("shift+d", "cycle to the previous theme"),
            ],
        ),
        (
            "everything else",
            [
                ("?", "this list"),
                ("ctrl+p", "command palette: every action by name"),
                ("q", "quit"),
            ],
        ),
    ]
    """The keys, grouped by what they are for."""

    def compose(self) -> ComposeResult:
        """Build the help.

        Yields:
            The dialog's widgets.
        """
        with VerticalScroll():
            yield Label("keys", classes="heading")
            for title, rows in self.SECTIONS:
                yield Label(title, classes="section")
                for key, description in rows:
                    yield Static(f"  [b]{key:<8}[/b] [dim]{description}[/dim]")

    def action_dismiss_help(self) -> None:
        """Close the help."""
        self.dismiss(None)
