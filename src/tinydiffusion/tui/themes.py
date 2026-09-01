"""Colour schemes for the dashboard, and remembering which one was chosen.

Textual ships a good set of themes, and this adds a set of its own on top:
palettes picked for a screen that is mostly charts, bars and half-block pixels,
where the accent colour is doing real work rather than decorating a form. They
are registered on the app at startup and take their place in the same cycle,
the same picker and the same command palette as the built-ins, so there is one
list of themes rather than ours and theirs.

The chosen theme outlives the run: it is written to a small file under the
user's config directory and read back the next time the dashboard opens. That
file is the only state the TUI keeps, and a missing or unreadable one is not an
error — it simply means the default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from textual.theme import Theme

__all__ = [
    "CUSTOM_THEMES",
    "DEFAULT_THEME",
    "cycle_order",
    "load_preferred_theme",
    "preferences_path",
    "save_preferred_theme",
]

DEFAULT_THEME = "tinydiffusion"
"""The theme a first run opens with."""


def _theme(
    name: str,
    *,
    primary: str,
    secondary: str,
    accent: str,
    background: str,
    surface: str,
    panel: str,
    foreground: str,
    success: str,
    warning: str,
    error: str,
    dark: bool = True,
) -> Theme:
    """Build one theme, with the variables the dashboard leans on filled in.

    Args:
        name: how the theme is referred to, in the picker and in the file.
        primary: the dominant hue — borders, the train-loss line, headings.
        secondary: the second series and the quieter accents.
        accent: the colour reserved for what the eye should land on.
        background: behind everything.
        surface: behind a panel.
        panel: behind a panel that sits on a panel.
        foreground: body text.
        success: a finished run, a falling loss.
        warning: a stop that has been asked for but has not happened yet.
        error: a run that died.
        dark: whether Textual should treat this as a dark theme, which decides
            how it derives the shades it has not been given.

    Returns:
        The theme, ready to register.
    """
    return Theme(
        name=name,
        primary=primary,
        secondary=secondary,
        accent=accent,
        background=background,
        surface=surface,
        panel=panel,
        foreground=foreground,
        success=success,
        warning=warning,
        error=error,
        dark=dark,
        variables={
            "footer-key-foreground": accent,
            "footer-description-foreground": foreground,
            "block-cursor-background": primary,
            "block-cursor-foreground": background,
        },
    )


CUSTOM_THEMES: tuple[Theme, ...] = (
    _theme(
        "tinydiffusion",
        primary="#6C8CFF",
        secondary="#3FD0C9",
        accent="#B48EFF",
        background="#0B0F16",
        surface="#141A24",
        panel="#1C2531",
        foreground="#D7DFEC",
        success="#5CD97B",
        warning="#FFB454",
        error="#FF6E6E",
    ),
    _theme(
        "tinydiffusion-light",
        primary="#3B5BDB",
        secondary="#0B8F8A",
        accent="#7048E8",
        background="#F6F8FC",
        surface="#FFFFFF",
        panel="#EAEFF7",
        foreground="#161B26",
        success="#2B8A3E",
        warning="#B26B00",
        error="#C92A2A",
        dark=False,
    ),
    _theme(
        "latent",
        primary="#B388FF",
        secondary="#FF7AC6",
        accent="#7CE0FF",
        background="#100820",
        surface="#1A1030",
        panel="#241640",
        foreground="#E6DCFF",
        success="#8CE99A",
        warning="#FFD07B",
        error="#FF7A9A",
    ),
    _theme(
        "ember",
        primary="#FF8A3D",
        secondary="#FFC46B",
        accent="#FF5F56",
        background="#120C09",
        surface="#1D1410",
        panel="#2A1D16",
        foreground="#F2E3D6",
        success="#9BD96A",
        warning="#FFD166",
        error="#FF5252",
    ),
    _theme(
        "mint",
        primary="#3DDC97",
        secondary="#7CF0C8",
        accent="#5AC8FA",
        background="#06110E",
        surface="#0D1D18",
        panel="#132A22",
        foreground="#DAF2E9",
        success="#6EE7A8",
        warning="#F2C94C",
        error="#FF6B6B",
    ),
    _theme(
        "oceanic",
        primary="#4FB6E0",
        secondary="#59D3C0",
        accent="#9AD1FF",
        background="#08131C",
        surface="#10212E",
        panel="#172D3E",
        foreground="#D2E3F0",
        success="#67D9A3",
        warning="#EFC078",
        error="#F2777A",
    ),
    _theme(
        "synthwave",
        primary="#FF4FD8",
        secondary="#00E5FF",
        accent="#FFD166",
        background="#150C2C",
        surface="#1F1240",
        panel="#2A1954",
        foreground="#F1E7FF",
        success="#5CF2C0",
        warning="#FFB347",
        error="#FF5C8A",
    ),
    _theme(
        "noir",
        primary="#C9C9C9",
        secondary="#8E8E8E",
        accent="#FFFFFF",
        background="#080808",
        surface="#131313",
        panel="#1C1C1C",
        foreground="#E3E3E3",
        success="#B8B8B8",
        warning="#D6D6D6",
        error="#F0F0F0",
    ),
    _theme(
        "paper",
        primary="#8A5A2B",
        secondary="#3F7D6B",
        accent="#B5651D",
        background="#FBF7EF",
        surface="#FFFDF8",
        panel="#F0E9DA",
        foreground="#2B2620",
        success="#3F7D3F",
        warning="#B07A17",
        error="#B23A3A",
        dark=False,
    ),
    _theme(
        "arctic",
        primary="#2A6FB0",
        secondary="#2E9E8F",
        accent="#5B58C9",
        background="#F1F5FA",
        surface="#FFFFFF",
        panel="#E2EBF4",
        foreground="#15202B",
        success="#1E7A4C",
        warning="#A06A00",
        error="#B32D3A",
        dark=False,
    ),
)
"""The themes this project adds, in the order the cycle key walks them."""

_BUILTIN_ORDER: tuple[str, ...] = (
    "textual-dark",
    "textual-light",
    "nord",
    "gruvbox",
    "dracula",
    "tokyo-night",
    "catppuccin-mocha",
    "catppuccin-latte",
    "monokai",
    "flexoki",
    "solarized-dark",
    "solarized-light",
    "rose-pine",
    "rose-pine-dawn",
    "atom-one-dark",
    "atom-one-light",
)
"""Built-ins worth putting in the cycle, after ours."""


def cycle_order(available: list[str]) -> list[str]:
    """Order the themes for the cycle key and the picker.

    Ours lead, then the built-ins worth having, then anything else registered —
    so the key walks a deliberate sequence rather than an alphabetical one, and
    a theme added later still appears rather than being silently dropped.

    Args:
        available: every theme name the app knows, in any order.

    Returns:
        The same names, ordered, with no duplicates.
    """
    known = set(available)
    ordered = [theme.name for theme in CUSTOM_THEMES if theme.name in known]
    ordered += [name for name in _BUILTIN_ORDER if name in known and name not in ordered]
    seen = set(ordered)
    ordered += sorted(name for name in known if name not in seen)
    return ordered


def preferences_path() -> Path:
    """Where the chosen theme is remembered.

    Honours ``TINYDIFFUSION_CONFIG_DIR`` first — which is what the tests set,
    and what anyone keeping a project self-contained would reach for — then
    ``XDG_CONFIG_HOME``, then the home directory.

    Returns:
        The file, which need not exist.
    """
    if override := os.environ.get("TINYDIFFUSION_CONFIG_DIR"):
        return Path(override) / "tui.json"
    if xdg := os.environ.get("XDG_CONFIG_HOME"):
        return Path(xdg) / "tinydiffusion" / "tui.json"
    return Path.home() / ".config" / "tinydiffusion" / "tui.json"


def load_preferred_theme() -> str | None:
    """Read back the theme last chosen.

    Returns:
        The name, or None if nothing has been saved or the file cannot be read
        as the JSON object it should be. A preference is a convenience, and
        failing to recall one is not worth reporting, let alone raising.
    """
    try:
        data = json.loads(preferences_path().read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if isinstance(data, dict) and isinstance(name := data.get("theme"), str):
        return name
    return None


def save_preferred_theme(name: str) -> None:
    """Remember `name` for the next run.

    Args:
        name: the theme to store.
    """
    path = preferences_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"theme": name}, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
