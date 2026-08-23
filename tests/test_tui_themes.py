"""The theme list, and the one file the dashboard keeps between runs."""

import pytest

pytest.importorskip("textual", reason="needs the 'tui' extra")

from tinydiffusion.tui.themes import (
    CUSTOM_THEMES,
    DEFAULT_THEME,
    cycle_order,
    load_preferred_theme,
    preferences_path,
    save_preferred_theme,
)


@pytest.fixture(autouse=True)
def config_dir(tmp_path, monkeypatch):
    """Point the preferences file at the test's own directory."""
    monkeypatch.setenv("TINYDIFFUSION_CONFIG_DIR", str(tmp_path / "config"))
    return tmp_path / "config"


# --- the list --------------------------------------------------------------


def test_the_themes_are_all_named_differently():
    names = [theme.name for theme in CUSTOM_THEMES]
    assert len(set(names)) == len(names)


def test_the_default_is_one_of_them():
    assert DEFAULT_THEME in {theme.name for theme in CUSTOM_THEMES}


def test_ours_lead_the_cycle():
    order = cycle_order(["nord", "textual-dark", *[theme.name for theme in CUSTOM_THEMES]])
    assert order[0] == CUSTOM_THEMES[0].name


def test_a_theme_nobody_listed_still_appears():
    # Registered later, or by Textual in a version we have not seen: dropping
    # it silently would be the one outcome worth ruling out.
    order = cycle_order(["tinydiffusion", "some-new-theme"])
    assert "some-new-theme" in order


def test_the_cycle_never_repeats_a_theme():
    order = cycle_order(["nord", "nord", "tinydiffusion"])
    assert len(order) == len(set(order))


def test_a_theme_the_app_does_not_have_is_left_out():
    assert cycle_order(["nord"]) == ["nord"]


# --- remembering -----------------------------------------------------------


def test_a_chosen_theme_comes_back():
    save_preferred_theme("dracula")
    assert load_preferred_theme() == "dracula"


def test_nothing_saved_reads_as_nothing():
    assert load_preferred_theme() is None


def test_an_unreadable_preferences_file_is_not_an_error():
    # A preference is a convenience; failing to recall one is not worth
    # reporting, let alone raising in front of a training run.
    path = preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_preferred_theme() is None


def test_a_preferences_file_of_the_wrong_shape_reads_as_nothing():
    path = preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"theme": 3}', encoding="utf-8")
    assert load_preferred_theme() is None


def test_a_directory_that_cannot_be_written_is_survived(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr("pathlib.Path.mkdir", refuse)
    save_preferred_theme("nord")  # must not raise


def test_the_config_directory_can_be_pointed_somewhere_else(tmp_path, monkeypatch):
    monkeypatch.delenv("TINYDIFFUSION_CONFIG_DIR")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert preferences_path().parent == tmp_path / "xdg" / "tinydiffusion"
