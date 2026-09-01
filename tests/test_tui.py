import asyncio
from pathlib import Path

import pytest

from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.observer import BatchProgress, TrainObserver, TrainPlan
from tinydiffusion.tui.preview import HALF_BLOCK, half_block_rows

pytest.importorskip("textual", reason="needs the 'tui' extra")

from tinydiffusion.tui.app import (
    LossChart,
    QuartileBars,
    SamplePreview,
    Series,
    StatTile,
    StatusBar,
    TinyDiffusionApp,
    TrainingEnded,
    TuiObserver,
    duration,
    epoch_summary,
    run_eta,
    two_columns,
)
from tinydiffusion.tui.screens import HelpScreen, ThemeScreen
from tinydiffusion.tui.themes import (
    CUSTOM_THEMES,
    DEFAULT_THEME,
    load_preferred_theme,
    save_preferred_theme,
)
from tinydiffusion.tui.widgets import resolve_colour


@pytest.fixture(autouse=True)
def config_dir(tmp_path, monkeypatch):
    """Keep the remembered theme inside the test's own directory.

    The dashboard writes the chosen theme to the user's config directory, and a
    suite that reached into a real home would both leak into it and be steered
    by whatever was already there.
    """
    monkeypatch.setenv("TINYDIFFUSION_CONFIG_DIR", str(tmp_path / "config"))
    return tmp_path / "config"


@pytest.fixture
def grid(tmp_path):
    """A 64x32 image, standing in for a sample grid."""
    from PIL import Image

    path = tmp_path / "sample_0001.png"
    Image.new("L", (64, 32), color=128).save(path)
    return path


def test_a_grid_is_reduced_to_cells_that_fit_the_box(grid):
    rows = half_block_rows(grid, max_width=32, max_height=32)
    assert len(rows[0]) == 32
    assert len(rows) == 8


def test_the_box_is_never_exceeded(grid):
    rows = half_block_rows(grid, max_width=8, max_height=2)
    assert len(rows) <= 2
    assert all(len(row) <= 8 for row in rows)


def test_a_small_image_is_not_blown_up(grid):
    rows = half_block_rows(grid, max_width=400, max_height=400)
    assert len(rows[0]) == 64


def test_a_box_with_no_room_renders_nothing(grid):
    assert half_block_rows(grid, max_width=0, max_height=10) == []
    assert half_block_rows(grid, max_width=10, max_height=0) == []


def test_greyscale_comes_back_as_colour_triples(grid):
    (top, bottom), *_ = half_block_rows(grid, max_width=8, max_height=8)[0]
    assert top == (128, 128, 128)
    assert bottom == (128, 128, 128)


def test_every_row_holds_a_top_and_a_bottom(grid):
    rows = half_block_rows(grid, max_width=15, max_height=15)
    assert all(len(cell) == 2 for row in rows for cell in row)


def test_an_unreadable_file_is_an_oserror(tmp_path):
    broken = tmp_path / "not-an-image.png"
    broken.write_text("nope")
    with pytest.raises(OSError):
        half_block_rows(broken)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0s"), (9, "9s"), (59, "59s"), (60, "1m00s"), (200, "3m20s"), (3720, "1h02m"), (-5, "0s")],
)
def test_durations_coarsen_as_they_grow(seconds, expected):
    assert duration(seconds) == expected


def test_a_series_is_bounded():
    from tinydiffusion.tui.app import MAX_POINTS

    series = Series()
    for value in range(MAX_POINTS + 50):
        series.add(float(value))
    assert len(series.values) == MAX_POINTS
    assert series.values[-1] == float(MAX_POINTS + 49)


def test_the_eta_extrapolates_from_the_epoch_so_far():
    progress = BatchProgress(
        epoch=0, num_epochs=4, batch=4, num_batches=10, loss=0.5, images=50, seconds=10.0
    )
    assert run_eta(progress) == pytest.approx(70.0)


def test_the_eta_holds_off_until_there_is_something_to_go_on():
    nothing_yet = BatchProgress(
        epoch=0, num_epochs=4, batch=0, num_batches=0, loss=None, images=0, seconds=0.0
    )
    assert run_eta(nothing_yet) is None


def test_throughput_is_zero_before_any_time_has_passed():
    progress = BatchProgress(
        epoch=0, num_epochs=1, batch=0, num_batches=4, loss=None, images=8, seconds=0.0
    )
    assert progress.images_per_second == 0.0


def test_an_epoch_summary_names_what_it_has():
    text = epoch_summary({"train/loss": 0.5, "val/loss": 0.25})
    assert "loss 0.5" in text
    assert "val 0.25" in text


def test_an_epoch_with_nothing_summarises_as_such():
    assert epoch_summary({}) == "no metrics"


def test_two_columns_aligns_and_does_not_trail_a_blank_line():
    rendered = two_columns([("a", "1"), ("longer", "2")])
    lines = rendered.plain.splitlines()
    assert len(lines) == 2
    assert not rendered.plain.endswith("\n")
    assert lines[0].index("1") == lines[1].index("2")


class FakeApp:
    """Stands in for the app: records what the observer asked it to run."""

    def __init__(self, fail=False):
        self.calls: list[tuple[str, tuple]] = []
        self.fail = fail

    def call_from_thread(self, fn, *args):
        if self.fail:
            raise RuntimeError("app is shutting down")
        self.calls.append((fn.__name__, args))

    def apply_plan(self, plan): ...
    def apply_message(self, text): ...
    def apply_progress(self, progress): ...
    def apply_epoch(self, step, metrics): ...
    def apply_sample(self, path): ...


def progress_at(batch: int) -> BatchProgress:
    return BatchProgress(
        epoch=0, num_epochs=1, batch=batch, num_batches=100, loss=0.5, images=8, seconds=1.0
    )


def test_the_observer_satisfies_the_protocol():
    assert isinstance(TuiObserver(FakeApp()), TrainObserver)


def test_the_observer_marshals_every_event_to_the_app():
    app = FakeApp()
    observer = TuiObserver(app)
    observer.on_message("hello")
    observer.on_epoch(0, {"train/loss": 1.0})
    observer.on_sample(Path("grid.png"))
    assert [name for name, _ in app.calls] == ["apply_message", "apply_epoch", "apply_sample"]


def test_batch_reports_are_throttled():
    app = FakeApp()
    observer = TuiObserver(app)
    for batch in range(50):
        observer.on_batch(progress_at(batch))
    assert len(app.calls) == 1


def test_a_stop_is_visible_from_the_training_thread():
    observer = TuiObserver(FakeApp())
    assert not observer.stop_requested()
    observer.request_stop()
    assert observer.stop_requested()


def test_an_app_that_has_gone_does_not_take_training_with_it():
    observer = TuiObserver(FakeApp(fail=True))
    observer.on_message("into the void")
    observer.on_epoch(0, {})


@pytest.fixture
def cfg(tmp_path):
    return TrainConfig(
        num_epochs=3,
        device="cpu",
        out_dir=tmp_path / "contents",
        ckpt_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        log_console=False,
    )


def text_of(widget) -> str:
    """A widget's rendered text, whether it holds a plain string or Content."""
    content = widget.content
    return getattr(content, "plain", content)


def drive(app, steps):
    """Run `steps(pilot)` against a mounted app, without needing pytest-asyncio."""

    async def go():
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await steps(pilot)

    asyncio.run(go())


def plan_for(cfg) -> TrainPlan:
    return TrainPlan(
        dataset="mnist",
        image_size=32,
        channels=1,
        device="cpu",
        device_description="cpu",
        parameters=6_950_000,
        precision="amp off",
        num_classes=10,
        start_epoch=0,
        num_epochs=cfg.num_epochs,
        steps_per_epoch=468,
        batch_size=128,
        grad_accum=1,
        validation_images=512,
        log_dir=cfg.log_dir,
    )


def test_the_app_mounts_with_every_panel(cfg):
    async def steps(pilot):
        app = pilot.app
        for selector in ("#plan", "#stats", "#log", "#epoch-bar", "#status-bar"):
            assert app.query_one(selector)
        assert app.query_one(QuartileBars)
        assert app.query_one(SamplePreview)
        assert app.query_one(LossChart)
        assert len(app.query(StatTile)) == 5
        assert not app.running

    drive(TinyDiffusionApp(cfg), steps)


def test_the_plan_fills_in_the_run_panel(cfg):
    async def steps(pilot):
        app = pilot.app
        app.apply_plan(plan_for(cfg))
        await pilot.pause()
        rendered = text_of(app.query_one("#plan"))
        assert "mnist" in rendered
        assert "6.95M" in rendered
        assert "10 classes" in rendered
        assert "512 images" in rendered
        assert "mnist" in app.sub_title

    drive(TinyDiffusionApp(cfg), steps)


def test_an_unconditional_plan_says_so(cfg):
    async def steps(pilot):
        app = pilot.app
        import dataclasses

        app.apply_plan(dataclasses.replace(plan_for(cfg), num_classes=None, validation_images=0))
        await pilot.pause()
        rendered = text_of(app.query_one("#plan"))
        assert "unconditional" in rendered
        assert "off" in rendered

    drive(TinyDiffusionApp(cfg), steps)


def test_progress_moves_the_bars_and_the_stats(cfg):
    async def steps(pilot):
        app = pilot.app
        app.apply_plan(plan_for(cfg))
        app.apply_progress(
            BatchProgress(
                epoch=1,
                num_epochs=3,
                batch=233,
                num_batches=468,
                loss=0.1234,
                images=29_952,
                seconds=30.0,
            )
        )
        await pilot.pause()
        assert text_of(app.query_one("#epoch-label")) == "epoch 2/3"
        assert text_of(app.query_one("#batch-label")) == "batch 234/468"
        stats = text_of(app.query_one("#stats"))
        assert "0.12340" in stats
        assert "img/s" in stats
        assert "eta" in stats

    drive(TinyDiffusionApp(cfg), steps)


def test_epoch_metrics_reach_the_charts_and_the_quartiles(cfg):
    async def steps(pilot):
        app = pilot.app
        app.apply_epoch(
            0,
            {
                "train/loss": 0.9,
                "val/loss": 0.8,
                "val/best_loss": 0.8,
                "train/lr": 2e-4,
                "train/loss_q0": 1.0,
                "train/loss_q1": 0.5,
                "train/loss_q2": 0.25,
                "train/loss_q3": 0.125,
            },
        )
        app.apply_epoch(1, {"train/loss": 0.7, "val/loss": 0.6})
        await pilot.pause()

        assert app.train_loss.values == [0.9, 0.7]
        assert app.val_loss.values == [0.8, 0.6]
        chart = text_of(app.query_one(LossChart))
        assert "train 0.7" in chart
        assert "val 0.6" in chart

        bars = text_of(app.query_one(QuartileBars))
        assert "t 0-25%" in bars and "t 75-100%" in bars
        lines = bars.splitlines()
        assert lines[0].count("█") > lines[3].count("█")

        stats = text_of(app.query_one("#stats"))
        assert "best val" in stats
        assert "lr" in stats

    drive(TinyDiffusionApp(cfg), steps)


def test_an_epoch_that_logged_no_quartiles_leaves_the_bars_alone(cfg):
    async def steps(pilot):
        app = pilot.app
        before = text_of(app.query_one(QuartileBars))
        app.apply_epoch(0, {"train/loss": 0.5})
        await pilot.pause()
        assert text_of(app.query_one(QuartileBars)) == before

    drive(TinyDiffusionApp(cfg), steps)


def test_a_sample_grid_is_drawn_as_half_blocks(cfg, grid):
    async def steps(pilot):
        app = pilot.app
        app.apply_sample(grid)
        await pilot.pause()
        rendered = text_of(app.query_one("#preview"))
        assert HALF_BLOCK in rendered

    drive(TinyDiffusionApp(cfg), steps)


def test_a_grid_that_cannot_be_read_is_reported_not_raised(cfg, tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_text("nope")

    async def steps(pilot):
        app = pilot.app
        app.apply_sample(broken)
        await pilot.pause()
        assert "could not read" in text_of(app.query_one("#preview"))

    drive(TinyDiffusionApp(cfg), steps)


def test_a_run_message_is_not_read_as_markup(cfg):
    async def steps(pilot):
        app = pilot.app
        app.apply_message("shape [4, 1, 16, 16] on /tmp/x")
        await pilot.pause()

    drive(TinyDiffusionApp(cfg), steps)


def test_stopping_when_nothing_runs_is_refused_not_crashed(cfg):
    async def steps(pilot):
        app = pilot.app
        await pilot.press("x")
        assert not app.running

    drive(TinyDiffusionApp(cfg), steps)


def test_the_log_pane_toggles(cfg):
    async def steps(pilot):
        app = pilot.app
        panel = app.query_one("#log-panel")
        assert panel.display
        await pilot.press("l")
        assert not panel.display
        await pilot.press("l")
        assert panel.display

    drive(TinyDiffusionApp(cfg), steps)


def test_the_theme_cycles_both_ways_and_is_remembered(cfg):
    async def steps(pilot):
        app = pilot.app
        first = app.theme
        await pilot.press("d")
        assert app.theme != first
        assert load_preferred_theme() == app.theme
        await pilot.press("D")
        assert app.theme == first

    drive(TinyDiffusionApp(cfg), steps)


def test_the_dashboard_opens_in_the_theme_last_chosen(cfg):
    save_preferred_theme("gruvbox")

    async def steps(pilot):
        assert pilot.app.theme == "gruvbox"

    drive(TinyDiffusionApp(cfg), steps)


def test_a_remembered_theme_that_no_longer_exists_falls_back(cfg):
    save_preferred_theme("a-theme-that-was-removed")

    async def steps(pilot):
        assert pilot.app.theme == DEFAULT_THEME

    drive(TinyDiffusionApp(cfg), steps)


def test_every_custom_theme_is_registered(cfg):
    async def steps(pilot):
        for theme in CUSTOM_THEMES:
            assert theme.name in pilot.app.available_themes

    drive(TinyDiffusionApp(cfg), steps)


def test_the_theme_picker_previews_and_can_be_cancelled(cfg):
    async def steps(pilot):
        app = pilot.app
        before = app.theme
        await pilot.press("t")
        await pilot.pause()
        assert isinstance(app.screen, ThemeScreen)
        await pilot.press("down")
        await pilot.pause()
        assert app.theme != before
        await pilot.press("escape")
        await pilot.pause()
        assert app.theme == before

    drive(TinyDiffusionApp(cfg), steps)


def test_the_theme_picker_keeps_what_was_selected(cfg):
    async def steps(pilot):
        app = pilot.app
        await pilot.press("t")
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, ThemeScreen)
        assert load_preferred_theme() == app.theme

    drive(TinyDiffusionApp(cfg), steps)


def test_the_help_lists_the_keys(cfg):
    async def steps(pilot):
        app = pilot.app
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, HelpScreen)

    drive(TinyDiffusionApp(cfg), steps)


def test_focus_mode_clears_the_screen_of_all_but_the_charts(cfg):
    async def steps(pilot):
        app = pilot.app
        assert app.query_one("#sidebar").display
        await pilot.press("f")
        await pilot.pause()
        assert not app.query_one("#sidebar").display
        await pilot.press("f")
        await pilot.pause()
        assert app.query_one("#sidebar").display

    drive(TinyDiffusionApp(cfg), steps)


def test_the_log_can_be_emptied(cfg):
    async def steps(pilot):
        app = pilot.app
        assert app.query_one("#log").lines
        await pilot.press("c")
        await pilot.pause()
        assert not app.query_one("#log").lines

    drive(TinyDiffusionApp(cfg), steps)


def test_the_status_bar_follows_the_run(cfg):
    async def steps(pilot):
        app = pilot.app
        status = app.query_one(StatusBar)
        assert status.state == "idle"
        app.running = True
        app.observer = TuiObserver(app)
        await pilot.press("x")
        assert status.state == "stopping"
        app.post_message(TrainingEnded(None))
        await pilot.pause()
        assert status.state == "done"

    drive(TinyDiffusionApp(cfg), steps)


def test_a_failed_run_says_so_in_the_status_bar(cfg):
    async def steps(pilot):
        app = pilot.app
        app.running = True
        app.post_message(TrainingEnded(RuntimeError("boom")))
        await pilot.pause()
        assert app.query_one(StatusBar).state == "failed"

    drive(TinyDiffusionApp(cfg), steps)


def test_a_narrow_terminal_drops_the_tiles_rather_than_squashing_them(cfg):
    async def go():
        app = TinyDiffusionApp(cfg)
        async with app.run_test(size=(80, 30)) as pilot:
            await pilot.pause()
            assert not app.query_one("#tiles").display
            assert app.query_one("#sidebar").display

    asyncio.run(go())


def test_a_very_narrow_terminal_gives_the_room_to_the_charts(cfg):
    async def go():
        app = TinyDiffusionApp(cfg)
        async with app.run_test(size=(60, 24)) as pilot:
            await pilot.pause()
            assert not app.query_one("#sidebar").display
            assert app.query_one(LossChart).display

    asyncio.run(go())


def test_a_failing_run_is_reported_rather_than_swallowed(cfg):
    async def steps(pilot):
        app = pilot.app
        from tinydiffusion.tui.app import TrainingEnded

        app.running = True
        app.post_message(TrainingEnded(RuntimeError("CUDA out of memory")))
        await pilot.pause()
        assert not app.running
        written = "\n".join(strip.text for strip in app.query_one("#log").lines)
        assert "CUDA out of memory" in written

    drive(TinyDiffusionApp(cfg), steps)


@pytest.fixture
def no_training(monkeypatch):
    """Let the start/stop keys be exercised without actually training.

    Everything below is about what the dashboard does around a run, and a real
    worker would download MNIST to find out.
    """
    monkeypatch.setattr(TinyDiffusionApp, "train_worker", lambda self: None)


def test_restarting_an_idle_dashboard_simply_starts(cfg, no_training):
    async def steps(pilot):
        await pilot.press("r")
        assert pilot.app.running

    drive(TinyDiffusionApp(cfg), steps)


def test_restarting_a_running_dashboard_waits_for_the_worker_to_leave(cfg, no_training):
    async def steps(pilot):
        app = pilot.app
        await pilot.press("s")
        assert app.running
        observer = app.observer
        await pilot.press("r")
        assert observer.stop_requested()
        assert not app.running or app.query_one(StatusBar).state == "stopping"
        app.post_message(TrainingEnded(None))
        await pilot.pause()
        assert app.running

    drive(TinyDiffusionApp(cfg), steps)


def test_a_failed_run_is_not_restarted(cfg, no_training):
    async def steps(pilot):
        app = pilot.app
        await pilot.press("s")
        await pilot.press("r")
        app.post_message(TrainingEnded(RuntimeError("CUDA out of memory")))
        await pilot.pause()
        assert not app.running

    drive(TinyDiffusionApp(cfg), steps)


def test_a_screenshot_lands_beside_the_run(cfg):
    async def steps(pilot):
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert list(cfg.log_dir.glob("*.svg"))

    drive(TinyDiffusionApp(cfg), steps)


def test_a_plain_colour_comes_back_as_itself():
    assert resolve_colour("#88C0D0", {}).lower() == "#88c0d0"


def test_a_faded_colour_is_blended_towards_what_is_behind_it():
    faded = resolve_colour("auto 50%", {"foreground": "#FFFFFF", "surface": "#000000"})
    assert faded.lower() in {"#7f7f7f", "#808080"}


def test_a_faded_colour_with_nothing_behind_it_keeps_its_strength():
    assert resolve_colour("auto 50%", {"foreground": "#FFFFFF"}).lower() == "#ffffff"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ansi_blue", "blue"),
        ("ansi_bright_red", "bright_red"),
        ("ansi_default", "default"),
        ("ansi_default 50%", "default"),
        ("transparent", "default"),
    ],
)
def test_the_terminals_own_colours_are_spelt_the_way_rich_spells_them(value, expected):
    assert resolve_colour(value, {}) == expected


def test_a_colour_that_cannot_be_read_falls_back_rather_than_raising():
    assert resolve_colour("not-a-colour", {}) == "#808080"
    assert resolve_colour("", {}) == "#808080"


def test_every_theme_renders_every_widget(cfg, grid):
    """Each theme, applied and drawn, with numbers in every panel.

    The ANSI themes broke the dashboard once: their colours are named
    `ansi_blue` and `ansi_default`, which Rich cannot parse, and the exception
    landed while someone was scrolling the picker. Walking the whole list is
    cheap and this is exactly the kind of bug it catches.
    """

    async def steps(pilot):
        app = pilot.app
        app.apply_epoch(
            0,
            {
                "train/loss": 0.9,
                "val/loss": 0.8,
                "train/loss_q0": 1.0,
                "train/loss_q1": 0.5,
                "train/loss_q2": 0.25,
                "train/loss_q3": 0.125,
            },
        )
        app.apply_sample(grid)
        app.query_one(StatusBar).detail = "mnist · cuda"
        app.query_one(StatusBar).note = "epoch 1/3"
        for name in app.available_themes:
            app.theme = name
            await pilot.pause()
            assert text_of(app.query_one(LossChart))
            assert text_of(app.query_one(QuartileBars))
            assert text_of(app.query_one("#status-bar"))

    drive(TinyDiffusionApp(cfg), steps)


def test_the_theme_picker_survives_being_scrolled_through(cfg):
    async def steps(pilot):
        app = pilot.app
        await pilot.press("t")
        await pilot.pause()
        for _ in range(len(app.available_themes) + 2):
            await pilot.press("down")
            await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    drive(TinyDiffusionApp(cfg), steps)
