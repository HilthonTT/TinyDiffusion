import asyncio
from pathlib import Path

import pytest

from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.observer import BatchProgress, TrainObserver, TrainPlan
from tinydiffusion.tui.preview import HALF_BLOCK, half_block_rows

# Everything below drives the app itself, so the whole module needs the extra.
# The message a missing Textual produces is covered in test_cli.py, which is
# where it has to keep working on an install that does not have it.
pytest.importorskip("textual", reason="needs the 'tui' extra")

from tinydiffusion.tui.app import (
    QuartileBars,
    SamplePreview,
    Series,
    TinyDiffusionApp,
    TuiObserver,
    duration,
    epoch_summary,
    run_eta,
    two_columns,
)

# --- the image preview -----------------------------------------------------


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
    # Two pixels to a row: a 64x32 image scaled to 32 wide is 16 pixels tall,
    # which is 8 rows -- half as many rows as columns for a 2:1 image.
    assert len(rows) == 8


def test_the_box_is_never_exceeded(grid):
    rows = half_block_rows(grid, max_width=8, max_height=2)
    assert len(rows) <= 2
    assert all(len(row) <= 8 for row in rows)


def test_a_small_image_is_not_blown_up(grid):
    # Scaling never goes above 1: enlarging a 32px grid to fill a wide terminal
    # would show interpolation, not the model's output.
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
    # An odd pixel height would leave the last cell with nothing underneath it.
    rows = half_block_rows(grid, max_width=15, max_height=15)
    assert all(len(cell) == 2 for row in rows for cell in row)


def test_an_unreadable_file_is_an_oserror(tmp_path):
    broken = tmp_path / "not-an-image.png"
    broken.write_text("nope")
    with pytest.raises(OSError):
        half_block_rows(broken)


# --- the small pure pieces -------------------------------------------------


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
    # The newest are the ones kept.
    assert series.values[-1] == float(MAX_POINTS + 49)


def test_a_single_point_is_doubled_for_the_sparkline():
    # One point has no range to scale against and draws as a blank; two draw
    # the flat line that is the honest picture of one measurement.
    series = Series()
    series.add(1.5)
    assert series.for_sparkline() == [1.5, 1.5]


def test_an_empty_series_still_gives_the_sparkline_something():
    assert Series().for_sparkline() == [0.0]


def test_the_eta_extrapolates_from_the_epoch_so_far():
    progress = BatchProgress(
        epoch=0, num_epochs=4, batch=4, num_batches=10, loss=0.5, images=50, seconds=10.0
    )
    # Half an epoch in 10s, so an epoch is 20s and there are 3.5 left.
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
    # The values line up under each other.
    assert lines[0].index("1") == lines[1].index("2")


# --- the observer ----------------------------------------------------------


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
    # call_from_thread blocks the training thread until the UI has rendered, so
    # an unthrottled observer would make training wait on the screen. Only the
    # first of a burst gets through.
    assert len(app.calls) == 1


def test_a_stop_is_visible_from_the_training_thread():
    observer = TuiObserver(FakeApp())
    assert not observer.stop_requested()
    observer.request_stop()
    assert observer.stop_requested()


def test_an_app_that_has_gone_does_not_take_training_with_it():
    # The display can be closed while a run is still going. Reporting into the
    # wreckage must not raise on the training thread.
    observer = TuiObserver(FakeApp(fail=True))
    observer.on_message("into the void")
    observer.on_epoch(0, {})


# --- the app --------------------------------------------------------------


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
        for selector in ("#plan", "#stats", "#train-spark", "#val-spark", "#log", "#epoch-bar"):
            assert app.query_one(selector)
        assert app.query_one(QuartileBars)
        assert app.query_one(SamplePreview)
        # Nothing is running until it is asked to be.
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
        assert app.query_one("#train-spark").data == [0.9, 0.7]

        bars = text_of(app.query_one(QuartileBars))
        assert "q0" in bars and "q3" in bars
        # One scale across all four, so the bars compare with each other: the
        # largest fills the width and the smallest does not.
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
        # The next epoch writes another grid; this is not worth a crash.
        assert "could not read" in text_of(app.query_one("#preview"))

    drive(TinyDiffusionApp(cfg), steps)


def test_a_run_message_is_not_read_as_markup(cfg):
    async def steps(pilot):
        app = pilot.app
        # Training prints tensor shapes and paths; a bare [dim] in one of them
        # must not be swallowed as a tag, or eat the rest of the line.
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


def test_the_theme_toggles(cfg):
    async def steps(pilot):
        app = pilot.app
        first = app.theme
        await pilot.press("d")
        assert app.theme != first

    drive(TinyDiffusionApp(cfg), steps)


def test_a_failing_run_is_reported_rather_than_swallowed(cfg):
    async def steps(pilot):
        app = pilot.app
        from tinydiffusion.tui.app import TrainingEnded

        app.running = True
        app.post_message(TrainingEnded(RuntimeError("CUDA out of memory")))
        await pilot.pause()
        # A run that dies on the first batch is exactly when a display earns
        # its keep; a blank screen would be the worst possible answer.
        assert not app.running
        # RichLog keeps rendered strips rather than the strings it was handed.
        written = "\n".join(strip.text for strip in app.query_one("#log").lines)
        assert "CUDA out of memory" in written

    drive(TinyDiffusionApp(cfg), steps)
