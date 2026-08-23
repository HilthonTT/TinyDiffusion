"""The training dashboard: a Textual app wrapped round :func:`train`.

Training is a blocking loop that wants a thread of its own, and a Textual app
is an asyncio event loop that must never be blocked. The two are joined at
exactly two points, and everything else here follows from them:

* :class:`TuiObserver` implements
  :class:`~tinydiffusion.training.observer.TrainObserver` and is called from
  the training thread. It touches no widget; it hands each event to the app
  with ``call_from_thread`` and lets the event loop apply it.
* Stopping is a :class:`threading.Event`. It is the one thing the training
  thread reads rather than writes, so it needs no marshalling at all — and it
  is why a run can be ended from a keypress without the Ctrl+C prompt, which
  has nobody to answer it while a display owns the terminal.

The screen itself is assembled from :mod:`tinydiffusion.tui.widgets`, which
knows how to draw and nothing else. What is worth showing, and when, is decided
here.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, ProgressBar, RichLog, Static

from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.observer import BatchProgress, TrainPlan
from tinydiffusion.training.train import train
from tinydiffusion.tui.screens import HelpScreen, ThemeScreen
from tinydiffusion.tui.themes import (
    CUSTOM_THEMES,
    DEFAULT_THEME,
    cycle_order,
    load_preferred_theme,
    save_preferred_theme,
)
from tinydiffusion.tui.widgets import (
    LossChart,
    QuartileBars,
    SamplePreview,
    StatTile,
    StatusBar,
)

if TYPE_CHECKING:
    from textual.binding import BindingType
    from textual.screen import Screen

__all__ = [
    "LossChart",
    "Panel",
    "QuartileBars",
    "SamplePreview",
    "Series",
    "StatTile",
    "StatusBar",
    "TinyDiffusionApp",
    "TrainingEnded",
    "TuiObserver",
]

UI_INTERVAL = 0.05
"""Seconds between UI updates driven from the training thread.

``call_from_thread`` blocks the caller until the event loop has run the
callback, so an unthrottled observer would have training wait on rendering. The
batch callback already fires only once per
:data:`~tinydiffusion.training.train.DRAIN_EVERY` batches; this bounds it again
in time, for the small model whose batches go by faster than a screen refresh.
"""

MAX_POINTS = 240
"""Epochs kept per chart series. A chart cannot show more than its width."""

NARROW = 100
"""Columns below which the tiles are dropped and the sidebar tightened."""

VERY_NARROW = 72
"""Columns below which the sidebar goes entirely and the charts take the room."""


class TrainingEnded(Message):
    """Posted when the training worker leaves, however it left.

    Args:
        error: the exception that ended the run, or None if it ran to the last
            epoch or stopped because it was asked to.
    """

    def __init__(self, error: BaseException | None) -> None:
        super().__init__()
        self.error = error


@dataclass
class Series:
    """One metric's history, bounded so a long run cannot grow without limit.

    Attributes:
        values: the points, oldest first.
    """

    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        """Append a point, dropping the oldest once :data:`MAX_POINTS` is reached.

        Args:
            value: the new point.
        """
        self.values.append(value)
        if len(self.values) > MAX_POINTS:
            del self.values[0]


class TuiObserver:
    """Carries training events from the worker thread to the app.

    Every method here runs on the training thread. None of them touches a
    widget: each hands the work to the event loop, which is the only place
    Textual allows a UI to be changed from.

    Args:
        app: the app to deliver events to.
    """

    def __init__(self, app: TinyDiffusionApp) -> None:
        self._app = app
        self._stop = threading.Event()
        self._last_ui = 0.0

    def request_stop(self) -> None:
        """Ask the run to end at the next batch boundary. Safe from any thread."""
        self._stop.set()

    def stop_requested(self) -> bool:
        """Whether a stop has been asked for.

        Returns:
            True once :meth:`request_stop` has been called.
        """
        return self._stop.is_set()

    def _deliver(self, name: str, *args: object) -> None:
        """Run one of the app's handlers on the event loop.

        Args:
            name: the app method to call.
            *args: its arguments.
        """
        # The app may be shutting down, its loop already gone. A run that
        # outlives the display has nothing useful to tell it, and that is not a
        # reason to take the training thread down too.
        with contextlib.suppress(RuntimeError):
            self._app.call_from_thread(getattr(self._app, name), *args)

    def on_plan(self, plan: TrainPlan) -> None:
        """Hand the resolved plan to the app.

        Args:
            plan: what the run settled on.
        """
        self._deliver("apply_plan", plan)

    def on_message(self, text: str) -> None:
        """Hand one of the run's lines to the log pane.

        Args:
            text: the message.
        """
        self._deliver("apply_message", text)

    def on_batch(self, progress: BatchProgress) -> None:
        """Hand batch progress over, at most every :data:`UI_INTERVAL`.

        Args:
            progress: where the run has got to.
        """
        now = time.monotonic()
        if now - self._last_ui < UI_INTERVAL:
            return
        self._last_ui = now
        self._deliver("apply_progress", progress)

    def on_epoch(self, step: int, metrics: Mapping[str, float]) -> None:
        """Hand an epoch's metrics over. Never throttled: there are few of them.

        Args:
            step: the epoch index.
            metrics: metric name to value.
        """
        self._deliver("apply_epoch", step, dict(metrics))

    def on_sample(self, path: Path) -> None:
        """Hand over a freshly written sample grid.

        Args:
            path: the PNG just saved.
        """
        self._deliver("apply_sample", path)


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
        id: str | None = None,  # noqa: A002 - Textual's own parameter name
        classes: str | None = None,
    ) -> None:
        super().__init__(*children, id=id, classes=classes)  # type: ignore[arg-type]
        self.border_title = title


class TinyDiffusionApp(App[None]):
    """Train a diffusion model, and watch it happen.

    Args:
        cfg: the configuration a run will use.
        resume: a checkpoint to continue from, or None to start fresh.
        autostart: begin training as soon as the app is ready, rather than
            waiting for the key.
    """

    TITLE = "TinyDiffusion"

    CSS = """
    Screen { background: $background; }

    #columns { height: 1fr; }

    /* Scrollable rather than fixed: the panels are sized by their content, and
       a narrow terminal or a long device name pushes the last of them past the
       bottom. Clipping it silently is the one outcome worth ruling out. */
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

    /* Both 1fr, so whatever the terminal has left is split between the curve
       and the pictures rather than one of them being clipped by a fixed size. */
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

    /* Below ~100 columns the tiles cannot hold five readable numbers side by
       side, and the sidebar's stats panel already carries every one of them. */
    Screen.-narrow #tiles { display: none; }
    Screen.-narrow #sidebar { width: 30; }
    Screen.-narrow #preview-panel { width: 50%; }
    Screen.-narrow #log-panel { height: 8; }
    Screen.-tiny #sidebar { display: none; }
    Screen.-tiny #log-panel { height: 6; }

    /* Focus mode: everything that is not a picture gets out of the way. */
    Screen.-focus #sidebar { display: none; }
    Screen.-focus #tiles { display: none; }
    Screen.-focus #log-panel { display: none; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("s", "start", "Start"),
        Binding("x", "stop", "Stop"),
        Binding("r", "restart", "Restart", show=False),
        Binding("l", "toggle_log", "Log"),
        Binding("f", "toggle_focus", "Focus"),
        Binding("c", "clear_log", "Clear", show=False),
        Binding("t", "pick_theme", "Theme"),
        Binding("d", "cycle_theme", "Next theme", show=False),
        Binding("D", "cycle_theme_back", "Previous theme", show=False),
        Binding("ctrl+s", "snapshot", "Screenshot", show=False),
        Binding("question_mark", "help", "Help", key_display="?"),
        Binding("q", "quit", "Quit"),
    ]

    running: reactive[bool] = reactive(False)
    """Whether a training worker is live. Decides what the keys are allowed to do."""

    def __init__(
        self,
        cfg: TrainConfig,
        resume: Path | None = None,
        *,
        autostart: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.resume = resume
        self.autostart = autostart
        self.observer: TuiObserver | None = None
        self.train_loss = Series()
        self.val_loss = Series()
        self.plan: TrainPlan | None = None
        self._stats: dict[str, str] = {}
        self._started: float | None = None
        self._restart = False

    # ---- layout --------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Build the widget tree.

        Yields:
            The app's widgets.
        """
        yield Header(show_clock=True, icon="◈")
        yield StatusBar()
        with Horizontal(id="columns"):
            with VerticalScroll(id="sidebar"):
                with Panel(title="run", id="run-panel"):
                    yield Static(self.plan_text(), id="plan")
                with Panel(title="progress", id="progress-panel"):
                    yield Label("epoch -", id="epoch-label")
                    yield ProgressBar(total=100, show_eta=False, id="epoch-bar")
                    yield Label("batch -", id="batch-label")
                    yield ProgressBar(total=100, show_eta=False, id="batch-bar")
                with Panel(title="stats", id="stats-panel"):
                    yield Static("not started", id="stats")
            with Vertical(id="main"):
                with Horizontal(id="tiles"):
                    yield StatTile("loss", "tile-loss", "text-primary")
                    yield StatTile("val", "tile-val", "text-secondary")
                    yield StatTile("img/s", "tile-rate", "text-accent")
                    yield StatTile("eta", "tile-eta", "text-warning")
                    yield StatTile("elapsed", "tile-elapsed", "text-muted")
                with Panel(title="loss per epoch", id="chart-panel"):
                    yield LossChart()
                with Horizontal(id="lower"):
                    with Panel(title="loss by timestep", id="quartile-panel"):
                        yield QuartileBars()
                    with Panel(title="samples", id="preview-panel"):
                        yield SamplePreview()
        with Panel(title="log", id="log-panel"):
            yield RichLog(markup=True, wrap=True, id="log")
        yield Footer()

    def on_mount(self) -> None:
        """Register the themes, say what is loaded, and start if asked to."""
        for theme in CUSTOM_THEMES:
            self.register_theme(theme)
        preferred = load_preferred_theme() or DEFAULT_THEME
        self.theme = preferred if preferred in self.available_themes else DEFAULT_THEME

        self.sub_title = f"{self.cfg.dataset} · {self.cfg.device}"
        self.query_one(StatusBar).detail = f"{self.cfg.dataset} · {self.cfg.device}"
        self.apply_width(self.size.width)
        # Elapsed is the one number that moves without an event to move it.
        self.set_interval(1.0, self.tick)

        self.log_line(f"[dim]logs:[/dim] {self.cfg.log_dir}")
        if self.resume is not None:
            self.log_line(f"[dim]resuming from:[/dim] {self.resume}")
        self.log_line("[dim]press[/dim] [b]s[/b] [dim]to start,[/dim] [b]?[/b] [dim]for keys[/dim]")
        if self.autostart:
            self.action_start()

    def on_resize(self) -> None:
        """Re-lay-out for the terminal's new width."""
        self.apply_width(self.size.width)

    def apply_width(self, width: int) -> None:
        """Choose the layout that fits `width` columns.

        Args:
            width: the terminal width.
        """
        screen = self.screen
        screen.set_class(width < NARROW, "-narrow")
        screen.set_class(width < VERY_NARROW, "-tiny")

    def get_system_commands(self, screen: Screen[Any]) -> Iterable[SystemCommand]:
        """Put the dashboard's own actions in the command palette.

        Args:
            screen: the screen the palette was opened over.

        Yields:
            Textual's commands, then ours — so everything the app can do is
            reachable by typing its name, not only by knowing its key.
        """
        yield from super().get_system_commands(screen)
        yield SystemCommand("Start training", "Begin a run", self.action_start)
        yield SystemCommand("Stop training", "End at the next batch", self.action_stop)
        yield SystemCommand("Restart training", "Stop, then start again", self.action_restart)
        yield SystemCommand("Theme picker", "Preview every theme", self.action_pick_theme)
        yield SystemCommand("Toggle focus mode", "Charts only", self.action_toggle_focus)
        yield SystemCommand("Toggle log", "Show or hide the log", self.action_toggle_log)
        yield SystemCommand("Keys", "Everything the dashboard does", self.action_help)

    # ---- actions -------------------------------------------------------

    def action_start(self) -> None:
        """Begin a training run, unless one is already going."""
        if self.running:
            self.notify("training is already running", severity="warning")
            return
        self.running = True
        self._started = time.monotonic()
        self.observer = TuiObserver(self)
        self.query_one(StatusBar).state = "running"
        self.log_line("[b green]starting[/b green]")
        self.train_worker()

    def action_stop(self) -> None:
        """Ask the run to stop at the next batch boundary, checkpointing as it goes."""
        if not self.running or self.observer is None:
            self.notify("nothing is running", severity="warning")
            return
        self.query_one(StatusBar).state = "stopping"
        self.log_line("[yellow]stopping at the next batch boundary…[/yellow]")
        self.observer.request_stop()

    def action_restart(self) -> None:
        """Stop the current run and begin another once the worker has left.

        Restarting is the commonest thing to want after changing nothing but
        one's mind about a seed or a checkpoint, and doing it by hand means
        watching for the stop to land before pressing start.
        """
        if not self.running:
            self.action_start()
            return
        self._restart = True
        self.action_stop()

    def action_toggle_log(self) -> None:
        """Show or hide the log pane, for when the pictures matter more."""
        panel = self.query_one("#log-panel")
        panel.display = not panel.display

    def action_clear_log(self) -> None:
        """Empty the log pane."""
        self.query_one("#log", RichLog).clear()

    def action_toggle_focus(self) -> None:
        """Hide everything that is not a chart, and put it back again."""
        screen = self.screen
        screen.toggle_class("-focus")
        self.notify("focus mode on" if screen.has_class("-focus") else "focus mode off")

    def action_pick_theme(self) -> None:
        """Open the theme picker."""
        self.push_screen(ThemeScreen())

    def action_help(self) -> None:
        """Open the key list."""
        self.push_screen(HelpScreen())

    def action_cycle_theme(self) -> None:
        """Move to the next theme in the cycle, and remember it."""
        self.step_theme(1)

    def action_cycle_theme_back(self) -> None:
        """Move to the previous theme in the cycle, and remember it."""
        self.step_theme(-1)

    def step_theme(self, step: int) -> None:
        """Walk the theme cycle by `step` places.

        Args:
            step: how far to move, negative to go back.
        """
        order = cycle_order(list(self.available_themes))
        if not order:
            return
        current = order.index(self.theme) if self.theme in order else 0
        name = order[(current + step) % len(order)]
        self.theme = name
        save_preferred_theme(name)
        self.notify(f"theme: {name}", timeout=2)

    def action_snapshot(self) -> None:
        """Write the dashboard to an SVG, for the run that is worth showing someone.

        Saved beside the run's metrics rather than in the working directory, so
        a screenshot belongs to the run it is of. A directory that does not
        exist yet — nothing has been trained — is made rather than raised over.
        """
        try:
            self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
            saved = self.save_screenshot(path=str(self.cfg.log_dir))
        except OSError as exc:
            self.notify(f"could not save a screenshot: {exc}", severity="error")
            return
        self.notify(f"saved {saved}")

    def watch_theme(self) -> None:
        """Redraw everything that colours itself from the theme.

        Textual restyles what it laid out; the charts, bars and tiles hold Rich
        text with colours baked in, so a theme change has to reach them — every
        widget that can redraw itself is asked to.
        """
        if not self.is_running:
            return
        for widget in self.query(Static):
            redraw = getattr(widget, "redraw", None)
            if callable(redraw):
                redraw()

    # ---- the worker ----------------------------------------------------

    @work(thread=True, exclusive=True)
    def train_worker(self) -> None:
        """Run the training loop off the event loop.

        Every exception is caught and reported rather than left to kill the
        worker quietly: a run that dies on the first batch — an out-of-memory,
        a dataset that will not download — is exactly when a display earns its
        keep, and a blank screen would be the worst possible answer.
        """
        error: BaseException | None = None
        try:
            train(self.cfg, self.resume, observer=self.observer)
        except Exception as exc:
            # Caught rather than left to kill the worker quietly; re-reported
            # to the UI below, which is the only place a user would see it.
            error = exc
        with contextlib.suppress(RuntimeError):  # pragma: no cover - app closed first
            self.call_from_thread(self.post_message, TrainingEnded(error))

    @on(TrainingEnded)
    def training_ended(self, message: TrainingEnded) -> None:
        """Return to idle, and say how the run ended.

        Args:
            message: carries the exception, if there was one.
        """
        self.running = False
        self.observer = None
        status = self.query_one(StatusBar)
        if message.error is not None:
            status.state = "failed"
            self._restart = False
            self.log_line(f"[b red]training failed:[/b red] {message.error}")
            self.notify(str(message.error), severity="error", timeout=10)
        else:
            status.state = "done"
            self.log_line("[b green]training finished[/b green]")
            self.notify("training finished")
        if self._restart:
            self._restart = False
            self.action_start()

    # ---- applied on the event loop, from TuiObserver -------------------

    def apply_plan(self, plan: TrainPlan) -> None:
        """Fill the run panel in from the resolved plan.

        Args:
            plan: what the run settled on.
        """
        self.plan = plan
        self.query_one("#plan", Static).update(self.plan_text(plan))
        self.query_one("#epoch-bar", ProgressBar).update(
            total=max(plan.num_epochs, 1), progress=plan.start_epoch
        )
        detail = (
            f"{plan.dataset} {plan.image_size}px · {plan.device_description} · {plan.precision}"
        )
        self.sub_title = f"{plan.dataset} · {plan.device} · {plan.precision}"
        self.query_one(StatusBar).detail = f"{detail} · {plan.parameters / 1e6:.2f}M params"

    def apply_message(self, text: str) -> None:
        """Put one of the run's own lines into the log.

        Args:
            text: the message. Escaped, since it is not markup and may hold
                square brackets that Rich would read as a tag.
        """
        self.log_line(Text(text))

    def apply_progress(self, progress: BatchProgress) -> None:
        """Move the bars and refresh the live figures.

        Args:
            progress: where the run has got to.
        """
        self.query_one("#epoch-label", Label).update(
            f"epoch {progress.epoch + 1}/{progress.num_epochs}"
        )
        self.query_one("#epoch-bar", ProgressBar).update(
            total=max(progress.num_epochs, 1),
            progress=progress.epoch + progress.epoch_fraction,
        )
        self.query_one("#batch-label", Label).update(
            f"batch {progress.batch + 1}/{progress.num_batches}"
        )
        self.query_one("#batch-bar", ProgressBar).update(
            total=max(progress.num_batches, 1), progress=progress.batch + 1
        )
        if progress.loss is not None:
            self._stats["train loss"] = f"{progress.loss:.5f}"
            self.tile("tile-loss", f"{progress.loss:.5f}")
        self._stats["throughput"] = f"{progress.images_per_second:.1f} img/s"
        self.tile("tile-rate", f"{progress.images_per_second:.1f}")
        self._stats["epoch time"] = duration(progress.seconds)
        if (eta := run_eta(progress)) is not None:
            self._stats["eta"] = duration(eta)
            self.tile("tile-eta", duration(eta))
        self.query_one(StatusBar).note = (
            f"epoch {progress.epoch + 1}/{progress.num_epochs}"
            f"  ·  {progress.epoch_fraction * 100:.0f}%"
        )
        self.refresh_stats()

    def apply_epoch(self, step: int, metrics: Mapping[str, float]) -> None:
        """Fold one epoch's metrics into the charts, the bars and the stats.

        Args:
            step: the epoch index.
            metrics: metric name to value.
        """
        if (loss := metrics.get("train/loss")) is not None:
            self.train_loss.add(loss)
        if (val := metrics.get("val/loss")) is not None:
            self.val_loss.add(val)
            self.tile("tile-val", f"{val:.5f}")
        self.refresh_chart()

        for key, label in (
            ("val/loss", "val loss"),
            ("val/best_loss", "best val"),
            ("train/lr", "lr"),
            ("train/grad_norm", "grad norm"),
        ):
            if (value := metrics.get(key)) is not None:
                self._stats[label] = f"{value:.6g}"

        self.query_one(QuartileBars).show(
            [metrics.get(f"train/loss_q{index}") for index in range(4)]
        )
        self.refresh_stats()
        self.log_line(f"[dim]epoch {step + 1}[/dim] {epoch_summary(metrics)}")

    def apply_sample(self, path: Path) -> None:
        """Show the grid an epoch just wrote.

        Args:
            path: the PNG.
        """
        self.query_one(SamplePreview).show(path)
        self.query_one("#preview-panel", Panel).border_title = f"samples · {path.name}"

    # ---- rendering -----------------------------------------------------

    def tick(self) -> None:
        """Once a second: move the clock-driven numbers on."""
        if self._started is not None:
            elapsed = duration(time.monotonic() - self._started)
            self.tile("tile-elapsed", elapsed)
            if self.running:
                self.refresh_stats()

    def tile(self, tile_id: str, value: str) -> None:
        """Set one of the headline tiles.

        Args:
            tile_id: the tile's widget id.
            value: the formatted number to show.
        """
        with contextlib.suppress(Exception):  # pragma: no cover - during teardown
            self.query_one(f"#{tile_id}", StatTile).value = value

    def refresh_chart(self) -> None:
        """Redraw the loss chart from the series held on the app."""
        self.query_one(LossChart).show(
            [
                ("train", "text-primary", self.train_loss.values),
                ("val", "text-secondary", self.val_loss.values),
            ]
        )

    def log_line(self, text: str | Text) -> None:
        """Append one line to the log pane.

        Args:
            text: markup, or a :class:`~rich.text.Text` for anything that must
                not be interpreted as markup.
        """
        self.query_one("#log", RichLog).write(text)

    def refresh_stats(self) -> None:
        """Redraw the stats panel from whatever is currently known."""
        rows = list(self._stats.items())
        if self._started is not None:
            rows.append(("elapsed", duration(time.monotonic() - self._started)))
        if not rows:
            return
        self.query_one("#stats", Static).update(two_columns(rows))

    def plan_text(self, plan: TrainPlan | None = None) -> Text:
        """Describe the run, before and after the loop has resolved it.

        Args:
            plan: the resolved plan, or None to describe the config alone —
                which is all there is to say before the first batch.

        Returns:
            A two-column block of the settings worth seeing.
        """
        if plan is None:
            spec = self.cfg.dataset_spec()
            rows = [
                ("dataset", f"{spec.name} {self.cfg.image_size}px"),
                ("device", self.cfg.device),
                ("epochs", str(self.cfg.num_epochs)),
                ("batch", str(self.cfg.batch_size)),
                ("lr", f"{self.cfg.lr:g}"),
                ("status", "not started"),
            ]
            return two_columns(rows)

        batch = str(plan.batch_size)
        if plan.grad_accum > 1:
            batch += f" x{plan.grad_accum}"
        rows = [
            ("dataset", f"{plan.dataset} {plan.image_size}px x{plan.channels}"),
            ("device", plan.device_description),
            ("params", f"{plan.parameters / 1e6:.2f}M"),
            ("precision", plan.precision),
            (
                "conditioning",
                "unconditional" if plan.num_classes is None else f"{plan.num_classes} classes",
            ),
            ("epochs", f"{plan.start_epoch + 1}-{plan.num_epochs}"),
            ("batch", batch),
            ("steps/epoch", str(plan.steps_per_epoch)),
            (
                "validation",
                f"{plan.validation_images} images" if plan.validation_images else "off",
            ),
        ]
        return two_columns(rows)


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
            # Between rows rather than after each, so a panel does not carry
            # a blank final line and stand a row taller than its content.
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
