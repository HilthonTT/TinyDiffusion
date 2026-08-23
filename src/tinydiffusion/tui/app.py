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
"""

import contextlib
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, ProgressBar, RichLog, Sparkline, Static

from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.observer import BatchProgress, TrainPlan
from tinydiffusion.training.train import train
from tinydiffusion.tui.preview import HALF_BLOCK, half_block_rows

if TYPE_CHECKING:
    from textual.binding import BindingType

__all__ = ["QuartileBars", "SamplePreview", "TinyDiffusionApp", "TrainingEnded", "TuiObserver"]

UI_INTERVAL = 0.05
"""Seconds between UI updates driven from the training thread.

``call_from_thread`` blocks the caller until the event loop has run the
callback, so an unthrottled observer would have training wait on rendering. The
batch callback already fires only once per
:data:`~tinydiffusion.training.train.DRAIN_EVERY` batches; this bounds it again
in time, for the small model whose batches go by faster than a screen refresh.
"""

MAX_POINTS = 240
"""Epochs kept per chart series. A sparkline cannot show more than its width."""


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

    def for_sparkline(self) -> list[float]:
        """The points, in the shape a sparkline can actually draw.

        Returns:
            The values, with a single point doubled: one point has no range to
            scale against and renders blank, where two draw the flat line that
            is the honest picture of a run that has produced one number.
        """
        if not self.values:
            return [0.0]
        return self.values if len(self.values) > 1 else self.values * 2


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


class SamplePreview(Static):
    """The latest sample grid, drawn two pixels to a character cell."""

    DEFAULT_CSS = """
    SamplePreview {
        height: 1fr;
        min-height: 4;
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
    glance.
    """

    BLOCKS = "▏▎▍▌▋▊▉█"
    """Eighth-width blocks, so a bar has eight times a cell's resolution."""

    def __init__(self) -> None:
        super().__init__("waiting for the first epoch", id="quartiles")

    def show(self, values: list[float | None]) -> None:
        """Draw one bar per quartile.

        Args:
            values: the four means, lowest timestep bucket first. None where an
                epoch logged nothing for that bucket.
        """
        present = [value for value in values if value is not None]
        if not present:
            return
        # One scale across all four, so the bars compare with each other rather
        # than each being normalised into saying nothing.
        top = max(present)
        width = max(self.size.width - 14, 4)

        text = Text()
        for index, value in enumerate(values):
            if index:
                text.append("\n")
            text.append(f"q{index} ", style="bold")
            if value is None:
                text.append(f"{'-':<{width}}", style="dim")
                continue
            filled = (value / top) * width if top > 0 else 0.0
            whole = int(filled)
            remainder = filled - whole
            bar = "█" * whole
            if whole < width and remainder > 0:
                bar += self.BLOCKS[min(int(remainder * 8), 7)]
            text.append(f"{bar:<{width}}", style="cyan")
            text.append(f" {value:.4f}")
        self.update(text)


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
    #columns { height: 1fr; }
    /* Scrollable rather than fixed: the panels are sized by their content, and
       a narrow terminal or a long device name pushes the last of them past the
       bottom. Clipping it silently is the one outcome worth ruling out. */
    #sidebar { width: 40; height: 1fr; }
    #sidebar > Vertical, #main > Vertical {
        border: round $foreground 30%;
        padding: 0 1;
        height: auto;
    }
    #main { width: 1fr; height: 1fr; }
    #preview-panel { height: 1fr; min-height: 6; }
    #log-panel { height: 10; border: round $foreground 30%; padding: 0 1; }
    .heading { text-style: bold; color: $accent; }
    Sparkline { height: 2; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("s", "start", "Start"),
        Binding("x", "stop", "Stop"),
        Binding("l", "toggle_log", "Log"),
        Binding("d", "cycle_theme", "Theme"),
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
        self._stats: dict[str, str] = {}
        self._started: float | None = None

    # ---- layout --------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Build the widget tree.

        Yields:
            The app's widgets.
        """
        yield Header(show_clock=True)
        with Horizontal(id="columns"):
            with VerticalScroll(id="sidebar"):
                with Vertical():
                    yield Label("run", classes="heading")
                    yield Static(self.plan_text(), id="plan")
                with Vertical():
                    yield Label("progress", classes="heading")
                    yield Label("epoch -", id="epoch-label")
                    yield ProgressBar(total=100, show_eta=False, id="epoch-bar")
                    yield Label("batch -", id="batch-label")
                    yield ProgressBar(total=100, show_eta=False, id="batch-bar")
                with Vertical():
                    yield Label("stats", classes="heading")
                    yield Static("not started", id="stats")
            with Vertical(id="main"):
                with Vertical():
                    yield Label("train/loss per epoch", classes="heading")
                    yield Sparkline([0.0], id="train-spark")
                    yield Label("val/loss per epoch", classes="heading")
                    yield Sparkline([0.0], id="val-spark")
                with Vertical():
                    yield Label("train loss by timestep quartile", classes="heading")
                    yield QuartileBars()
                with Vertical(id="preview-panel"):
                    yield Label("latest samples (generated above real)", classes="heading")
                    yield SamplePreview()
        with Vertical(id="log-panel"):
            yield RichLog(markup=True, wrap=True, id="log")
        yield Footer()

    def on_mount(self) -> None:
        """Say what is loaded, and start straight away if that was asked for."""
        self.sub_title = f"{self.cfg.dataset} · {self.cfg.device}"
        self.log_line(f"[dim]logs:[/dim] {self.cfg.log_dir}")
        if self.resume is not None:
            self.log_line(f"[dim]resuming from:[/dim] {self.resume}")
        self.log_line(
            "[b]s[/b] start · [b]x[/b] stop · [b]l[/b] log · [b]d[/b] theme · [b]q[/b] quit"
        )
        if self.autostart:
            self.action_start()

    # ---- actions -------------------------------------------------------

    def action_start(self) -> None:
        """Begin a training run, unless one is already going."""
        if self.running:
            self.notify("training is already running", severity="warning")
            return
        self.running = True
        self._started = time.monotonic()
        self.observer = TuiObserver(self)
        self.log_line("[b green]starting[/b green]")
        self.train_worker()

    def action_stop(self) -> None:
        """Ask the run to stop at the next batch boundary, checkpointing as it goes."""
        if not self.running or self.observer is None:
            self.notify("nothing is running", severity="warning")
            return
        self.log_line("[yellow]stopping at the next batch boundary…[/yellow]")
        self.observer.request_stop()

    def action_toggle_log(self) -> None:
        """Show or hide the log pane, for when the pictures matter more."""
        panel = self.query_one("#log-panel")
        panel.display = not panel.display

    def action_cycle_theme(self) -> None:
        """Switch between the light and dark themes."""
        self.theme = "textual-light" if self.theme == "textual-dark" else "textual-dark"

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
        if message.error is not None:
            self.log_line(f"[b red]training failed:[/b red] {message.error}")
            self.notify(str(message.error), severity="error", timeout=10)
        else:
            self.log_line("[b green]training finished[/b green]")
            self.notify("training finished")

    # ---- applied on the event loop, from TuiObserver -------------------

    def apply_plan(self, plan: TrainPlan) -> None:
        """Fill the run panel in from the resolved plan.

        Args:
            plan: what the run settled on.
        """
        self.query_one("#plan", Static).update(self.plan_text(plan))
        self.query_one("#epoch-bar", ProgressBar).update(
            total=max(plan.num_epochs, 1), progress=plan.start_epoch
        )
        self.sub_title = f"{plan.dataset} · {plan.device} · {plan.precision}"

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
        self._stats["throughput"] = f"{progress.images_per_second:.1f} img/s"
        self._stats["epoch time"] = duration(progress.seconds)
        if (eta := run_eta(progress)) is not None:
            self._stats["eta"] = duration(eta)
        self.refresh_stats()

    def apply_epoch(self, step: int, metrics: Mapping[str, float]) -> None:
        """Fold one epoch's metrics into the charts, the bars and the stats.

        Args:
            step: the epoch index.
            metrics: metric name to value.
        """
        if (loss := metrics.get("train/loss")) is not None:
            self.train_loss.add(loss)
            self.query_one("#train-spark", Sparkline).data = self.train_loss.for_sparkline()
        if (val := metrics.get("val/loss")) is not None:
            self.val_loss.add(val)
            self.query_one("#val-spark", Sparkline).data = self.val_loss.for_sparkline()

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

    # ---- rendering -----------------------------------------------------

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
