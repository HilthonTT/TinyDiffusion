"""The join between the training thread and the event loop.

Training is a blocking loop that wants a thread of its own; a Textual app is an
asyncio event loop that must never be blocked. :class:`TuiObserver` is where
the two meet: it is called from the training thread, touches no widget, and
hands each event to the app with ``call_from_thread`` for the event loop to
apply. Stopping goes the other way, as a :class:`threading.Event` the training
thread only ever reads.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from textual.message import Message

from tinydiffusion.training.observer import BatchProgress, TrainPlan

if TYPE_CHECKING:
    from tinydiffusion.tui.app import TinyDiffusionApp

__all__ = [
    "UI_INTERVAL",
    "TrainingEnded",
    "TuiObserver",
]


UI_INTERVAL = 0.05
"""Seconds between UI updates driven from the training thread.

``call_from_thread`` blocks the caller until the event loop has run the
callback, so an unthrottled observer would have training wait on rendering. The
batch callback already fires only once per
:data:`~tinydiffusion.training.reporting.DRAIN_EVERY` batches; this bounds it again
in time, for the small model whose batches go by faster than a screen refresh.
"""


class TrainingEnded(Message):
    """Posted when the training worker leaves, however it left.

    Args:
        error: the exception that ended the run, or None if it ran to the last
            epoch or stopped because it was asked to.
    """

    def __init__(self, error: BaseException | None) -> None:
        super().__init__()
        self.error = error


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
