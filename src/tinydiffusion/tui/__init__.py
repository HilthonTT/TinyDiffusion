"""The terminal dashboard for a training run.

Built on Textual, which is an optional dependency — install the ``tui`` extra.
Nothing else in the package imports this, so everything else works without it.

The entry point is :func:`run_tui`; the app itself lives in
:mod:`tinydiffusion.tui.app`, and is imported only once the extra has been
confirmed present, so a missing Textual is one line rather than a traceback.
"""

from pathlib import Path

from tinydiffusion.training.config import TrainConfig

__all__ = ["run_tui"]


def run_tui(
    cfg: TrainConfig | None = None,
    resume: Path | None = None,
    *,
    autostart: bool = False,
) -> None:
    """Open the dashboard.

    Blocks until the user quits. Training is started from inside, either by the
    key or by `autostart`, and runs on a worker thread.

    Args:
        cfg: the configuration a run will use. Defaults are used when omitted.
        resume: a checkpoint to continue from, or None to start fresh.
        autostart: begin training as soon as the app is ready.

    Raises:
        ImportError: if the ``tui`` extra is not installed. The same shape the
            other extras raise, so the CLI turns it into one line rather than a
            traceback.
    """
    try:
        from tinydiffusion.tui.app import TinyDiffusionApp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the dashboard needs the 'tui' extra: pip install 'tinydiffusion[tui]'"
        ) from exc

    TinyDiffusionApp(cfg or TrainConfig(), resume, autostart=autostart).run()
