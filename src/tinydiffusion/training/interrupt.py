"""Turn Ctrl+C during training into a confirmation prompt.

A training run is expensive enough that a stray Ctrl+C should not throw the
epoch away. The signal handler here only raises a flag; the training loop
notices it at the next batch boundary — a point where the model, optimiser and
EMA are all consistent — and asks the user what to do.
"""

import signal
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType

__all__ = ["InterruptChoice", "InterruptGuard", "interrupt_guard"]


@dataclass(frozen=True)
class InterruptChoice:
    """What the user decided to do about a pending interrupt.

    Attributes:
        stop: whether training should end now.
        save: whether to write a resumable checkpoint before ending.
    """

    stop: bool
    save: bool = False


_UNATTENDED = InterruptChoice(stop=True, save=True)


def _ask_yes_no(question: str, *, default: bool, prompt: Callable[[str], str]) -> bool:
    """Ask a yes/no question, repeating until the answer parses.

    Args:
        question: text shown before the hint, without trailing punctuation.
        default: answer used when the user just presses Enter.
        prompt: input function, injectable for tests.

    Returns:
        The user's answer.

    Raises:
        EOFError: if the input stream closes.
    """
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        answer = prompt(f"{question} {hint} ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("please answer 'y' or 'n'")


class InterruptGuard:
    """Flag set by SIGINT, resolved into a decision at a safe point."""

    def __init__(self) -> None:
        self._requested = False

    @property
    def requested(self) -> bool:
        """Whether a Ctrl+C is waiting to be resolved."""
        return self._requested

    def request(self) -> None:
        """Record that an interrupt arrived."""
        self._requested = True

    def clear(self) -> None:
        """Drop a pending interrupt without asking about it.

        For the ranks of a distributed run that saw the Ctrl+C but left the
        decision to the main rank: once that decision has been shared, every
        rank's flag must go down, or the next check re-raises the question.
        """
        self._requested = False

    def resolve(self, *, prompt: Callable[[str], str] = input) -> InterruptChoice:
        """Ask the user whether to cancel, and whether to checkpoint first.

        The flag is cleared first, so declining to stop leaves the guard ready
        for the next Ctrl+C. While the questions are on screen the default
        signal handler is restored: a second Ctrl+C then aborts outright, which
        is the escape hatch for anyone who wants out immediately.

        Args:
            prompt: input function, injectable for tests.

        Returns:
            The user's decision.

        Raises:
            KeyboardInterrupt: if the user interrupts again while answering.
        """
        self._requested = False
        if not sys.stdin or not sys.stdin.isatty():
            print("interrupted; saving a checkpoint before exiting")
            return _UNATTENDED

        with _default_sigint():
            try:
                if not _ask_yes_no("\nstop training?", default=False, prompt=prompt):
                    print("continuing")
                    return InterruptChoice(stop=False)
                save = _ask_yes_no(
                    "save a checkpoint so training can resume later?",
                    default=True,
                    prompt=prompt,
                )
            except EOFError:
                print()
                return _UNATTENDED
        return InterruptChoice(stop=True, save=save)


@contextmanager
def _default_sigint() -> Generator[None]:
    """Restore Python's default SIGINT behaviour for the duration of a block."""
    try:
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    except ValueError:  # pragma: no cover
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


@contextmanager
def interrupt_guard() -> Generator[InterruptGuard]:
    """Install a deferring SIGINT handler for the duration of a block.

    Outside the main thread signals cannot be installed at all; the guard is
    still yielded, it simply never fires and Ctrl+C keeps its usual meaning.

    Yields:
        The guard the training loop should poll.
    """
    guard = InterruptGuard()

    def handler(_signum: int, _frame: FrameType | None) -> None:
        guard.request()

    try:
        previous = signal.signal(signal.SIGINT, handler)
    except ValueError:  # pragma: no cover
        yield guard
        return
    try:
        yield guard
    finally:
        signal.signal(signal.SIGINT, previous)
