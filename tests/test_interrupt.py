import signal

import pytest

from tinydiffusion.training.interrupt import InterruptGuard, interrupt_guard


class FakePrompt:
    """Answers a scripted sequence of questions and records what was asked."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.questions = []

    def __call__(self, question):
        self.questions.append(question)
        if not self.answers:
            raise AssertionError(f"unexpected question: {question}")
        return self.answers.pop(0)


@pytest.fixture
def tty(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)


def test_guard_starts_clean():
    assert not InterruptGuard().requested


def test_sigint_sets_the_flag_instead_of_raising():
    with interrupt_guard() as guard:
        signal.raise_signal(signal.SIGINT)
        assert guard.requested


def test_handler_is_restored_on_exit():
    before = signal.getsignal(signal.SIGINT)
    with interrupt_guard():
        pass
    assert signal.getsignal(signal.SIGINT) is before


def test_declining_keeps_training_and_rearms_the_guard(tty):
    guard = InterruptGuard()
    guard.request()

    choice = guard.resolve(prompt=FakePrompt("n"))

    assert not choice.stop
    assert not guard.requested


def test_confirming_stops_and_saves_by_default(tty):
    guard = InterruptGuard()
    guard.request()

    prompt = FakePrompt("y", "")

    choice = guard.resolve(prompt=prompt)

    assert (choice.stop, choice.save) == (True, True)
    assert len(prompt.questions) == 2


def test_confirming_can_decline_the_checkpoint(tty):
    choice = InterruptGuard().resolve(prompt=FakePrompt("y", "n"))
    assert (choice.stop, choice.save) == (True, False)


def test_blank_answer_defaults_to_continuing(tty):
    assert not InterruptGuard().resolve(prompt=FakePrompt("")).stop


def test_unparseable_answers_are_re_asked(tty, capsys):
    prompt = FakePrompt("maybe", "y", "n")

    choice = InterruptGuard().resolve(prompt=prompt)

    assert choice.stop
    assert len(prompt.questions) == 3
    assert "y" in capsys.readouterr().out


def test_non_interactive_run_saves_without_asking(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)

    choice = InterruptGuard().resolve(prompt=FakePrompt())

    assert (choice.stop, choice.save) == (True, True)


def test_closed_stdin_mid_question_saves_and_stops(tty):
    def closed(_question):
        raise EOFError

    choice = InterruptGuard().resolve(prompt=closed)

    assert (choice.stop, choice.save) == (True, True)


def test_second_interrupt_while_asking_aborts(tty):
    def interrupting(_question):
        signal.raise_signal(signal.SIGINT)
        return "n"

    with interrupt_guard() as guard:
        guard.request()
        with pytest.raises(KeyboardInterrupt):
            guard.resolve(prompt=interrupting)
