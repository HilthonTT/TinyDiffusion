import json

import pytest
import torch

from tinydiffusion.utils.tracking import (
    METRICS_FILENAME,
    ConsoleBackend,
    JsonlBackend,
    LoggerBackend,
    RunLogger,
    null_logger,
    quartile_means,
    timestep_quartile_losses,
    timestep_quartile_totals,
)


class RecordingBackend:
    def __init__(self):
        self.writes = []
        self.closed = False

    def write(self, metrics, step):
        self.writes.append((dict(metrics), step))

    def close(self):
        self.closed = True


def test_a_plain_object_satisfies_the_backend_protocol():
    assert isinstance(RecordingBackend(), LoggerBackend)


def test_accumulated_values_are_flushed_as_means():
    backend = RecordingBackend()
    logger = RunLogger([backend])

    logger.accumulate(loss=1.0)
    logger.accumulate(loss=3.0)
    metrics = logger.flush(step=0)

    assert metrics == {"loss": 2.0}
    assert backend.writes == [({"loss": 2.0}, 0)]


def test_set_values_are_reported_as_is_and_win_over_means():
    logger = RunLogger([])
    logger.accumulate(lr=1.0)
    logger.set(lr=3e-4)
    assert logger.flush(step=0) == {"lr": 3e-4}


def test_flushing_resets_the_buffers():
    backend = RecordingBackend()
    logger = RunLogger([backend])

    logger.accumulate(loss=2.0)
    logger.set(lr=1.0)
    logger.flush(step=0)
    logger.accumulate(loss=8.0)

    assert logger.flush(step=1) == {"loss": 8.0}


def test_an_empty_flush_writes_nothing():
    backend = RecordingBackend()
    RunLogger([backend]).flush(step=0)
    assert backend.writes == []


def test_the_context_manager_closes_every_backend():
    backends = [RecordingBackend(), RecordingBackend()]
    with RunLogger(list(backends)) as logger:
        logger.accumulate(loss=1.0)
    assert all(b.closed for b in backends)


def test_close_reports_every_failing_backend():
    class Exploding:
        def write(self, metrics, step): ...

        def close(self):
            raise OSError("nope")

        # A healthy backend after the failing one must still be closed.

    healthy = RecordingBackend()
    with pytest.raises(ExceptionGroup):
        RunLogger([Exploding(), healthy]).close()
    assert healthy.closed


def test_for_run_writes_jsonl_into_the_log_dir(tmp_path):
    log_dir = tmp_path / "runs" / "mnist"
    with RunLogger.for_run(log_dir, console=False) as logger:
        logger.accumulate(**{"train/loss": 0.5})
        logger.flush(step=0)
        logger.accumulate(**{"train/loss": 0.25})
        logger.flush(step=1)

    records = [
        json.loads(line) for line in (log_dir / METRICS_FILENAME).read_text().splitlines() if line
    ]
    assert [r["step"] for r in records] == [0, 1]
    assert [r["train/loss"] for r in records] == [0.5, 0.25]
    assert all("time" in r for r in records)


def test_jsonl_appends_to_an_existing_file(tmp_path):
    path = tmp_path / "metrics.jsonl"
    for value in (1.0, 2.0):
        backend = JsonlBackend(path)
        backend.write({"loss": value}, 0)
        backend.close()
    assert len(path.read_text().splitlines()) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_value_is_recorded_as_null(tmp_path, value):
    # json would write the bare tokens NaN/Infinity, which are a Python
    # extension: a strict reader rejects the whole file over one of them, and
    # a diverged run is exactly the one whose log is worth reading.
    path = tmp_path / "metrics.jsonl"
    backend = JsonlBackend(path)
    backend.write({"train/loss": value, "train/lr": 1e-4}, 2)
    backend.close()

    record = json.loads(path.read_text(), parse_constant=_reject)
    assert record["train/loss"] is None
    assert record["train/lr"] == 1e-4
    assert record["step"] == 2


def test_a_metric_cannot_overwrite_the_step_it_was_logged_at(tmp_path):
    # step is what every reader joins on, so it outranks a metric of the
    # same name rather than the other way round.
    path = tmp_path / "metrics.jsonl"
    backend = JsonlBackend(path)
    backend.write({"step": 99.0, "time": 0.0}, 3)
    backend.close()

    record = json.loads(path.read_text())
    assert record["step"] == 3
    assert record["time"] > 0.0


def _reject(token):
    raise AssertionError(f"non-JSON token written: {token}")


def test_closing_twice_is_harmless(tmp_path):
    backend = JsonlBackend(tmp_path / "metrics.jsonl")
    backend.close()
    backend.close()


def test_the_null_logger_discards_everything():
    with null_logger() as logger:
        logger.accumulate(loss=1.0)
        assert logger.flush(step=0) == {"loss": 1.0}


def test_console_output_holds_every_metric(capsys):
    ConsoleBackend().write({"train/loss": 0.25, "train/lr": 2e-4}, step=3)
    out = capsys.readouterr().out
    assert "step 3" in out
    assert "train/loss" in out
    assert "0.2500" in out
    # A learning rate would round to 0.0002 in fixed point.
    assert "2.0e-04" in out


def test_console_ignores_an_empty_write(capsys):
    ConsoleBackend().write({}, step=0)
    assert capsys.readouterr().out == ""


def test_quartiles_split_on_the_schedule_length():
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    steps = torch.tensor([0, 30, 55, 60, 90])
    out = timestep_quartile_losses(losses, steps, num_timesteps=100)
    assert out == {"loss_q0": 1.0, "loss_q1": 2.0, "loss_q2": 3.5, "loss_q3": 5.0}


def test_empty_quartiles_are_omitted():
    out = timestep_quartile_losses(
        torch.tensor([1.0, 2.0]), torch.tensor([0, 1]), num_timesteps=100
    )
    assert set(out) == {"loss_q0"}


def test_the_last_timestep_stays_in_the_final_quartile():
    out = timestep_quartile_losses(torch.tensor([1.0]), torch.tensor([99]), num_timesteps=100)
    assert set(out) == {"loss_q3"}


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="does not match"):
        timestep_quartile_losses(torch.zeros(3), torch.zeros(2, dtype=torch.long), 100)


def test_totals_are_sums_and_counts_rather_than_means():
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    steps = torch.tensor([0, 30, 55, 60, 90])

    sums, counts = timestep_quartile_totals(losses, steps, num_timesteps=100)

    assert torch.equal(sums, torch.tensor([1.0, 2.0, 7.0, 5.0]))
    assert torch.equal(counts, torch.tensor([1.0, 1.0, 2.0, 1.0]))


def test_totals_leave_everything_on_the_device_they_were_given():
    # The whole reason they are totals: a training loop adds them into a
    # running pair and reads the result back once, at the end of the epoch.
    losses = torch.tensor([1.0, 2.0])
    sums, counts = timestep_quartile_totals(losses, torch.tensor([0, 90]), num_timesteps=100)

    assert isinstance(sums, torch.Tensor) and isinstance(counts, torch.Tensor)
    assert sums.device == losses.device and counts.device == losses.device
    assert sums.dtype is losses.dtype


def test_totals_accumulate_across_batches_into_a_pooled_mean():
    # One batch lands a single sample in q0, the next lands three. Pooling has
    # to weight them 1:3; averaging the two batch means would call it 1:1 and
    # report 6.0 instead.
    sums, counts = torch.zeros(4), torch.zeros(4)
    for losses, steps in (
        (torch.tensor([10.0]), torch.tensor([0])),
        (torch.tensor([2.0, 2.0, 2.0]), torch.tensor([0, 1, 2])),
    ):
        batch_sums, batch_counts = timestep_quartile_totals(losses, steps, num_timesteps=100)
        sums += batch_sums
        counts += batch_counts

    assert quartile_means(sums, counts) == {"loss_q0": 4.0}


def test_means_omit_buckets_that_saw_nothing():
    sums = torch.tensor([3.0, 0.0, 8.0, 0.0])
    counts = torch.tensor([1.0, 0.0, 2.0, 0.0])

    assert quartile_means(sums, counts) == {"loss_q0": 3.0, "loss_q2": 4.0}


def test_the_convenience_form_is_the_two_halves_composed():
    losses, steps = torch.rand(64), torch.randint(0, 1000, (64,))

    assert timestep_quartile_losses(losses, steps, 1000) == quartile_means(
        *timestep_quartile_totals(losses, steps, 1000)
    )
