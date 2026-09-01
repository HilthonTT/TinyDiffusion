import json

import pytest
import torch

from tinydiffusion.utils.tracking import (
    METRICS_FILENAME,
    ConsoleBackend,
    JsonlBackend,
    LoggerBackend,
    RunLogger,
    WandbBackend,
    null_logger,
    quartile_means,
    read_metrics,
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


class ExplodingBackend:
    """A backend that cannot be closed."""

    def write(self, metrics, step): ...

    def close(self):
        raise OSError("nope")


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
    healthy = RecordingBackend()
    with pytest.raises(ExceptionGroup):
        RunLogger([ExplodingBackend(), healthy]).close()
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
    path = tmp_path / "metrics.jsonl"
    backend = JsonlBackend(path)
    backend.write({"train/loss": value, "train/lr": 1e-4}, 2)
    backend.close()

    record = json.loads(path.read_text(), parse_constant=_reject)
    assert record["train/loss"] is None
    assert record["train/lr"] == 1e-4
    assert record["step"] == 2


def test_a_metric_cannot_overwrite_the_step_it_was_logged_at(tmp_path):
    path = tmp_path / "metrics.jsonl"
    backend = JsonlBackend(path)
    backend.write({"step": 99.0, "time": 0.0}, 3)
    backend.close()

    record = json.loads(path.read_text())
    assert record["step"] == 3
    assert record["time"] > 0.0


def _reject(token):
    raise AssertionError(f"non-JSON token written: {token}")


def test_each_reopen_stamps_a_new_session(tmp_path):
    path = tmp_path / "metrics.jsonl"
    for _ in range(3):
        backend = JsonlBackend(path)
        backend.write({"loss": 1.0}, 0)
        backend.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["session"] for r in records] == [0, 1, 2]


def test_a_resumed_run_reads_back_as_one_record_per_step(tmp_path):
    path = tmp_path / "metrics.jsonl"
    first = JsonlBackend(path)
    for step, loss in enumerate((3.0, 2.0, 1.0)):
        first.write({"loss": loss}, step)
    first.close()

    resumed = JsonlBackend(path)
    for step, loss in ((1, 0.5), (2, 0.25)):
        resumed.write({"loss": loss}, step)
    resumed.close()

    assert len(path.read_text().splitlines()) == 5
    assert [(r["step"], r["loss"]) for r in read_metrics(path)] == [(0, 3.0), (1, 0.5), (2, 0.25)]


def test_reading_a_file_written_before_sessions_existed(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 0, "loss": 1.0}\n{"step": 0, "loss": 2.0}\n')
    assert read_metrics(path) == [{"step": 0, "loss": 2.0}]


def test_a_truncated_final_line_does_not_lose_the_epochs_before_it(tmp_path):
    path = tmp_path / "metrics.jsonl"
    backend = JsonlBackend(path)
    backend.write({"loss": 1.0}, 0)
    backend.close()
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"step": 1, "los')

    assert [r["step"] for r in read_metrics(path)] == [0]


def test_reading_a_missing_file_is_empty(tmp_path):
    assert read_metrics(tmp_path / "nothing.jsonl") == []


def test_a_failing_close_does_not_shadow_the_error_that_ended_the_run():
    with (
        pytest.warns(UserWarning, match="nope"),
        pytest.raises(RuntimeError, match="diverged"),
        RunLogger([ExplodingBackend()]),
    ):
        raise RuntimeError("diverged")


def test_a_failing_close_still_raises_when_the_block_succeeded():
    with pytest.raises(ExceptionGroup), RunLogger([ExplodingBackend()]):
        pass


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
    losses = torch.tensor([1.0, 2.0])
    sums, counts = timestep_quartile_totals(losses, torch.tensor([0, 90]), num_timesteps=100)

    assert isinstance(sums, torch.Tensor) and isinstance(counts, torch.Tensor)
    assert sums.device == losses.device and counts.device == losses.device
    assert sums.dtype is losses.dtype


def test_totals_accumulate_across_batches_into_a_pooled_mean():
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


class _FakeWandbRun:
    """The parts of a wandb run this backend touches."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.logged = []
        self.finished = False
        self.fail = False

    def log(self, metrics, step):
        if self.fail:
            raise RuntimeError("network down")
        self.logged.append((dict(metrics), step))

    def finish(self):
        self.finished = True


@pytest.fixture
def fake_wandb(monkeypatch):
    """Install a stand-in ``wandb`` module and hand back the run it creates.

    The backend imports wandb lazily and inside its constructor, which is what
    makes this substitutable at all: there is no import at module scope to have
    already resolved by the time a test runs.
    """
    import sys
    import types

    module = types.ModuleType("wandb")
    created = []

    def init(**kwargs):
        run = _FakeWandbRun(**kwargs)
        created.append(run)
        return run

    module.init = init
    monkeypatch.setitem(sys.modules, "wandb", module)
    return created


def test_wandb_sends_each_step_and_finishes_on_close(tmp_path, fake_wandb):
    backend = WandbBackend(tmp_path / "run", project="proj", name="mnist", config={"lr": 1e-4})
    backend.write({"train/loss": 0.5}, step=3)
    backend.close()

    (run,) = fake_wandb
    assert run.init_kwargs["project"] == "proj"
    assert run.init_kwargs["name"] == "mnist"
    assert run.init_kwargs["config"] == {"lr": 1e-4}
    assert run.logged == [({"train/loss": 0.5}, 3)]
    assert run.finished


def test_a_dropped_connection_warns_rather_than_ending_the_run(tmp_path, fake_wandb):
    """metrics.jsonl already holds the numbers, so a remote sink must not raise.

    Losing an epoch of training to a network blip would make this backend cost
    more than it is worth.
    """
    backend = WandbBackend(tmp_path / "run", project="proj")
    (run,) = fake_wandb
    run.fail = True

    with pytest.warns(UserWarning, match="wandb logging failed"):
        backend.write({"train/loss": 0.5}, step=0)


def test_for_run_wires_wandb_in_when_asked(tmp_path, fake_wandb):
    logger = RunLogger.for_run(
        tmp_path,
        console=False,
        jsonl=False,
        wandb=True,
        wandb_project="proj",
        wandb_config={"seed": 0},
    )
    logger.set(train_loss=1.0)
    logger.flush(step=0)
    logger.close()

    (run,) = fake_wandb
    assert run.logged == [({"train_loss": 1.0}, 0)]
    assert run.init_kwargs["config"] == {"seed": 0}


def test_for_run_leaves_wandb_alone_by_default(tmp_path, fake_wandb):
    RunLogger.for_run(tmp_path, console=False, jsonl=False).close()
    assert fake_wandb == []


def test_a_missing_wandb_names_the_extra_that_provides_it(tmp_path, monkeypatch):
    """The failure a base install hits, reported as a missing optional dependency."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no module named wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(RuntimeError, match="'tracking' extra"):
        RunLogger.for_run(tmp_path, console=False, jsonl=False, wandb=True)
