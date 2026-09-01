import json

import pytest

from tinydiffusion.sweep import (
    SweepRun,
    point_name,
    run_sweep,
    sweep_points,
    sweep_summary,
)
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.utils.tracking import METRICS_FILENAME

BASE = TrainConfig(num_epochs=1, device="cpu")


def write_metrics(log_dir, *records):
    """Stand in for a finished run's metrics.jsonl."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / METRICS_FILENAME
    path.write_text(
        "".join(json.dumps({"step": i, **record}) + "\n" for i, record in enumerate(records)),
        encoding="utf-8",
    )
    return path


def test_a_point_is_named_after_what_distinguishes_it():
    assert point_name({"lr": 0.0001, "batch_size": 64}) == "lr=0.0001_batch_size=64"


def test_a_name_survives_a_filesystem():
    """A tuple or a path on an axis must not produce an unusable directory."""
    name = point_name({"channel_mult": (1, 2, 2)})
    assert not set(name) & set('<>:"/\\|?*')


def test_a_sweep_with_no_axes_is_one_run():
    assert point_name({}) == "base"


def test_every_combination_gets_a_point(tmp_path):
    points = sweep_points(BASE, [("lr", [1e-4, 2e-4]), ("batch_size", [32, 64, 128])], tmp_path)
    assert len(points) == 6
    assert [p.overrides["lr"] for p in points] == [1e-4] * 3 + [2e-4] * 3


def test_each_point_writes_somewhere_of_its_own(tmp_path):
    """The whole reason this is not a shell loop."""
    points = sweep_points(BASE, [("lr", [1e-4, 2e-4])], tmp_path)
    directories = {p.config.log_dir for p in points}
    directories |= {p.config.ckpt_dir for p in points}
    directories |= {p.config.out_dir for p in points}
    assert len(directories) == 6
    for point in points:
        assert point.config.log_dir.parent == tmp_path


def test_the_axis_value_reaches_the_config(tmp_path):
    (point,) = sweep_points(BASE, [("lr", [3e-4])], tmp_path)
    assert point.config.lr == 3e-4


def test_a_value_that_needs_coercing_is_coerced(tmp_path):
    """TOML hands back a list where the field holds a tuple."""
    (point,) = sweep_points(BASE, [("channel_mult", [[1, 2]])], tmp_path)
    assert point.config.channel_mult == (1, 2)


def test_sweeping_a_directory_field_is_refused(tmp_path):
    with pytest.raises(ValueError, match="cannot sweep over 'log_dir'"):
        sweep_points(BASE, [("log_dir", ["a", "b"])], tmp_path)


def test_an_unknown_field_is_caught_before_anything_trains(tmp_path):
    with pytest.raises(ValueError, match="unknown config field 'lr_rate'"):
        sweep_points(BASE, [("lr_rate", [1e-4])], tmp_path)


def test_an_empty_axis_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no values"):
        sweep_points(BASE, [("lr", [])], tmp_path)


def test_a_repeated_axis_is_refused(tmp_path):
    with pytest.raises(ValueError, match="given twice"):
        sweep_points(BASE, [("lr", [1e-4]), ("lr", [2e-4])], tmp_path)


def test_an_untrainable_combination_fails_at_expansion(tmp_path):
    """Better than discovering it after the points before it have run."""
    with pytest.raises(ValueError, match="guidance"):
        sweep_points(BASE, [("guidance", [2.0])], tmp_path)


def test_every_point_is_trained_in_order(tmp_path):
    points = sweep_points(BASE, [("lr", [1e-4, 2e-4])], tmp_path)
    trained = []
    runs = list(run_sweep(points, train=trained.append, say=lambda _: None))

    assert [cfg.lr for cfg in trained] == [1e-4, 2e-4]
    assert all(run.ok for run in runs)


def test_a_failing_point_does_not_stop_the_sweep(tmp_path):
    """Five good runs are not worth losing to one bad combination."""
    points = sweep_points(BASE, [("lr", [1e-4, 2e-4, 4e-4])], tmp_path)

    def train(cfg):
        if cfg.lr == 2e-4:
            raise RuntimeError("out of memory")

    runs = list(run_sweep(points, train=train, say=lambda _: None))

    assert [run.ok for run in runs] == [True, False, True]
    assert "out of memory" in str(runs[1].error)


def test_an_interrupt_ends_the_sweep_rather_than_one_point(tmp_path):
    """A failing point is a fact; an interrupt is an instruction."""
    points = sweep_points(BASE, [("lr", [1e-4, 2e-4])], tmp_path)

    def train(cfg):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        list(run_sweep(points, train=train, say=lambda _: None))


def test_a_finished_point_reports_its_best_validation_loss(tmp_path):
    points = sweep_points(BASE, [("lr", [1e-4])], tmp_path)

    def train(cfg):
        write_metrics(cfg.log_dir, {"val/loss": 0.5}, {"val/loss": 0.3}, {"val/loss": 0.4})

    (run,) = list(run_sweep(points, train=train, say=lambda _: None))
    assert run.best_val_loss == pytest.approx(0.3)
    assert run.epochs == 3


def test_a_run_without_validation_has_no_loss_to_report(tmp_path):
    points = sweep_points(BASE, [("lr", [1e-4])], tmp_path)

    def train(cfg):
        write_metrics(cfg.log_dir, {"train/loss": 0.5})

    (run,) = list(run_sweep(points, train=train, say=lambda _: None))
    assert run.best_val_loss is None
    assert run.epochs == 1


def test_skip_existing_leaves_a_finished_point_alone(tmp_path):
    """What makes an interrupted sweep resumable rather than restartable."""
    points = sweep_points(BASE, [("lr", [1e-4, 2e-4])], tmp_path)
    write_metrics(points[0].log_dir, {"val/loss": 0.25})

    trained = []
    runs = list(run_sweep(points, train=trained.append, skip_existing=True, say=lambda _: None))

    assert [cfg.lr for cfg in trained] == [2e-4]
    assert runs[0].best_val_loss == pytest.approx(0.25)


def test_skip_existing_is_off_by_default(tmp_path):
    points = sweep_points(BASE, [("lr", [1e-4])], tmp_path)
    write_metrics(points[0].log_dir, {"val/loss": 0.25})

    trained = []
    list(run_sweep(points, train=trained.append, say=lambda _: None))
    assert len(trained) == 1


def test_the_summary_ranks_by_validation_loss(tmp_path):
    points = sweep_points(BASE, [("lr", [1e-4, 2e-4])], tmp_path)
    runs = [
        SweepRun(point=points[0], best_val_loss=0.4, epochs=2),
        SweepRun(point=points[1], best_val_loss=0.2, epochs=2),
    ]
    lines = sweep_summary(runs).splitlines()
    assert lines[1].startswith(points[1].name)


def test_the_summary_puts_the_unranked_at_the_bottom(tmp_path):
    """A missing loss is unranked, not infinitely bad."""
    points = sweep_points(BASE, [("lr", [1e-4, 2e-4])], tmp_path)
    runs = [
        SweepRun(point=points[0], error=RuntimeError("boom")),
        SweepRun(point=points[1], best_val_loss=0.2, epochs=2),
    ]
    summary = sweep_summary(runs)
    lines = summary.splitlines()

    assert lines[1].startswith(points[1].name)
    assert "failed" in lines[2]
    assert "1 of 2 points failed" in summary
    assert "boom" in summary


def test_an_empty_sweep_says_so():
    assert sweep_summary([]) == "no points to run"
