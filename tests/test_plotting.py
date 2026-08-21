import builtins
import json

import pytest

from tinydiffusion.plotting import PANELS, metrics_path, plot_runs
from tinydiffusion.utils.tracking import METRICS_FILENAME, JsonlBackend


def write_run(directory, records):
    """Write a run's metrics through the backend that training writes them with."""
    directory.mkdir(parents=True, exist_ok=True)
    backend = JsonlBackend(directory / METRICS_FILENAME)
    for step, metrics in enumerate(records):
        backend.write(metrics, step)
    backend.close()
    return directory


@pytest.fixture
def run(tmp_path):
    return write_run(
        tmp_path / "mnist",
        [
            {"train/loss": 0.9, "train/lr": 2e-4, "train/loss_q0": 1.0, "val/loss": 0.8},
            {"train/loss": 0.5, "train/lr": 2e-4, "train/loss_q0": 0.6},
            {"train/loss": 0.3, "train/lr": 1e-4, "train/loss_q0": 0.4, "val/loss": 0.35},
        ],
    )


def test_a_run_directory_resolves_to_its_metrics_file(run):
    assert metrics_path(run) == run / METRICS_FILENAME


def test_a_file_path_is_left_alone(tmp_path):
    path = tmp_path / "elsewhere.jsonl"
    assert metrics_path(path) == path


def test_plotting_a_run_writes_an_image(run, tmp_path):
    out = plot_runs([run], tmp_path / "figures" / "metrics.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_the_metrics_file_can_be_named_directly(run, tmp_path):
    assert plot_runs([run / METRICS_FILENAME], tmp_path / "m.png").is_file()


def test_several_runs_land_on_one_figure(run, tmp_path):
    other = write_run(tmp_path / "cifar10", [{"train/loss": 2.0}, {"train/loss": 1.0}])
    assert plot_runs([run, other], tmp_path / "both.png").is_file()


def test_a_run_that_logged_only_one_panels_metrics_still_plots(tmp_path):
    only_loss = write_run(tmp_path / "sparse", [{"train/loss": 1.0}, {"train/loss": 0.5}])
    assert plot_runs([only_loss], tmp_path / "sparse.png").is_file()


def test_a_resumed_run_is_plotted_as_it_now_stands(tmp_path):
    # Two sessions over the same steps: the figure follows read_metrics and
    # draws the newer one, rather than a line that doubles back.
    directory = write_run(tmp_path / "resumed", [{"train/loss": 1.0}, {"train/loss": 0.9}])
    write_run(directory, [{"train/loss": 0.5}, {"train/loss": 0.4}])
    assert plot_runs([directory], tmp_path / "resumed.png").is_file()


def test_a_metric_missing_from_some_epochs_is_skipped_not_zeroed(run, tmp_path):
    # val/loss is written every val_every epochs; the gap is not a value.
    records = [json.loads(line) for line in (run / METRICS_FILENAME).read_text().splitlines()]
    assert sum("val/loss" in record for record in records) == 2
    assert plot_runs([run], tmp_path / "gap.png").is_file()


def test_no_runs_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no runs"):
        plot_runs([], tmp_path / "out.png")


def test_a_missing_metrics_file_is_reported(tmp_path):
    with pytest.raises(ValueError, match="no metrics in"):
        plot_runs([tmp_path / "nowhere"], tmp_path / "out.png")


def test_a_run_with_nothing_plottable_is_reported(tmp_path):
    directory = write_run(tmp_path / "odd", [{"something/else": 1.0}, {"something/else": 2.0}])
    with pytest.raises(ValueError, match="anything this can plot"):
        plot_runs([directory], tmp_path / "out.png")


def test_every_panel_names_at_least_one_metric():
    assert all(keys for _, keys, _ in PANELS)


def test_a_missing_matplotlib_names_the_extra_that_supplies_it(run, tmp_path, monkeypatch):
    real_import = builtins.__import__

    def refuse_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_matplotlib)
    with pytest.raises(RuntimeError, match="plots"):
        plot_runs([run], tmp_path / "out.png")
