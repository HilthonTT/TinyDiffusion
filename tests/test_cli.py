import argparse
import builtins
import dataclasses
import sys
from pathlib import Path

import pytest
import torch

from tinydiffusion import __version__, cli
from tinydiffusion import version as version_module
from tinydiffusion.cli import build_parser, main
from tinydiffusion.cli import commands as cli_commands
from tinydiffusion.metrics.evaluate import DEFAULT_FID_IMAGES
from tinydiffusion.metrics.kid import DEFAULT_KID_SUBSET_SIZE, DEFAULT_KID_SUBSETS
from tinydiffusion.metrics.precision_recall import DEFAULT_NEIGHBOURS
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.utils.precision import DEFAULT_PRECISION, PRECISIONS


def test_version_is_exposed():
    assert __version__
    assert __version__ == version_module.__version__


def test_the_version_flag_prints_the_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"tinydiffusion {__version__}"


def test_the_short_version_flag_matches():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-V"])


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_train_parses_config_and_seed():
    args = build_parser().parse_args(["train", "--config", "cfg.toml", "--seed", "7"])
    assert args.command == "train"
    assert args.config == Path("cfg.toml")
    assert args.seed == 7


def test_train_overrides_default_to_none():
    args = build_parser().parse_args(["train"])
    assert args.config is None
    assert (args.seed, args.device, args.num_epochs, args.resume) == (None, None, None, None)
    assert (args.log_dir, args.tensorboard, args.log_console) == (None, None, None)


def test_train_parses_the_tracking_flags():
    args = build_parser().parse_args(["train", "--log-dir", "runs/x", "--tensorboard", "--quiet"])
    assert args.log_dir == Path("runs/x")
    assert args.tensorboard is True
    assert args.log_console is False


def test_sample_defaults():
    args = build_parser().parse_args(["sample", "--checkpoint", "model.pt"])
    assert args.num_images == 8
    assert args.eta == 0.0
    assert args.steps is None
    assert args.labels is None
    assert args.guidance is None
    assert args.guidance_rescale is None
    assert args.sampler is None
    assert args.batch_size is None
    assert args.save_individual is False


def test_sample_parses_the_batch_size():
    args = build_parser().parse_args(["sample", "--checkpoint", "m.pt", "--batch-size", "64"])
    assert args.batch_size == 64


def test_sample_parses_save_individual():
    args = build_parser().parse_args(["sample", "--checkpoint", "m.pt", "--save-individual"])
    assert args.save_individual is True


@pytest.mark.parametrize(
    ("given", "expected"), [("3", [3]), ("0,1,2", [0, 1, 2]), ("7, 7", [7, 7])]
)
def test_sample_parses_labels(given, expected):
    args = build_parser().parse_args(["sample", "--checkpoint", "m.pt", "--labels", given])
    assert args.labels == expected


@pytest.mark.parametrize("given", ["", "one", "1,,x"])
def test_sample_rejects_labels_that_are_not_whole_numbers(given):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["sample", "--checkpoint", "m.pt", "--labels", given])


@pytest.mark.parametrize("command", ["sample", "fid"])
def test_the_sampler_can_be_chosen_per_command(command):
    args = build_parser().parse_args([command, "--checkpoint", "m.pt", "--sampler", "dpmpp"])
    assert args.sampler == "dpmpp"


@pytest.mark.parametrize("command", ["sample", "fid"])
def test_an_unknown_sampler_is_refused_by_the_parser(command):
    with pytest.raises(SystemExit):
        build_parser().parse_args([command, "--checkpoint", "m.pt", "--sampler", "euler"])


def test_sample_parses_the_guidance_scale():
    args = build_parser().parse_args(["sample", "--checkpoint", "m.pt", "--guidance", "2.5"])
    assert args.guidance == 2.5


def test_sample_parses_the_guidance_rescale():
    args = build_parser().parse_args(
        ["sample", "--checkpoint", "m.pt", "--guidance", "5", "--guidance-rescale", "0.7"]
    )
    assert (args.guidance, args.guidance_rescale) == (5.0, 0.7)


def test_fid_defaults():
    args = build_parser().parse_args(["fid", "--checkpoint", "model.pt"])
    assert args.command == "fid"
    assert args.num_images == DEFAULT_FID_IMAGES
    assert args.split == "train"
    assert args.eta == 0.0
    assert args.use_ema is True
    assert (args.steps, args.guidance, args.batch_size, args.data_root) == (None,) * 4
    assert args.guidance_rescale is None


def test_fid_parses_its_overrides():
    args = build_parser().parse_args(
        [
            "fid",
            "--checkpoint",
            "m.pt",
            "--num-images",
            "512",
            "--split",
            "test",
            "--steps",
            "20",
            "--guidance",
            "1.5",
            "--guidance-rescale",
            "0.7",
            "--no-ema",
        ]
    )
    assert (args.num_images, args.split, args.steps, args.guidance) == (512, "test", 20, 1.5)
    assert args.guidance_rescale == 0.7
    assert args.use_ema is False


def test_fid_requires_a_checkpoint():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["fid"])


def test_fid_runs_and_prints_a_report(capsys, monkeypatch, tmp_path):
    class FakeResult:
        def format(self):
            return "fid 1.234"

    seen = {}

    def fake_fid(checkpoint, **kwargs):
        seen["checkpoint"] = checkpoint
        seen.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(cli_commands, "fid_for_checkpoint", fake_fid)
    assert main(["fid", "--checkpoint", str(tmp_path / "m.pt"), "--num-images", "64"]) == 0
    assert "fid 1.234" in capsys.readouterr().out
    assert seen["num_images"] == 64
    assert seen["checkpoint"] == tmp_path / "m.pt"


def test_main_reports_a_missing_fid_checkpoint(capsys, tmp_path):
    assert main(["fid", "--checkpoint", str(tmp_path / "nope.pt"), "--num-images", "4"]) == 1
    assert "error:" in capsys.readouterr().err


def test_serve_defaults():
    args = build_parser().parse_args(["serve", "--checkpoint", "model.pt"])
    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.use_ema is True
    assert args.cors_origins is None
    assert (args.device, args.image_dir) == (None, None)


def test_serve_parses_its_overrides():
    args = build_parser().parse_args(
        [
            "serve",
            "--checkpoint",
            "m.pt",
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--max-images",
            "4",
            "--cors-origin",
            "http://a.test",
            "--cors-origin",
            "http://b.test",
            "--no-ema",
        ]
    )
    assert (args.host, args.port, args.max_images) == ("0.0.0.0", 9001, 4)
    assert args.cors_origins == ["http://a.test", "http://b.test"]
    assert args.use_ema is False


def test_serve_requires_a_checkpoint():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["serve"])


def test_serve_builds_a_config_and_runs(capsys, monkeypatch, tmp_path):
    pytest.importorskip("fastapi", reason="needs the 'server' extra")
    checkpoint = tmp_path / "m.pt"
    checkpoint.write_bytes(b"")
    seen = {}
    monkeypatch.setattr("tinydiffusion.server.app.serve", lambda config: seen.update(config=config))

    assert main(["serve", "--checkpoint", str(checkpoint), "--port", "9999"]) == 0
    assert seen["config"].checkpoint == checkpoint
    assert seen["config"].port == 9999
    assert "http://127.0.0.1:9999" in capsys.readouterr().out


def test_serve_reports_a_missing_checkpoint_before_binding(capsys, tmp_path):
    assert main(["serve", "--checkpoint", str(tmp_path / "nope.pt")]) == 1
    assert "no such checkpoint" in capsys.readouterr().err


@pytest.fixture
def trained(monkeypatch):
    """Capture the config `main` hands to the training loop."""
    seen = {}

    def fake_train(cfg, resume=None):
        seen["cfg"] = cfg
        seen["resume"] = resume

    monkeypatch.setattr(cli_commands, "train_run", fake_train)
    return seen


def _checkpoint(tmp_path, **overrides) -> Path:
    """A checkpoint carrying nothing but the config it was trained with."""
    cfg = dataclasses.replace(TrainConfig(), **overrides)
    path = tmp_path / "last.pt"
    stored = {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()}
    torch.save({"config": stored}, path)
    return path


def test_a_bare_resume_continues_the_checkpoints_own_config(tmp_path, trained):
    path = _checkpoint(tmp_path, base_channels=32, num_epochs=7, dataset="cifar10")

    assert main(["train", "--resume", str(path)]) == 0

    assert trained["cfg"].base_channels == 32
    assert trained["cfg"].num_epochs == 7
    assert trained["cfg"].dataset == "cifar10"
    assert trained["resume"] == path


def test_an_explicit_config_still_wins_over_the_checkpoints(tmp_path, trained):
    path = _checkpoint(tmp_path, base_channels=32)
    config = tmp_path / "cfg.toml"
    config.write_text("[model]\nbase_channels = 16\n", encoding="utf-8")

    assert main(["train", "--config", str(config), "--resume", str(path)]) == 0

    assert trained["cfg"].base_channels == 16


def test_flags_still_override_a_resumed_config(tmp_path, trained):
    path = _checkpoint(tmp_path, num_epochs=7, seed=3)

    assert main(["train", "--resume", str(path), "--epochs", "9"]) == 0

    assert trained["cfg"].num_epochs == 9
    assert trained["cfg"].seed == 3


def test_a_resume_without_provenance_asks_for_a_config(capsys, tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"epoch": 0}, path)

    assert main(["train", "--resume", str(path)]) == 1
    assert "--config" in capsys.readouterr().err


def test_main_reports_a_missing_config(capsys, tmp_path):
    assert main(["train", "--config", str(tmp_path / "nope.toml")]) == 1
    assert "error:" in capsys.readouterr().err


def test_main_reports_a_missing_checkpoint(capsys, tmp_path):
    assert main(["sample", "--checkpoint", str(tmp_path / "nope.pt")]) == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("batch_size=64", ("batch_size", 64)),
        ("lr=1e-4", ("lr", 1e-4)),
        ("amp=false", ("amp", False)),
        ("channel_mult=[1, 2, 2]", ("channel_mult", [1, 2, 2])),
        ("dataset=cifar10", ("dataset", "cifar10")),
        ("out_dir=runs/sweep", ("out_dir", "runs/sweep")),
        ("sample_spacing=quadratic", ("sample_spacing", "quadratic")),
        ('device="cuda:1"', ("device", "cuda:1")),
        (" seed = 5", ("seed", 5)),
    ],
)
def test_config_override_types_itself_like_the_config_file(raw, expected):
    assert cli.config_override(raw) == expected


@pytest.mark.parametrize("raw", ["batch_size", "", "=64"])
def test_config_override_rejects_a_malformed_pair(raw):
    with pytest.raises(argparse.ArgumentTypeError, match="field=value"):
        cli.config_override(raw)


def test_set_overrides_the_config_file(tmp_path, trained):
    config = tmp_path / "cfg.toml"
    config.write_text("[data]\nbatch_size = 128\n", encoding="utf-8")

    assert main(["train", "--config", str(config), "--set", "batch_size=64"]) == 0

    assert trained["cfg"].batch_size == 64


def test_set_is_repeatable(tmp_path, trained):
    assert (
        main(
            [
                "train",
                "--set",
                "lr=1e-3",
                "--set",
                "num_workers=0",
                "--set",
                "sample_spacing=quadratic",
            ]
        )
        == 0
    )
    assert trained["cfg"].lr == 1e-3
    assert trained["cfg"].num_workers == 0
    assert trained["cfg"].sample_spacing == "quadratic"


def test_set_coerces_paths_and_tuples(trained):
    assert main(["train", "--set", "out_dir=runs/sweep", "--set", "channel_mult=[1, 2]"]) == 0
    assert trained["cfg"].out_dir == Path("runs/sweep")
    assert trained["cfg"].channel_mult == (1, 2)


def test_set_overrides_a_resumed_config(tmp_path, trained):
    path = _checkpoint(tmp_path, base_channels=32, batch_size=128)

    assert main(["train", "--resume", str(path), "--set", "batch_size=32"]) == 0

    assert trained["cfg"].base_channels == 32
    assert trained["cfg"].batch_size == 32


def test_set_wins_over_a_named_flag_for_the_same_field(trained):
    assert main(["train", "--epochs", "9", "--set", "num_epochs=3"]) == 0
    assert trained["cfg"].num_epochs == 3


def test_set_leaves_the_rest_of_the_config_alone(trained):
    assert main(["train", "--set", "batch_size=64"]) == 0
    default = TrainConfig()
    assert trained["cfg"].lr == default.lr
    assert trained["cfg"].base_channels == default.base_channels


def test_set_reports_an_unknown_field(capsys, trained):
    assert main(["train", "--set", "batch_sizes=64"]) == 1
    err = capsys.readouterr().err
    assert "unknown config field" in err
    assert "batch_sizes" in err


def test_set_still_validates_the_result(capsys, trained):
    assert main(["train", "--set", "batch_size=0"]) == 1
    assert "batch_size must be positive" in capsys.readouterr().err


def test_no_set_leaves_the_config_untouched(tmp_path, trained):
    config = tmp_path / "cfg.toml"
    config.write_text("[data]\nbatch_size = 64\n", encoding="utf-8")

    assert main(["train", "--config", str(config)]) == 0

    assert trained["cfg"].batch_size == 64


def test_sample_and_fid_default_the_spacing_to_the_checkpoints():
    parser = build_parser()
    assert parser.parse_args(["sample", "--checkpoint", "m.pt"]).spacing is None
    assert parser.parse_args(["fid", "--checkpoint", "m.pt"]).spacing is None


def test_sample_parses_the_spacing():
    args = build_parser().parse_args(["sample", "--checkpoint", "m.pt", "--spacing", "quadratic"])
    assert args.spacing == "quadratic"


def test_an_unregistered_spacing_is_refused_by_the_parser():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["sample", "--checkpoint", "m.pt", "--spacing", "linear"])


def test_fid_defaults_to_using_the_cache():
    assert build_parser().parse_args(["fid", "--checkpoint", "m.pt"]).cache is True


def test_no_cache_turns_the_cache_off():
    args = build_parser().parse_args(["fid", "--checkpoint", "m.pt", "--no-cache"])
    assert args.cache is False


def test_fid_defaults_leave_the_opt_in_metrics_off():
    args = build_parser().parse_args(["fid", "--checkpoint", "m.pt"])
    assert args.kid is False
    assert args.precision_recall is False
    assert (args.kid_subsets, args.kid_subset_size) == (
        DEFAULT_KID_SUBSETS,
        DEFAULT_KID_SUBSET_SIZE,
    )
    assert args.neighbours == DEFAULT_NEIGHBOURS


def test_fid_parses_the_metric_flags():
    args = build_parser().parse_args(
        [
            "fid",
            "--checkpoint",
            "m.pt",
            "--kid",
            "--kid-subsets",
            "20",
            "--kid-subset-size",
            "250",
            "--precision-recall",
            "--neighbours",
            "5",
        ]
    )
    assert (args.kid, args.kid_subsets, args.kid_subset_size) == (True, 20, 250)
    assert (args.precision_recall, args.neighbours) == (True, 5)


def test_the_metric_flags_reach_the_scorer(monkeypatch, tmp_path):
    seen = {}

    class FakeResult:
        def format(self):
            return "fid 1.0"

    def fake_fid(checkpoint, **kwargs):
        seen.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(cli_commands, "fid_for_checkpoint", fake_fid)
    main(["fid", "--checkpoint", str(tmp_path / "m.pt"), "--kid", "--precision-recall"])
    assert seen["kid"] is True
    assert seen["precision_recall"] is True
    assert seen["neighbours"] == DEFAULT_NEIGHBOURS


def test_plot_defaults():
    args = build_parser().parse_args(["plot", "runs/mnist"])
    assert args.command == "plot"
    assert args.runs == [Path("runs/mnist")]
    assert args.out == Path("contents/metrics.png")
    assert args.dpi == 120


def test_plot_takes_several_runs():
    args = build_parser().parse_args(["plot", "runs/a", "runs/b", "--out", "fig.svg"])
    assert args.runs == [Path("runs/a"), Path("runs/b")]
    assert args.out == Path("fig.svg")


def test_plot_requires_a_run():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["plot"])


def test_plot_runs_and_reports_where_it_wrote(capsys, monkeypatch, tmp_path):
    seen = {}

    def fake_plot(runs, out, **kwargs):
        seen["runs"], seen["out"] = runs, out
        return out

    monkeypatch.setattr(cli_commands, "plot_runs", fake_plot)
    out = tmp_path / "fig.png"
    assert main(["plot", "runs/mnist", "--out", str(out)]) == 0
    assert str(out) in capsys.readouterr().out
    assert seen["runs"] == [Path("runs/mnist")]


def test_main_reports_a_run_with_no_metrics(pyplot, capsys, tmp_path):
    assert main(["plot", str(tmp_path / "nowhere"), "--out", str(tmp_path / "f.png")]) == 1
    assert "error:" in capsys.readouterr().err


def test_main_reports_a_missing_plots_extra(capsys, tmp_path, monkeypatch):
    """Without the extra, `plot` is a one-line message rather than a traceback.

    The CLI catches ImportError for exactly this, the way `serve` relies on for
    the server extra. Raising anything else out of plot_runs walks straight
    past that handler, and the only install that notices is the one without
    matplotlib.
    """
    real_import = builtins.__import__

    def refuse_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_matplotlib)
    assert main(["plot", str(tmp_path), "--out", str(tmp_path / "f.png")]) == 1
    assert "plots" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["sample", "fid", "interpolate", "serve"])
def test_every_sampling_command_defaults_to_float32(command):
    args = build_parser().parse_args([command, "--checkpoint", "model.pt"])
    assert args.precision == DEFAULT_PRECISION


@pytest.mark.parametrize("command", ["sample", "fid", "interpolate", "serve"])
@pytest.mark.parametrize("name", PRECISIONS)
def test_every_sampling_command_accepts_every_precision(command, name):
    args = build_parser().parse_args([command, "--checkpoint", "m.pt", "--precision", name])
    assert args.precision == name


def test_an_unknown_precision_is_refused_by_the_parser(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["sample", "--checkpoint", "m.pt", "--precision", "float8"])
    assert "invalid choice" in capsys.readouterr().err


def test_precision_reaches_the_sampler(monkeypatch, tmp_path):
    seen = {}

    def fake_sample(checkpoint, out, **kwargs):
        seen.update(kwargs)
        return out

    monkeypatch.setattr(cli_commands, "sample_from_checkpoint", fake_sample)
    assert (
        main(
            [
                "sample",
                "--checkpoint",
                "m.pt",
                "--out",
                str(tmp_path / "s.png"),
                "--precision",
                "bf16",
            ]
        )
        == 0
    )
    assert seen["precision"] == "bf16"


def test_precision_reaches_fid_under_its_own_name(monkeypatch):
    seen = {}

    def fake_fid(checkpoint, **kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(cli_commands, "fid_for_checkpoint", fake_fid)
    with pytest.raises(SystemExit):
        main(["fid", "--checkpoint", "m.pt", "--precision", "fp16"])
    assert seen["sample_precision"] == "fp16"
    assert seen["precision_recall"] is False


def test_tui_defaults():
    args = build_parser().parse_args(["tui", "--config", "configs/mnist.toml"])
    assert args.command == "tui"
    assert args.start is False
    assert (args.resume, args.device, args.num_epochs) == (None,) * 3


def test_tui_takes_the_same_overrides_train_does():
    args = build_parser().parse_args(
        ["tui", "--set", "lr=1e-4", "--set", "batch_size=64", "--epochs", "3", "--start"]
    )
    assert args.overrides == [("lr", 1e-4), ("batch_size", 64)]
    assert args.num_epochs == 3
    assert args.start is True


def test_main_reports_a_missing_tui_extra(capsys, monkeypatch):
    """Without the extra, `tui` is a one-line message rather than a traceback.

    The same contract `plot` and `serve` have: an optional dependency that is
    not installed is a user error the CLI reports, and ImportError is what it
    catches to do it.
    """
    real_import = builtins.__import__

    def refuse_textual(name, *args, **kwargs):
        if name.startswith("textual"):
            raise ImportError("no textual")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "tinydiffusion.tui.app", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse_textual)
    assert main(["tui"]) == 1
    assert "tui" in capsys.readouterr().err


def test_the_dashboard_silences_the_console_backend(monkeypatch):
    """stdout belongs to the display, so the per-epoch table has to be off.

    Left on, every flush would be printed straight through the widgets. The
    JSONL backend is untouched, so the run still records what it always did.
    """
    seen = {}

    def fake_run_tui(cfg, resume=None, *, autostart=False):
        seen["cfg"] = cfg
        seen["autostart"] = autostart

    monkeypatch.setattr("tinydiffusion.tui.run_tui", fake_run_tui)
    assert main(["tui", "--set", "log_jsonl=true", "--start"]) == 0
    assert seen["cfg"].log_console is False
    assert seen["cfg"].log_jsonl is True
    assert seen["autostart"] is True


def test_interpolate_defaults():
    args = build_parser().parse_args(["interpolate", "--checkpoint", "m.pt"])
    assert args.command == "interpolate"
    assert args.steps == 8
    assert (args.seed_start, args.seed_end) == (0, 1)
    assert args.out == Path("contents/interpolation.png")
    assert (args.num_steps, args.sampler, args.spacing, args.labels) == (None,) * 4
    assert (args.guidance, args.guidance_rescale) == (None, None)


def test_interpolate_parses_its_overrides():
    args = build_parser().parse_args(
        [
            "interpolate",
            "--checkpoint",
            "m.pt",
            "--steps",
            "12",
            "--denoise-steps",
            "30",
            "--labels",
            "7",
            "--seed-start",
            "4",
            "--seed-end",
            "9",
            "--guidance",
            "3",
        ]
    )
    assert (args.steps, args.num_steps, args.labels) == (12, 30, [7])
    assert (args.seed_start, args.seed_end, args.guidance) == (4, 9, 3.0)


def test_interpolate_requires_a_checkpoint():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["interpolate"])


def test_interpolate_runs_and_reports_where_it_wrote(capsys, monkeypatch, tmp_path):
    seen = {}

    def fake_walk(checkpoint, out, **kwargs):
        seen["checkpoint"], seen["out"] = checkpoint, out
        seen.update(kwargs)
        return out

    monkeypatch.setattr(cli_commands, "interpolate_from_checkpoint", fake_walk)
    out = tmp_path / "walk.png"
    assert main(["interpolate", "--checkpoint", "m.pt", "--steps", "6", "--out", str(out)]) == 0
    assert str(out) in capsys.readouterr().out
    assert seen["steps"] == 6
    assert seen["seed_start"] == 0


def test_main_reports_a_missing_interpolate_checkpoint(capsys, tmp_path):
    assert main(["interpolate", "--checkpoint", str(tmp_path / "nope.pt")]) == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["sample", "fid", "interpolate"])
def test_karras_spacing_is_offered_everywhere_a_spacing_is(command):
    args = build_parser().parse_args([command, "--checkpoint", "m.pt", "--spacing", "karras"])
    assert args.spacing == "karras"
