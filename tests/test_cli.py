from pathlib import Path

import pytest

from tinydiffusion import __version__
from tinydiffusion import version as version_module
from tinydiffusion.cli import build_parser, main


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
    # Tracking flags too: an unpassed flag must not override the config file.
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
    # Unset, so the checkpoint's own conditioning settings decide.
    assert args.labels is None
    assert args.guidance is None


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


def test_sample_parses_the_guidance_scale():
    args = build_parser().parse_args(["sample", "--checkpoint", "m.pt", "--guidance", "2.5"])
    assert args.guidance == 2.5


def test_main_reports_a_missing_config(capsys, tmp_path):
    assert main(["train", "--config", str(tmp_path / "nope.toml")]) == 1
    assert "error:" in capsys.readouterr().out


def test_main_reports_a_missing_checkpoint(capsys, tmp_path):
    assert main(["sample", "--checkpoint", str(tmp_path / "nope.pt")]) == 1
    assert "error:" in capsys.readouterr().out
