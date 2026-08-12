from pathlib import Path

import pytest

from tinydiffusion import __version__
from tinydiffusion.cli import build_parser, main


def test_version_is_exposed():
    assert __version__


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


def test_sample_defaults():
    args = build_parser().parse_args(["sample", "--checkpoint", "model.pt"])
    assert args.num_images == 8
    assert args.eta == 0.0
    assert args.steps is None


def test_main_reports_a_missing_config(capsys, tmp_path):
    assert main(["train", "--config", str(tmp_path / "nope.toml")]) == 1
    assert "error:" in capsys.readouterr().out


def test_main_reports_a_missing_checkpoint(capsys, tmp_path):
    assert main(["sample", "--checkpoint", str(tmp_path / "nope.pt")]) == 1
    assert "error:" in capsys.readouterr().out
