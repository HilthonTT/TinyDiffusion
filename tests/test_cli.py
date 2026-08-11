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
    assert args.config == "cfg.toml"
    assert args.seed == 7


def test_sample_defaults():
    args = build_parser().parse_args(["sample", "--checkpoint", "model.pt"])
    assert args.num_images == 8


def test_main_reports_unimplemented(capsys):
    assert main(["train", "--config", "cfg.toml"]) == 1
    assert "not implemented" in capsys.readouterr().out
