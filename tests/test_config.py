from pathlib import Path

import pytest

from tinydiffusion.training.config import TrainConfig, load_config


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cfg.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_config_matches_the_defaults():
    cfg = load_config(Path("configs/mnist.toml"))
    defaults = TrainConfig()
    assert cfg.channel_mult == defaults.channel_mult
    assert cfg.num_epochs == defaults.num_epochs
    assert cfg.data_root == defaults.data_root


def test_tables_are_flattened_and_coerced(tmp_path):
    cfg = load_config(
        write(
            tmp_path,
            """
            [model]
            channel_mult = [1, 2]
            [bookkeeping]
            out_dir = "runs/one"
            """,
        )
    )
    assert cfg.channel_mult == (1, 2)
    assert cfg.out_dir == Path("runs/one")


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown config field"):
        load_config(write(tmp_path, "[model]\nbase = 64\n"))


def test_duplicate_key_across_tables_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="both"):
        load_config(write(tmp_path, "[a]\nseed = 1\n[b]\nseed = 2\n"))


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError, match="unknown schedule"):
        TrainConfig(schedule="quadratic")
    with pytest.raises(ValueError, match="sample_steps"):
        TrainConfig(sample_steps=0)
    with pytest.raises(ValueError, match="num_samples"):
        TrainConfig(num_samples=0)


def test_from_mapping_round_trips_a_checkpoint_config():
    # save_checkpoint stringifies Paths; from_mapping must undo that.
    cfg = TrainConfig(out_dir=Path("runs/two"), channel_mult=(1, 2))
    stored = {"out_dir": "runs/two", "channel_mult": (1, 2), "device": "cpu"}
    restored = TrainConfig.from_mapping(stored)
    assert restored.out_dir == cfg.out_dir
    assert restored.channel_mult == cfg.channel_mult
