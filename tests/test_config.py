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


def test_the_shipped_configs_train_conditionally():
    # Conditioning is the one place mnist.toml departs from the defaults, and
    # smoke.toml carries it too so the smoke run covers the conditional path.
    for name in ("mnist", "smoke"):
        cfg = load_config(Path(f"configs/{name}.toml"))
        assert cfg.num_classes == 10
        assert cfg.class_dropout > 0
        assert cfg.guidance > 1.0


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


def test_invalid_conditioning_is_rejected():
    with pytest.raises(ValueError, match="num_classes"):
        TrainConfig(num_classes=0)
    with pytest.raises(ValueError, match="class_dropout"):
        TrainConfig(class_dropout=1.0)
    with pytest.raises(ValueError, match="guidance"):
        TrainConfig(guidance=-1.0)


def test_guidance_needs_a_conditional_model():
    with pytest.raises(ValueError, match="needs a conditional model"):
        TrainConfig(guidance=2.0)


def test_guidance_needs_a_trained_null_token():
    # class_dropout=0 never shows the network the null token, so guidance would
    # extrapolate away from an untrained embedding.
    with pytest.raises(ValueError, match="class_dropout"):
        TrainConfig(num_classes=10, class_dropout=0.0, guidance=2.0)
    # Plain conditional sampling is fine without it.
    assert TrainConfig(num_classes=10, class_dropout=0.0).guidance == 1.0


def test_from_mapping_round_trips_a_checkpoint_config():
    # save_checkpoint stringifies Paths; from_mapping must undo that.
    cfg = TrainConfig(out_dir=Path("runs/two"), channel_mult=(1, 2))
    stored = {"out_dir": "runs/two", "channel_mult": (1, 2), "device": "cpu"}
    restored = TrainConfig.from_mapping(stored)
    assert restored.out_dir == cfg.out_dir
    assert restored.channel_mult == cfg.channel_mult
