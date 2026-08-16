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
    with pytest.raises(ValueError, match="lr_warmup"):
        TrainConfig(lr_warmup=-1)


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


def test_the_default_config_names_a_registered_dataset():
    cfg = TrainConfig()
    assert cfg.dataset_spec().name == cfg.dataset


def test_an_unregistered_dataset_is_refused_up_front():
    with pytest.raises(ValueError, match="unknown dataset 'imagenet'"):
        TrainConfig(dataset="imagenet")


def test_the_class_count_has_to_match_the_datasets_label_space():
    # The labels come from the dataset, so a smaller count indexes past the
    # embedding table the moment a batch carries a higher one.
    with pytest.raises(ValueError, match="does not match mnist"):
        TrainConfig(dataset="mnist", num_classes=4)


def test_a_dataset_may_be_trained_unconditionally():
    assert TrainConfig(dataset="cifar10", num_classes=None).num_classes is None


def test_switching_dataset_changes_the_channel_count_the_model_is_built_from():
    assert TrainConfig(dataset="mnist").dataset_spec().channels == 1
    assert TrainConfig(dataset="cifar10").dataset_spec().channels == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"grad_accum": 0}, "grad_accum"),
        ({"weight_decay": -0.1}, "weight_decay"),
        ({"betas": (0.9,)}, "betas"),
        ({"betas": (0.9, 1.0)}, "betas"),
        ({"amp_dtype": "fp8"}, "amp_dtype"),
        ({"lr_schedule": "linear"}, "lr_schedule"),
    ],
)
def test_bad_optimisation_settings_are_refused_when_the_config_is_read(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TrainConfig(**kwargs)


def test_betas_survive_the_toml_round_trip(tmp_path):
    # TOML has no tuple, so an uncoerced list would reach AdamW as a list and
    # compare unequal to the checkpoint's provenance on resume.
    path = tmp_path / "c.toml"
    path.write_text("""
    [optimisation]
    betas = [0.85, 0.995]
    """)
    assert load_config(path).betas == (0.85, 0.995)


def test_the_performance_settings_default_to_the_conservative_choice():
    # Each is a behaviour or throughput change the user should opt into.
    cfg = TrainConfig()
    assert (cfg.compile, cfg.channels_last, cfg.grad_accum) == (False, False, 1)
    assert (cfg.amp_dtype, cfg.lr_schedule, cfg.weight_decay) == ("fp16", "constant", 0.0)
