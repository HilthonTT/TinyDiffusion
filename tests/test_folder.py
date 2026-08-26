"""Training on a directory of your own images."""

import shutil

import pytest
from PIL import Image

from tinydiffusion.data import (
    FOLDER_DATASET,
    dataset_names,
    dataset_spec,
    folder_spec,
    image_dataloader,
    load_folder,
    scan_folder,
)
from tinydiffusion.data.datasets import image_transform
from tinydiffusion.training.checkpoints import ARCHITECTURE_FIELDS
from tinydiffusion.training.config import TrainConfig


def _write(path, *, size=(32, 32), mode="RGB"):
    """Put one image on disk, making its directory if it is not there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size).save(path)
    return path


@pytest.fixture
def flat(tmp_path):
    """A directory of loose images: unlabelled data."""
    root = tmp_path / "flat"
    for index in range(20):
        _write(root / f"img{index:03d}.png")
    return root


@pytest.fixture
def classed(tmp_path):
    """One subdirectory per class, 10 images each."""
    root = tmp_path / "classed"
    for name in ("cats", "dogs", "emus"):
        for index in range(10):
            _write(root / name / f"{name}{index}.png")
    return root


def test_folder_is_a_name_a_config_may_give_but_not_a_registry_entry():
    # It has no fixed spec to look up, so the lookup says where to get one
    # rather than reporting the name as unknown.
    assert FOLDER_DATASET in dataset_names()
    with pytest.raises(ValueError, match=r"no fixed spec.*folder_spec"):
        dataset_spec(FOLDER_DATASET)


def test_a_flat_directory_is_unlabelled_data(flat):
    scan = scan_folder(flat, holdout=0.0)

    assert len(scan.paths) == 20
    assert scan.classes == []
    assert set(scan.labels) == {0}


def test_subdirectories_become_classes_numbered_by_sorted_name(classed):
    scan = scan_folder(classed, holdout=0.0)

    assert scan.classes == ["cats", "dogs", "emus"]
    assert scan.num_classes == 3
    # Every class contributes, and the label is its directory's index.
    pairs = zip(scan.paths, scan.labels, strict=True)
    assert {label: path.parent.name for path, label in pairs} == {0: "cats", 1: "dogs", 2: "emus"}


def test_loose_images_beside_class_directories_are_rejected(classed):
    _write(classed / "stray.png")

    with pytest.raises(ValueError, match=r"both.*loose image.*class director"):
        scan_folder(classed)


def test_the_two_splits_partition_the_directory(flat):
    train = scan_folder(flat, train=True, holdout=0.25)
    test = scan_folder(flat, train=False, holdout=0.25)

    assert set(train.paths).isdisjoint(test.paths)
    assert len(train.paths) + len(test.paths) == 20
    assert test.paths, "a nonzero holdout has to hold something out"


def test_adding_an_image_moves_only_that_image_between_splits(flat):
    # The split is a hash of each path, not a cut through a sorted list: an
    # index-based split would move every image after the insertion point, and
    # quietly promote already-scored images into the training set.
    before = set(scan_folder(flat, train=False, holdout=0.5).paths)
    added = _write(flat / "img999.png")
    after = set(scan_folder(flat, train=False, holdout=0.5).paths)

    assert after - before <= {added}
    assert not before - after


def test_the_split_is_the_same_on_every_platform(flat):
    # Keyed on the POSIX relative path, so a run on Windows scores the same
    # images a run on Linux does — and a cached FID keyed on the split stays
    # valid across both.
    held_out = sorted(path.name for path in scan_folder(flat, train=False, holdout=0.5).paths)

    assert held_out == [
        "img001.png",
        "img003.png",
        "img004.png",
        "img005.png",
        "img009.png",
        "img010.png",
        "img011.png",
        "img012.png",
        "img014.png",
        "img017.png",
        "img018.png",
    ]


def test_a_zero_holdout_keeps_everything_for_training(flat):
    assert len(scan_folder(flat, train=True, holdout=0.0).paths) == 20
    with pytest.raises(ValueError, match=r"held-out split.*is empty"):
        scan_folder(flat, train=False, holdout=0.0)


def test_explicit_split_directories_are_used_verbatim(tmp_path):
    root = tmp_path / "split"
    for index in range(12):
        _write(root / "train" / "a" / f"{index}.png")
    for index in range(3):
        _write(root / "test" / "a" / f"{index}.png")

    # A holdout that would otherwise cut the training split in half is ignored:
    # the directory already says what the splits are.
    assert len(scan_folder(root, train=True, holdout=0.5).paths) == 12
    assert len(scan_folder(root, train=False, holdout=0.5).paths) == 3


def test_val_is_accepted_in_place_of_test(tmp_path):
    root = tmp_path / "split"
    _write(root / "train" / "a.png")
    _write(root / "val" / "b.png")

    assert [path.name for path in scan_folder(root, train=False).paths] == ["b.png"]


def test_a_train_directory_without_a_held_out_one_is_an_error(tmp_path):
    # Falling back to the hash split here would score the held-out loss on the
    # training images and never say so.
    root = tmp_path / "split"
    _write(root / "train" / "a.png")

    with pytest.raises(ValueError, match=r"no test/ or val/"):
        scan_folder(root, train=False)


def test_non_images_and_hidden_directories_are_not_training_data(flat):
    (flat / "notes.txt").write_text("not an image")
    _write(flat / ".cache" / "hidden.png")
    # What `fid` leaves behind in the dataset root once it has run.
    (flat / "fid_cache").mkdir()
    (flat / "fid_cache" / "stats.pt").write_bytes(b"not an image")

    scan = scan_folder(flat, holdout=0.0)

    assert len(scan.paths) == 20
    assert scan.classes == [], "a cache directory is not a class"


def test_a_missing_directory_says_nothing_will_be_downloaded(tmp_path):
    with pytest.raises(ValueError, match=r"does not exist.*does not download"):
        scan_folder(tmp_path / "absent")


def test_an_empty_directory_says_what_it_looked_for(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(ValueError, match=r"no images found.*png"):
        scan_folder(tmp_path / "empty")


def test_a_class_count_that_does_not_match_the_directory_is_caught_at_load(classed, flat):
    transform = image_transform(3, 32, crop=True)

    with pytest.raises(ValueError, match=r"num_classes=5 but.*3 class"):
        load_folder(
            classed, train=True, channels=3, num_classes=5, holdout=0.0, transform=transform
        )
    with pytest.raises(ValueError, match=r"num_classes=2 but.*no class subdirectories"):
        load_folder(flat, train=True, channels=3, num_classes=2, holdout=0.0, transform=transform)


def test_an_unconditional_run_does_not_have_to_match_the_layout(classed):
    # Same as num_classes=None on MNIST: the labels are still read and still
    # ignored, so class directories are no obstacle to unconditional training.
    dataset = load_folder(
        classed,
        train=True,
        channels=3,
        num_classes=None,
        holdout=0.0,
        transform=image_transform(3, 32, crop=True),
    )

    assert len(dataset) == 30


@pytest.mark.parametrize("channels", [1, 3])
def test_images_are_converted_to_the_declared_channel_count(tmp_path, channels):
    # A directory of photographs is rarely uniform, and the U-Net's input width
    # is fixed by the config rather than by whatever the last file happened to be.
    root = tmp_path / "mixed"
    _write(root / "grey.png", mode="L")
    _write(root / "rgb.png", mode="RGB")
    _write(root / "alpha.png", mode="RGBA")

    loader = image_dataloader(
        folder_spec(channels=channels, holdout=0.0),
        root,
        batch_size=3,
        image_size=32,
        num_workers=0,
    )
    images, _ = next(iter(loader))

    assert images.shape == (3, channels, 32, 32)
    assert images.min() >= -1.0 and images.max() <= 1.0


def test_oblong_images_are_cropped_to_square(tmp_path):
    # Resize takes a single length to mean the short side, so without the crop
    # a 4:3 photo stays 4:3 and no two images in the batch share a shape.
    root = tmp_path / "photos"
    for index, size in enumerate([(96, 72), (72, 96), (40, 40)]):
        _write(root / f"{index}.png", size=size)

    loader = image_dataloader(
        folder_spec(channels=3, holdout=0.0), root, batch_size=3, image_size=32, num_workers=0
    )
    images, _ = next(iter(loader))

    assert images.shape == (3, 3, 32, 32)


def test_the_packaged_datasets_do_not_crop():
    # Already square, so the crop would be a no-op on every image.
    assert not dataset_spec("mnist").crop
    assert folder_spec().crop


def test_a_folder_config_resolves_without_touching_the_disk():
    # The point of declaring rather than detecting: a checkpoint trained on a
    # folder has to stay loadable, and samplable, where the images never were.
    cfg = TrainConfig(dataset=FOLDER_DATASET, data_root="nowhere-at-all", folder_channels=1)
    spec = cfg.dataset_spec()

    assert spec.name == FOLDER_DATASET
    assert spec.channels == 1
    assert spec.num_classes == 0


def test_the_config_carries_the_folder_settings_into_the_spec():
    cfg = TrainConfig(
        dataset=FOLDER_DATASET,
        image_size=64,
        num_classes=3,
        class_dropout=0.1,
        folder_channels=3,
        folder_hflip=False,
    )
    spec = cfg.dataset_spec()

    assert (spec.channels, spec.native_size, spec.num_classes, spec.hflip) == (3, 64, 3, False)


def test_the_folder_channel_count_is_tied_to_the_weights():
    # It is the U-Net's input and output width, so a --resume that changes it
    # has to be refused with the other architecture fields rather than reaching
    # load_state_dict as a size mismatch.
    assert "folder_channels" in ARCHITECTURE_FIELDS


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("folder_channels", 2, "folder_channels must be 1 or 3"),
        ("folder_holdout", 1.0, r"folder_holdout must lie in \[0, 1\)"),
        ("folder_holdout", -0.1, r"folder_holdout must lie in \[0, 1\)"),
    ],
)
def test_the_folder_settings_are_checked_while_the_config_is_read(field, value, message):
    with pytest.raises(ValueError, match=message):
        TrainConfig(dataset=FOLDER_DATASET, **{field: value})


def test_a_folder_run_trains_end_to_end(tmp_path, classed):
    # The whole path in one: scan, split, load, train an epoch, score the
    # held-out split and write a checkpoint that samples without the images.
    from tinydiffusion.sampling import sample_from_checkpoint
    from tinydiffusion.training.train import train

    cfg = TrainConfig(
        dataset=FOLDER_DATASET,
        data_root=classed,
        image_size=16,
        batch_size=8,
        num_workers=0,
        base_channels=8,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attn_resolutions=(),
        num_classes=3,
        class_dropout=0.1,
        num_timesteps=20,
        num_epochs=1,
        sample_steps=4,
        num_samples=4,
        val_steps=2,
        val_batches=1,
        device="cpu",
        amp=False,
        out_dir=tmp_path / "out",
        ckpt_dir=tmp_path / "ckpt",
        log_dir=tmp_path / "logs",
        log_console=False,
        log_jsonl=False,
    )
    train(cfg)

    checkpoint = tmp_path / "ckpt" / "last.pt"
    assert checkpoint.is_file()

    # The images go away, and the checkpoint still samples: nothing downstream
    # of training needs the directory the run was named after.
    shutil.rmtree(classed)
    grid = sample_from_checkpoint(
        checkpoint, tmp_path / "gen.png", num_images=2, num_steps=2, device="cpu"
    )

    assert grid.is_file()
