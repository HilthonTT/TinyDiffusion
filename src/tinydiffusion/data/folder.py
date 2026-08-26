"""Training on a directory of your own images, rather than a packaged dataset.

The three entries in :data:`~tinydiffusion.data.datasets.DATASETS` are
torchvision downloads whose channel count, label space and split are known
before anything touches the disk. A folder of images is not: those facts live
in the directory, and the directory may not be there at all by the time a
checkpoint is loaded somewhere else to sample from.

So the split of responsibility here is *declare, then verify*. The config
declares what the folder holds — ``folder_channels``, ``num_classes`` — and
:func:`folder_spec` turns that declaration into an ordinary
:class:`~tinydiffusion.data.datasets.DatasetSpec` without reading anything.
:func:`load_folder` is what finally opens the directory, and it is where a
declaration that does not match what is on disk is caught. That keeps
``TrainConfig`` constructible, and a checkpoint loadable, on a machine that has
never seen the images.

Two layouts are understood, and which one is in use is inferred rather than
configured:

.. code-block:: text

    photos/                     photos/
        img001.png                  cats/
        img002.png                      img001.png
        ...                         dogs/
                                        img002.png

The flat one is unconditional data — every image gets label 0. The second gives
each immediate subdirectory a class, numbered by sorted name, which is what a
conditional run's ``num_classes`` has to agree with. A directory holding both
loose images and class subdirectories is rejected rather than guessed at.

Neither layout carries a train/test split, and everything downstream wants one:
``val_every`` scores a held-out slice, ``eval`` and ``fid`` take ``--split``.
If the directory has ``train/`` and ``test/`` (or ``val/``) subdirectories they
are used as the splits verbatim. Otherwise a fraction of the images —
``folder_holdout`` — is held back, chosen by hashing each image's path. Hashing
rather than slicing a sorted list is what makes the split stable: adding one
photo moves that photo alone between splits, where an index-based cut would
reshuffle every image after it and quietly move already-scored images into the
training set.
"""

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

__all__ = [
    "FOLDER_DATASET",
    "IMAGE_SUFFIXES",
    "FolderScan",
    "ImageFolderDataset",
    "ImageTransform",
    "load_folder",
    "scan_folder",
]

FOLDER_DATASET = "folder"
"""The name a config gives ``dataset`` to train on :attr:`~TrainConfig.data_root`."""

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".tif", ".tiff", ".webp"})
"""File extensions treated as images. Everything else in the directory is ignored.

Deliberately not "whatever Pillow will open": a directory of photos usually has
a ``.DS_Store``, a ``notes.txt`` and — once ``fid`` has run once — a
``fid_cache/`` of tensors in it, and none of those should become a training
image or, worse, a class.
"""

_SPLIT_DIRS = {True: ("train",), False: ("test", "val")}
"""Subdirectory names that mark an explicit split, by whether it is the training one."""

ImageTransform = Callable[[Image.Image], torch.Tensor]
"""What :func:`~tinydiffusion.data.datasets.image_transform` builds: PIL in, tensor out."""


class FolderScan:
    """What one pass over a directory found.

    Attributes:
        paths: the images in this split, in a fixed order.
        labels: each image's class index, all 0 when the layout is flat.
        classes: class directory names in label order, empty when flat.
        root: the directory that was scanned, for error messages.
    """

    __slots__ = ("classes", "labels", "paths", "root")

    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        classes: list[str],
        root: Path,
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.classes = classes
        self.root = root

    @property
    def num_classes(self) -> int:
        """How many classes the layout describes, 0 for a flat directory."""
        return len(self.classes)


def _is_image(path: Path) -> bool:
    """Whether `path` is a file this module will try to open."""
    return path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()


def _visible(path: Path, relative_to: Path) -> bool:
    """Whether no part of `path` below `relative_to` is a dot-entry.

    Hidden directories hold caches and editor state, not training data, and
    ``fid`` writes its reference statistics inside the dataset root — so an
    unfiltered walk would eventually train on whatever the last tool left there.
    """
    return not any(part.startswith(".") for part in path.relative_to(relative_to).parts)


def _images_under(directory: Path) -> list[Path]:
    """Every image anywhere under `directory`, in a platform-independent order.

    Args:
        directory: the tree to walk.

    Returns:
        Sorted paths. Sorted by POSIX string rather than by
        :class:`~pathlib.Path`, whose ordering is case-insensitive on Windows
        and not elsewhere — the order decides which images a truncated
        ``--num-images`` run scores, so it has to be the same on both.
    """
    found = [path for path in directory.rglob("*") if _is_image(path) and _visible(path, directory)]
    return sorted(found, key=lambda path: path.as_posix())


def _class_dirs(root: Path) -> list[Path]:
    """The immediate subdirectories of `root` that hold at least one image.

    An empty directory, or one holding only a cache, is not a class: it would
    otherwise take a label that nothing ever trains and shift every label after
    it.

    Args:
        root: the directory to look in.

    Returns:
        Sorted by name, which is what fixes the label numbering.
    """
    subdirs = (path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    return sorted((path for path in subdirs if _images_under(path)), key=lambda path: path.name)


def _split_root(root: Path, *, train: bool) -> Path | None:
    """The subdirectory holding `train`'s split, if the layout has explicit ones.

    Args:
        root: the dataset directory.
        train: which split is wanted.

    Returns:
        The split's directory, or None if `root` does not use explicit splits.

    Raises:
        ValueError: if ``train/`` exists but the other split's directory does
            not. Falling back to the hash split there would score held-out loss
            on the training images and say nothing about it.
    """
    if not (root / "train").is_dir():
        return None
    for name in _SPLIT_DIRS[train]:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    wanted = " or ".join(f"{name}/" for name in _SPLIT_DIRS[train])
    raise ValueError(
        f"{root} has a train/ directory but no {wanted}, so there is no held-out split "
        f"to score on; add one, or remove train/ and let folder_holdout split the images"
    )


def _in_holdout(relative: str, holdout: float) -> bool:
    """Whether the image at `relative` falls in the held-out fraction.

    Args:
        relative: the image's path below the dataset root, POSIX-style so the
            same image lands in the same split on every platform.
        holdout: the fraction to hold out, in [0, 1).

    Returns:
        True if this image belongs to the test split.
    """
    if holdout <= 0.0:
        return False
    digest = hashlib.blake2b(relative.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") < holdout * float(2**64)


def scan_folder(root: str | Path, *, train: bool = True, holdout: float = 0.1) -> FolderScan:
    """Find one split's images in a directory, and what their labels are.

    Args:
        root: the dataset directory. See the module docstring for the two
            layouts, and for how the split is decided.
        train: return the training split rather than the held-out one.
        holdout: fraction of the images to hold out, when the directory does
            not carry explicit ``train/`` and ``test/`` subdirectories. Ignored
            when it does.

    Returns:
        The images, labels and class names for that split.

    Raises:
        ValueError: if the directory is missing, holds no images, mixes loose
            images with class subdirectories, or leaves the requested split
            empty.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(
            f"dataset directory {root} does not exist; point data_root at a directory of "
            f"images (dataset='folder' does not download anything)"
        )
    if not 0.0 <= holdout < 1.0:
        raise ValueError(f"holdout must lie in [0, 1), got {holdout}")

    explicit = _split_root(root, train=train)
    scan_root = explicit if explicit is not None else root

    classes = _class_dirs(scan_root)
    loose = sorted(
        (path for path in scan_root.glob("*") if _is_image(path)),
        key=lambda path: path.as_posix(),
    )
    if classes and loose:
        raise ValueError(
            f"{scan_root} holds both {len(loose)} loose image(s) and the class "
            f"director{'y' if len(classes) == 1 else 'ies'} "
            f"{', '.join(path.name for path in classes)}; a folder dataset is either "
            f"flat (unlabelled) or one subdirectory per class, not both"
        )

    if classes:
        paths: list[Path] = []
        labels: list[int] = []
        for label, directory in enumerate(classes):
            found = _images_under(directory)
            paths += found
            labels += [label] * len(found)
        names = [directory.name for directory in classes]
    else:
        paths = _images_under(scan_root)
        labels = [0] * len(paths)
        names = []

    if not paths:
        raise ValueError(
            f"no images found under {scan_root} (looked for {', '.join(sorted(IMAGE_SUFFIXES))})"
        )

    if explicit is None:
        # One hash per image decides its split, so both calls see the same
        # partition without either having to know about the other.
        wanted = [
            (path, label)
            for path, label in zip(paths, labels, strict=True)
            if _in_holdout(path.relative_to(scan_root).as_posix(), holdout) is not train
        ]
        if not wanted:
            split = "training" if train else "held-out"
            raise ValueError(
                f"the {split} split of {scan_root} is empty: {len(paths)} image(s) at "
                f"folder_holdout={holdout}. Lower it, raise it, or add more images"
            )
        paths = [path for path, _ in wanted]
        labels = [label for _, label in wanted]

    return FolderScan(paths, labels, names, scan_root)


class ImageFolderDataset(Dataset[tuple[torch.Tensor, int]]):
    """A list of image files on disk, read one at a time.

    Decoding happens in :meth:`__getitem__` rather than up front, so the memory
    cost is one batch however large the directory is, and the dataloader's
    worker processes are what absorb the decode. Only the path list crosses
    into a worker, which is why this holds paths and a transform and nothing
    that would not pickle.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        labels: Sequence[int],
        *,
        channels: int,
        transform: ImageTransform,
        classes: Sequence[str] = (),
    ) -> None:
        """Wrap an already-scanned list of images.

        Args:
            paths: the image files, in the order they should be visited.
            labels: each image's class index, the same length as `paths`.
            channels: 1 to read as greyscale, 3 as RGB. Whatever the files hold
                is converted to this, since the U-Net's input width is fixed by
                the config and a directory of photos is rarely uniform.
            transform: the pipeline from
                :func:`~tinydiffusion.data.datasets.image_transform`.
            classes: class directory names in label order, kept for messages.

        Raises:
            ValueError: if `paths` and `labels` are different lengths, or
                `channels` is neither 1 nor 3.
        """
        if len(paths) != len(labels):
            raise ValueError(f"got {len(paths)} paths but {len(labels)} labels")
        if channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {channels}")
        self.paths = list(paths)
        self.labels = list(labels)
        self.classes = list(classes)
        self.channels = channels
        self.transform = transform
        self._mode = "L" if channels == 1 else "RGB"

    def __len__(self) -> int:
        """How many images are in this split."""
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Read, convert and transform one image.

        Args:
            index: position in the split.

        Returns:
            ``(image, label)`` with image in [-1, 1] at the configured
            resolution and channel count.
        """
        path = self.paths[index]
        with Image.open(path) as handle:
            # Phone cameras store the rotation in EXIF rather than in the
            # pixels, so without this a portrait photo trains as a landscape one.
            image = ImageOps.exif_transpose(handle)
            image = (image if image is not None else handle).convert(self._mode)
        return self.transform(image), self.labels[index]


def load_folder(
    root: str | Path,
    *,
    train: bool,
    channels: int,
    num_classes: int | None,
    holdout: float,
    transform: ImageTransform,
) -> ImageFolderDataset:
    """Scan a directory and wrap it as a dataset, checking it is what was declared.

    This is where a config's description of the folder meets the folder. The
    config could not check it — see the module docstring — so the mismatch is
    caught here, before the first epoch rather than at the embedding table.

    Args:
        root: the dataset directory.
        train: load the training split.
        channels: 1 or 3, from ``folder_channels``.
        num_classes: the config's class count, or None for an unconditional
            run. When set it has to equal the number of class subdirectories.
        holdout: fraction held out when the layout has no explicit splits.
        transform: the preprocessing pipeline to apply to each image.

    Returns:
        The split, ready for a dataloader.

    Raises:
        ValueError: if the directory cannot be scanned, or holds a different
            number of classes than `num_classes` claims.
    """
    scan = scan_folder(root, train=train, holdout=holdout)
    if num_classes is not None and num_classes != scan.num_classes:
        found = (
            f"{scan.num_classes} class subdirector"
            f"{'y' if scan.num_classes == 1 else 'ies'} ({', '.join(scan.classes)})"
            if scan.classes
            else "no class subdirectories, only loose images"
        )
        raise ValueError(
            f"num_classes={num_classes} but {scan.root} has {found}; a conditional run "
            f"takes its labels from the directory layout, so either match the count or "
            f"leave num_classes unset to train unconditionally"
        )
    return ImageFolderDataset(
        scan.paths,
        scan.labels,
        channels=channels,
        transform=transform,
        classes=scan.classes,
    )
