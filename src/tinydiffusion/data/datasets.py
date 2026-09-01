"""What a run can train on: the dataset registry, and loading from it.

Everything downstream of a config — the U-Net's channel count, the shape the
samplers draw, the reference side of a FID — reads what it needs from the
:class:`DatasetSpec` the config names, rather than from a constant. Adding a
dataset is therefore an entry in :data:`DATASETS` and nothing else.

The three registered here share torchvision's ``(root, train, download,
transform)`` constructor, which is what lets one builder serve all of them.

:data:`~tinydiffusion.data.folder.FOLDER_DATASET` is the fourth name a config
may give, and the one that is not in :data:`DATASETS`: a directory of your own
images has no fixed channel count or label space, so its spec is built from the
config by :func:`folder_spec` rather than looked up. See
:mod:`tinydiffusion.data.folder`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

from tinydiffusion.data.folder import FOLDER_DATASET, load_folder

__all__ = [
    "DATASETS",
    "DEFAULT_DATASET",
    "FOLDER_DATASET",
    "DatasetSpec",
    "dataset_names",
    "dataset_spec",
    "denormalize",
    "folder_spec",
    "image_dataloader",
    "image_dataset",
    "image_transform",
    "set_loader_epoch",
]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Everything a run needs to know about a dataset before it loads one.

    Attributes:
        name: the key it is registered under, and what a config names.
        channels: image channels. Sets the U-Net's input and output width, so
            it is part of what a checkpoint's weights are tied to.
        native_size: the resolution the images ship at, before any resize.
        num_classes: how many classes the labels span. A conditional run's
            ``num_classes`` has to agree with this, since the labels come from
            here.
        hflip: whether a horizontal flip is a label-preserving augmentation.
            True for natural images; false for anything where the mirror image
            means something else, which is why the digit sets opt out.
        crop: whether the images need cropping to square after the resize. The
            packaged datasets are already square, so a resize is the whole job;
            a directory of photographs is not, and ``Resize`` alone matches the
            *short* side and leaves the long one over — which the collate then
            fails on, since no two ragged images share a shape.
        builder: dataset class or factory taking ``(root, train, download,
            transform)``.
    """

    name: str
    channels: int
    native_size: int
    num_classes: int
    hflip: bool
    builder: Callable[..., Dataset[tuple[torch.Tensor, int]]]
    crop: bool = False


DATASETS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist",
        channels=1,
        native_size=28,
        num_classes=10,
        hflip=False,
        builder=MNIST,
    ),
    "fashion_mnist": DatasetSpec(
        name="fashion_mnist",
        channels=1,
        native_size=28,
        num_classes=10,
        hflip=True,
        builder=FashionMNIST,
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        channels=3,
        native_size=32,
        num_classes=10,
        hflip=True,
        builder=CIFAR10,
    ),
}
"""The datasets a config may name, keyed by that name."""

DEFAULT_DATASET = "mnist"
"""What a config trains on unless it says otherwise."""


def dataset_names() -> tuple[str, ...]:
    """Every dataset name a config may give, sorted.

    Returns:
        The keys of :data:`DATASETS` plus
        :data:`~tinydiffusion.data.folder.FOLDER_DATASET`, for error messages
        and CLI choices. The last one names no registry entry — see
        :func:`folder_spec` — but it is a name ``dataset`` accepts, so leaving
        it out would make ``--dataset folder`` an unknown choice.
    """
    return tuple(sorted((*DATASETS, FOLDER_DATASET)))


def dataset_spec(name: str) -> DatasetSpec:
    """Look up a packaged dataset by name.

    Args:
        name: a key of :data:`DATASETS`.

    Returns:
        The matching spec.

    Raises:
        ValueError: if nothing is registered under `name`, including for
            :data:`~tinydiffusion.data.folder.FOLDER_DATASET`, whose spec
            depends on config fields this has no access to. Callers holding a
            config want :meth:`~tinydiffusion.training.config.TrainConfig.dataset_spec`,
            which resolves both kinds.
    """
    if name == FOLDER_DATASET:
        raise ValueError(
            f"dataset {name!r} has no fixed spec — its channel count and label space come "
            f"from the config's folder_channels and num_classes; build it with folder_spec(), "
            f"or call TrainConfig.dataset_spec()"
        )
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset {name!r}, expected one of: {', '.join(dataset_names())}"
        ) from None


def folder_spec(
    *,
    channels: int = 3,
    image_size: int = 32,
    num_classes: int | None = None,
    hflip: bool = True,
    holdout: float = 0.1,
) -> DatasetSpec:
    """Build the spec for a directory of images, without reading the directory.

    The packaged datasets know their own shape; a folder's is whatever the
    config says it is, and is only checked against the disk when
    :func:`~tinydiffusion.data.folder.load_folder` finally opens it. That
    ordering is deliberate — see :mod:`tinydiffusion.data.folder`.

    Args:
        channels: 1 to read the images as greyscale, 3 as RGB. This is the
            U-Net's input and output width, so it is part of what a
            checkpoint's weights are tied to.
        image_size: the resolution the run trains at, reported as the spec's
            ``native_size``. A folder has no native resolution of its own —
            every image is resized and cropped to this.
        num_classes: how many class subdirectories the folder is expected to
            have, or None to train unconditionally. Verified at load time.
        hflip: whether a horizontal flip is a label-preserving augmentation.
            True by default, since a folder is usually natural images; set it
            false for anything with a handedness to it, like text or digits.
        holdout: fraction of the images to hold out for scoring, when the
            directory has no explicit ``train/`` and ``test/`` subdirectories.

    Returns:
        A spec whose builder scans the directory it is handed.
    """

    def builder(
        root: str | Path,
        train: bool,
        download: bool,
        transform: Any,
    ) -> Dataset[tuple[torch.Tensor, int]]:
        del download
        return load_folder(
            root,
            train=train,
            channels=channels,
            num_classes=num_classes,
            holdout=holdout,
            transform=transform,
        )

    return DatasetSpec(
        name=FOLDER_DATASET,
        channels=channels,
        native_size=image_size,
        num_classes=num_classes if num_classes is not None else 0,
        hflip=hflip,
        crop=True,
        builder=builder,
    )


def image_transform(
    channels: int, image_size: int = 32, *, hflip: bool = False, crop: bool = False
) -> transforms.Compose:
    """Build the preprocessing pipeline for a dataset.

    Images are resized and scaled to [-1, 1], which is the range the DDPM
    forward process assumes: x_0 has roughly unit scale, so the noised
    x_t = sqrt(abar) * x_0 + sqrt(1 - abar) * eps stays well conditioned.

    Args:
        channels: image channels, which the normalisation is sized to.
        image_size: target resolution. It has to leave the U-Net an exact
            halving per level — 32 is what the shipped configs use, and it
            keeps 28x28 digits intact.
        hflip: prepend a random horizontal flip. Training only, and only for a
            dataset whose spec allows it: the flip draws from the global RNG,
            so applying it while scoring would make a held-out number move for
            reasons that have nothing to do with the weights.
        crop: centre-crop to ``image_size`` after the resize, for a source
            whose images are not already square. ``Resize`` takes a single
            length to mean the *short* side, so without this a 4:3 photo comes
            out 4:3 and the batch cannot be collated. Off for the packaged
            datasets, where it would be a no-op on every image.

    Returns:
        A transform mapping a PIL image to a ``(channels, image_size,
        image_size)`` tensor in [-1, 1].
    """
    steps: list[Callable[[Any], Any]] = [transforms.Resize(image_size)]
    if crop:
        steps.append(transforms.CenterCrop(image_size))
    if hflip:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * channels, (0.5,) * channels),
    ]
    return transforms.Compose(steps)


def image_dataset(
    spec: DatasetSpec,
    root: str | Path = "data/",
    *,
    train: bool = True,
    image_size: int = 32,
    download: bool = True,
    augment: bool = False,
) -> Dataset[tuple[torch.Tensor, int]]:
    """Load a registered dataset from disk, downloading it if needed.

    Args:
        spec: which dataset to load.
        root: directory holding (or receiving) the raw files.
        train: load the training split, else the held-out one.
        image_size: resolution passed to :func:`image_transform`.
        download: fetch the archives when they are missing from ``root``.
        augment: apply the spec's training augmentation. Off by default, so a
            caller has to ask: scoring an augmented split silently measures
            something other than the split.

    Returns:
        A dataset yielding ``(image, label)`` where image is in [-1, 1].
    """
    return spec.builder(
        root=str(root),
        train=train,
        download=download,
        transform=image_transform(
            spec.channels,
            image_size,
            hflip=augment and spec.hflip,
            crop=spec.crop,
        ),
    )


def image_dataloader(
    spec: DatasetSpec,
    root: str | Path = "data/",
    *,
    batch_size: int = 128,
    train: bool = True,
    image_size: int = 32,
    num_workers: int = 4,
    download: bool = True,
    pin_memory: bool | None = None,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
    augment: bool = False,
    generator: torch.Generator | None = None,
    num_replicas: int | None = None,
    rank: int | None = None,
    seed: int = 0,
) -> DataLoader[tuple[torch.Tensor, int]]:
    """Build a dataloader over a registered dataset, ready for a training loop.

    Args:
        spec: which dataset to load.
        root: directory holding (or receiving) the raw files.
        batch_size: samples per batch.
        train: use the training split. Also the default for ``shuffle`` and
            ``drop_last``, which is what a training loop wants.
        image_size: resolution passed to :func:`image_transform`.
        num_workers: worker processes. Set to 0 when debugging, or on Windows
            inside a notebook where process spawning is awkward.
        download: fetch the archives when they are missing from ``root``.
        pin_memory: pin host memory for faster H2D copies. Defaults to whether
            CUDA is available.
        shuffle: shuffle each epoch, or None to follow ``train``.
        drop_last: discard the final partial batch, or None to follow ``train``.
            A ragged last batch perturbs the loss average during training, but
            dropping it while scoring would silently omit images.
        augment: apply the spec's training augmentation. See
            :func:`image_dataset`.
        generator: RNG the shuffle order is drawn from, or None for the global
            one. The sampler draws a fresh permutation from it at the start of
            every epoch, so re-seeding it between epochs is what lets a caller
            make the order a function of the epoch rather than of how many
            epochs have already run.
        num_replicas: how many processes are splitting this dataset between
            them, or None for the ordinary undivided loader. Passing it swaps
            the shuffling sampler for a
            :class:`~torch.utils.data.DistributedSampler`, so each process
            draws a disjoint shard and one pass over the loader is one pass
            over ``1/num_replicas`` of the data.
        rank: this process's index in that split. Required alongside
            `num_replicas` and ignored without it.
        seed: base seed for the distributed shuffle. Combined with the epoch
            that :func:`set_loader_epoch` sets, so sharded batch order is a
            function of ``(seed, epoch)`` exactly as `generator` makes it for
            the undivided loader.

    Returns:
        A configured :class:`~torch.utils.data.DataLoader`.

    Raises:
        ValueError: if `num_replicas` is given without `rank`.
    """
    if num_replicas is not None and rank is None:
        raise ValueError("num_replicas needs a rank to go with it")
    dataset = image_dataset(
        spec,
        root,
        train=train,
        image_size=image_size,
        download=download,
        augment=augment,
    )

    loader_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    wants_shuffle = train if shuffle is None else shuffle
    wants_drop_last = train if drop_last is None else drop_last

    if num_replicas is not None:
        loader_kwargs["sampler"] = DistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=wants_shuffle,
            seed=seed,
            drop_last=wants_drop_last,
        )
        wants_shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=wants_shuffle,
        drop_last=wants_drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        generator=generator,
        **loader_kwargs,
    )


def set_loader_epoch(loader: DataLoader[tuple[torch.Tensor, int]], epoch: int) -> None:
    """Tell a sharded loader which epoch is starting.

    A :class:`~torch.utils.data.DistributedSampler` draws its permutation from
    ``seed + epoch``, and it has no way to know the epoch advanced unless it is
    told: left alone it reshuffles to the *same* order every epoch, and each
    rank sees the identical shard of the data from start to finish. That is a
    quiet failure — the loss still falls, just on a fraction of the dataset.

    Args:
        loader: a loader from :func:`image_dataloader`. One built without
            ``num_replicas`` has no sampler to advance, and is left alone.
        epoch: the epoch about to run.
    """
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Map model-space images in [-1, 1] back to [0, 1] for saving or display.

    Args:
        x: tensor of any shape produced by the sampler.

    Returns:
        The same shape, clamped to [0, 1].
    """
    return (x + 1.0).div(2.0).clamp(0.0, 1.0)
