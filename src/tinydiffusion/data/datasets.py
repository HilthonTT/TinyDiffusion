"""What a run can train on: the dataset registry, and loading from it.

Everything downstream of a config — the U-Net's channel count, the shape the
samplers draw, the reference side of a FID — reads what it needs from the
:class:`DatasetSpec` the config names, rather than from a constant. Adding a
dataset is therefore an entry in :data:`DATASETS` and nothing else.

The three registered here share torchvision's ``(root, train, download,
transform)`` constructor, which is what lets one builder serve all of them.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

__all__ = [
    "DATASETS",
    "DEFAULT_DATASET",
    "DatasetSpec",
    "dataset_names",
    "dataset_spec",
    "denormalize",
    "image_dataloader",
    "image_dataset",
    "image_transform",
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
        builder: torchvision dataset class taking ``(root, train, download,
            transform)``.
    """

    name: str
    channels: int
    native_size: int
    num_classes: int
    hflip: bool
    builder: Callable[..., Dataset[tuple[torch.Tensor, int]]]


DATASETS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist",
        channels=1,
        native_size=28,
        num_classes=10,
        # A mirrored 2 is not a 2, and a mirrored 5 is nothing at all.
        hflip=False,
        builder=MNIST,
    ),
    "fashion_mnist": DatasetSpec(
        name="fashion_mnist",
        channels=1,
        native_size=28,
        num_classes=10,
        # Same shape and size as MNIST but far less separable, which makes it
        # the cheapest way to see whether a change helps or only helps on MNIST.
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
    """Every registered dataset name, sorted.

    Returns:
        The keys of :data:`DATASETS`, for error messages and CLI choices.
    """
    return tuple(sorted(DATASETS))


def dataset_spec(name: str) -> DatasetSpec:
    """Look up a dataset by name.

    Args:
        name: a key of :data:`DATASETS`.

    Returns:
        The matching spec.

    Raises:
        ValueError: if nothing is registered under `name`.
    """
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(
            f"unknown dataset {name!r}, expected one of: {', '.join(dataset_names())}"
        ) from None


def image_transform(
    channels: int, image_size: int = 32, *, hflip: bool = False
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

    Returns:
        A transform mapping a PIL image to a ``(channels, image_size,
        image_size)`` tensor in [-1, 1].
    """
    steps: list[Callable[[Any], Any]] = [transforms.Resize(image_size)]
    if hflip:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [
        transforms.ToTensor(),  # [0, 255] uint8 -> [0, 1] float
        transforms.Normalize((0.5,) * channels, (0.5,) * channels),  # [0, 1] -> [-1, 1]
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
        transform=image_transform(spec.channels, image_size, hflip=augment and spec.hflip),
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

    Returns:
        A configured :class:`~torch.utils.data.DataLoader`.
    """
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
        # Re-spawning workers every epoch dominates runtime on a dataset this small.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train if shuffle is None else shuffle,
        drop_last=train if drop_last is None else drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        generator=generator,
        **loader_kwargs,
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Map model-space images in [-1, 1] back to [0, 1] for saving or display.

    Args:
        x: tensor of any shape produced by the sampler.

    Returns:
        The same shape, clamped to [0, 1].
    """
    return (x + 1.0).div(2.0).clamp(0.0, 1.0)
