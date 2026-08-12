"""MNIST dataset loading for diffusion training."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

MNIST_CHANNELS = 1
"""MNIST is greyscale, so the U-Net needs in_channels=out_channels=1."""

MNIST_NATIVE_SIZE = 28
"""Native MNIST resolution. Resized to a power of two so downsampling is exact."""


def mnist_transform(image_size: int = 32) -> transforms.Compose:
    """Build the preprocessing pipeline for MNIST.

    Images are resized and scaled to [-1, 1], which is the range the DDPM
    forward process assumes: x_0 has roughly unit scale, so the noised
    x_t = sqrt(abar) * x_0 + sqrt(1 - abar) * eps stays well conditioned.

    Args:
        image_size: target resolution. 32 keeps the 28x28 digits intact while
            allowing three exact halvings (32 -> 16 -> 8 -> 4).

    Returns:
        A transform mapping a PIL image to a (1, image_size, image_size) tensor.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),  # [0, 255] uint8 -> [0, 1] float
            transforms.Normalize((0.5,), (0.5,)),  # [0, 1] -> [-1, 1]
        ]
    )


def mnist_dataset(
    root: str | Path = "data/",
    *,
    train: bool = True,
    image_size: int = 32,
    download: bool = True,
) -> Dataset[tuple[torch.Tensor, int]]:
    """Load the MNIST dataset from disk, downloading it if needed.

    Args:
        root: directory holding (or receiving) the raw MNIST files.
        train: load the 60k training split, else the 10k test split.
        image_size: resolution passed to :func:`mnist_transform`.
        download: fetch the archives when they are missing from ``root``.

    Returns:
        A dataset yielding ``(image, label)`` where image is in [-1, 1].
    """
    return MNIST(
        root=str(root),
        train=train,
        download=download,
        transform=mnist_transform(image_size),
    )


def mnist_dataloader(
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
) -> DataLoader[tuple[torch.Tensor, int]]:
    """Build a dataloader over MNIST ready for a training loop.

    Args:
        root: directory holding (or receiving) the raw MNIST files.
        batch_size: samples per batch.
        train: use the training split. Also the default for ``shuffle`` and
            ``drop_last``, which is what a training loop wants.
        image_size: resolution passed to :func:`mnist_transform`.
        num_workers: worker processes. Set to 0 when debugging, or on Windows
            inside a notebook where process spawning is awkward.
        download: fetch the archives when they are missing from ``root``.
        pin_memory: pin host memory for faster H2D copies. Defaults to whether
            CUDA is available.
        shuffle: shuffle each epoch, or None to follow ``train``.
        drop_last: discard the final partial batch, or None to follow ``train``.
            A ragged last batch perturbs the loss average during training, but
            dropping it while scoring would silently omit images.

    Returns:
        A configured :class:`~torch.utils.data.DataLoader`.
    """
    dataset = mnist_dataset(root, train=train, image_size=image_size, download=download)

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
