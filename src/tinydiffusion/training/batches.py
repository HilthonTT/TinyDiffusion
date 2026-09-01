"""The batches a run fixes once and reuses every epoch.

Neither read here is taken off the training loop: the held-out slice has to be
the same images every epoch for ``best.pt`` to mean anything, and the real
strip has to be the same images for the sample grids to be a flipbook rather
than a slideshow. :func:`epoch_seed` is the other half of that determinism,
making the shuffle order a function of the epoch index alone.
"""

import torch

from tinydiffusion.data.datasets import image_dataloader
from tinydiffusion.training.config import TrainConfig

__all__ = [
    "epoch_seed",
    "reference_batch",
    "validation_batches",
]


def epoch_seed(seed: int, epoch: int) -> int:
    """Seed for one epoch's shuffle order.

    A function of ``(seed, epoch)`` alone, deliberately: seeding the loader
    once at startup makes the order depend on how many epochs have already run
    in *this process*, so a run resumed at epoch 5 replays the ordering a fresh
    run used for epoch 0, and every later epoch follows suit. Deriving it here
    means epoch 5 draws epoch 5's batches whether it was reached by resuming or
    by training straight through.

    Args:
        seed: the run's seed.
        epoch: zero-based epoch index.

    Returns:
        A seed inside the range ``torch.Generator.manual_seed`` accepts.
    """
    return (seed * 1_000_003 + epoch) & ((1 << 63) - 1)


def validation_batches(cfg: TrainConfig) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialise the held-out slice the run is scored on each epoch.

    Read once and kept in host memory rather than reloaded per epoch: the slice
    is small, and it has to be *the same images every time* for the epoch-to-
    epoch comparison — and so for ``best.pt`` — to mean anything.

    Args:
        cfg: run configuration. ``val_batches`` bounds the slice; 0 takes the
            whole test split.

    Returns:
        ``(images, labels)`` pairs on the CPU, empty if ``val_every`` is off.
    """
    if cfg.val_every <= 0:
        return []

    loader = image_dataloader(
        cfg.dataset_spec(),
        cfg.data_root,
        batch_size=cfg.batch_size,
        train=False,
        image_size=cfg.image_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, (x, y) in enumerate(loader):
        if cfg.val_batches and index >= cfg.val_batches:
            break
        batches.append((x, y))
    return batches


def reference_batch(cfg: TrainConfig) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Materialise the real strip every sample grid is compared against.

    Read from the front of the *unshuffled* training split rather than taken
    from whichever batch the loop happens to see first. The shuffle order is a
    function of the epoch index, so a batch picked off the loop would differ
    between a straight run and a ``--resume`` — and for a conditional run those
    labels are what the generated half is drawn from, so the grids would stop
    being a flipbook of one set of images at exactly the point the run was
    picked up again. This depends on the dataset alone, which is the same
    property :func:`~tinydiffusion.training.train.train`'s fixed ``x_T`` has.

    Unaugmented, for the same reason it is unshuffled: a flip that lands
    differently per read would move the real strip on its own.

    Args:
        cfg: run configuration. ``num_samples`` bounds the strip, and
            ``sample_every`` of 0 skips the read entirely.

    Returns:
        ``(images, labels)`` on the CPU. Labels are None for an unconditional
        run, and both are None when no grid will ever be drawn.
    """
    if cfg.sample_every <= 0:
        return None, None

    loader = image_dataloader(
        cfg.dataset_spec(),
        cfg.data_root,
        batch_size=cfg.num_samples,
        train=True,
        image_size=cfg.image_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )
    for x, y in loader:
        return x[: cfg.num_samples], y[: cfg.num_samples] if cfg.num_classes is not None else None
    return None, None
