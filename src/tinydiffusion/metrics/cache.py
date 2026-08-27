"""On-disk cache for the reference half of a FID score.

The generated half of a FID changes with every checkpoint and every sampling
setting, so it has to be drawn each time. The real half does not: for a given
dataset, split, resolution and image count it is the same set of images through
the same feature network, and running it again produces the same statistics to
the last bit. Sweeping `--guidance` over five values therefore pushes 50,000
real images through Inception-v3 to compute one number five times.

This module keeps that number. The payload is the accumulator's raw moments —
see :meth:`~tinydiffusion.metrics.fid.FeatureStats.state_dict` — so a restored
set is indistinguishable from a freshly computed one, including for
:meth:`~tinydiffusion.metrics.fid.FeatureStats.merge`.

Every input that moves the statistics is in the filename, which is what makes a
stale entry impossible rather than merely unlikely: change the resolution or
the extractor and the key changes with it. A file that cannot be read, or that
does not describe what its name claims, is treated as absent and recomputed.

An entry is not small: the second moment is ``dim x dim`` in float64, so
Inception's 2048 features come to about 33 MB however many images went into it.
That is the price of keeping the accumulation in the precision
:class:`~tinydiffusion.metrics.fid.FeatureStats` needs — float32 loses most of
the covariance's significant digits by a few tens of thousands of images — and
it buys back an Inception pass over the whole reference split. The entries are
plain files under ``<data_root>/fid_cache``; deleting them costs only the next
score's real pass, and ``--no-cache`` skips them without deleting anything.

KID and precision/recall need the feature vectors themselves rather than their
moments, so they cache a second kind of entry — a
:class:`~tinydiffusion.metrics.features.FeatureBank`, under the same key with a
``_features`` suffix. That one *does* grow with the image count: about 8 KB an
image, so 80 MB for the usual 10,000 against the moments' flat 33 MB. It is
written only when a score actually asked for those metrics, so a run wanting
FID alone keeps paying the smaller price. Both kinds are derived data and safe
to delete.
"""

import os
from pathlib import Path
from typing import Any

import torch

from tinydiffusion.metrics.features import FeatureBank
from tinydiffusion.metrics.fid import FeatureStats
from tinydiffusion.metrics.inception import FeatureExtractor

__all__ = [
    "CACHE_DIRNAME",
    "extractor_id",
    "load_reference_features",
    "load_reference_stats",
    "reference_features_path",
    "reference_stats_path",
    "save_reference_features",
    "save_reference_stats",
    "spatial_stats_path",
]

CACHE_DIRNAME = "fid_cache"
"""Subdirectory of the dataset root the cached statistics live in."""

_FORMAT = 1
"""Payload version. A file written by a different one is recomputed, not read."""


def extractor_id(extractor: FeatureExtractor) -> str:
    """A short name for the feature network a cached entry was built with.

    The class name rather than its weights: a cached entry is only ever reused
    by a process that constructed the same extractor the same way, and the name
    is what separates a real Inception-v3 from the stand-ins that keep the FID
    plumbing testable without downloading 100 MB of weights.

    Args:
        extractor: the feature network.

    Returns:
        A filename-safe identifier, e.g. ``"inceptionfeatures2048"``.
    """
    name = "".join(ch for ch in type(extractor).__name__ if ch.isalnum()).lower()
    return f"{name}{extractor.dim}"


def reference_stats_path(
    root: Path,
    *,
    dataset: str,
    split: str,
    num_images: int,
    image_size: int,
    extractor: FeatureExtractor,
) -> Path:
    """Where the statistics for one reference set are cached.

    Args:
        root: the dataset directory. The cache sits beside the raw files it
            describes, so deleting a dataset takes its statistics with it.
        dataset: registered dataset name.
        split: ``"train"`` or ``"test"``.
        num_images: how many images were *asked* for. The realised count can be
            smaller when the split runs out, and is stored in the payload; the
            request is what identifies the set, since the same request always
            reads the same images.
        image_size: the resolution they were resized to.
        extractor: the feature network they were run through.

    Returns:
        The path the entry would live at, whether or not it exists.
    """
    name = f"{dataset}_{split}_{num_images}_{image_size}px_{extractor_id(extractor)}.pt"
    return root / CACHE_DIRNAME / name


def reference_features_path(
    root: Path,
    *,
    dataset: str,
    split: str,
    num_images: int,
    image_size: int,
    extractor: FeatureExtractor,
) -> Path:
    """Where the retained features for one reference set are cached.

    The same key as :func:`reference_stats_path` — the two describe the same
    images through the same network — plus a suffix, so the two entries sit
    beside each other instead of one overwriting the other.

    Args:
        root: the dataset directory.
        dataset: registered dataset name.
        split: ``"train"`` or ``"test"``.
        num_images: how many images were asked for.
        image_size: the resolution they were resized to.
        extractor: the feature network they were run through.

    Returns:
        The path the entry would live at, whether or not it exists.
    """
    stats = reference_stats_path(
        root,
        dataset=dataset,
        split=split,
        num_images=num_images,
        image_size=image_size,
        extractor=extractor,
    )
    return stats.with_name(f"{stats.stem}_features{stats.suffix}")


def spatial_stats_path(
    root: Path,
    *,
    dataset: str,
    split: str,
    num_images: int,
    image_size: int,
    extractor: FeatureExtractor,
) -> Path:
    """Where the *spatial* statistics for one reference set are cached.

    sFID is a FID taken in a different reading of the same network, so its
    reference half is cached the same way and under the same key, plus a
    suffix. The payload's own feature width is what actually distinguishes it —
    :func:`load_reference_stats` rejects an entry whose ``dim`` disagrees with
    the one asked for — so a suffix collision could not produce a wrong score,
    only a wasted read.

    Args:
        root: the dataset directory.
        dataset: registered dataset name.
        split: ``"train"`` or ``"test"``.
        num_images: how many images were asked for.
        image_size: the resolution they were resized to.
        extractor: the feature network they were run through.

    Returns:
        The path the entry would live at, whether or not it exists.
    """
    stats = reference_stats_path(
        root,
        dataset=dataset,
        split=split,
        num_images=num_images,
        image_size=image_size,
        extractor=extractor,
    )
    return stats.with_name(f"{stats.stem}_spatial{stats.suffix}")


def _load_payload(path: Path) -> dict[str, Any] | None:
    """Read one cache entry, or report there is nothing usable at `path`.

    A cache is an optimisation, so every way of failing to read one returns
    None and lets the caller recompute: a missing file, a truncated or
    half-written one, and a payload from a format this version does not write.

    Args:
        path: the file to read.

    Returns:
        The payload, or None.
    """
    if not path.is_file():
        return None
    try:
        payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Deliberately everything. A partial write, a file from another torch,
        # or something that is not a checkpoint at all each come back as a
        # different type — torch.load documents no exception contract for
        # malformed input, so enumerating them is guesswork that goes stale.
        # None of them is worth failing a scoring run over, and the only cost
        # of being wrong is recomputing what was already going to be computed
        # before this cache existed. BaseException still propagates, so a
        # Ctrl+C during the read is not swallowed with it.
        return None
    if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
        return None
    return payload


def load_reference_stats(
    path: Path, *, dim: int, device: torch.device | str = "cpu"
) -> FeatureStats | None:
    """Read cached reference statistics, or report that there are none usable.

    A cache is an optimisation, so every way of failing to read one returns
    None and lets the caller recompute: a missing file, a truncated or
    half-written one, a payload from an older format, and a payload whose
    shapes do not match the dimension asked for.

    Args:
        path: the file to read, from :func:`reference_stats_path`.
        dim: the feature dimension the caller is about to score in. A payload
            that disagrees is not this extractor's, whatever the filename says.
        device: device to accumulate on from here on.

    Returns:
        The restored statistics, or None if nothing usable is on disk.
    """
    payload = _load_payload(path)
    if payload is None:
        return None
    try:
        stats = FeatureStats.from_state_dict(payload, device=device)
    except ValueError:
        return None
    if stats.dim != dim or stats.n < 2:
        # Fewer than two vectors leaves the covariance undefined, so an entry
        # that small is no more useful than no entry at all.
        return None
    return stats


def _save_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write one cache entry, or leave whatever was already there.

    Written to a temporary name in the same directory and moved into place, so
    a run cancelled mid-write leaves either the previous entry or none — never
    a truncated file that the next run has to detect. The temporary name
    carries the process id, so two runs scoring the same reference set at once
    do not write over each other's partial file.

    Args:
        path: the file to write.
        payload: what to save, already stamped with its format.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    except OSError:
        # A read-only or full dataset directory is not a reason to fail a score
        # that has already been computed; the next run simply recomputes.
        tmp.unlink(missing_ok=True)
    except BaseException:
        # Anything else is a bug rather than a full disk, so it propagates —
        # but not while leaving a half-written temporary behind. Ctrl+C during
        # the write lands here too, which is exactly when the cleanup matters.
        tmp.unlink(missing_ok=True)
        raise


def save_reference_stats(path: Path, stats: FeatureStats) -> None:
    """Write reference statistics for a later run to pick up.

    Args:
        path: the file to write, from :func:`reference_stats_path`.
        stats: the accumulated reference statistics.
    """
    _save_atomically(path, {"format": _FORMAT, **stats.state_dict()})


def load_reference_features(
    path: Path, *, dim: int, device: torch.device | str = "cpu"
) -> FeatureBank | None:
    """Read a cached reference feature bank, or report there is none usable.

    Rejects an entry for the same reasons :func:`load_reference_stats` does.

    Args:
        path: the file to read, from :func:`reference_features_path`.
        dim: the feature dimension the caller is about to score in. A payload
            that disagrees is not this extractor's, whatever the filename says.
        device: where to keep the restored vectors.

    Returns:
        The restored bank, or None if nothing usable is on disk.
    """
    payload = _load_payload(path)
    if payload is None:
        return None
    try:
        bank = FeatureBank.from_state_dict(payload, device=device)
    except ValueError:
        return None
    if bank.dim != dim or bank.n < 2:
        # Fewer than two vectors is below what every metric here needs, so an
        # entry that small is no more useful than no entry at all.
        return None
    return bank


def save_reference_features(path: Path, bank: FeatureBank) -> None:
    """Write a reference feature bank for a later run to pick up.

    Args:
        path: the file to write, from :func:`reference_features_path`.
        bank: the retained reference features.
    """
    _save_atomically(path, {"format": _FORMAT, **bank.state_dict()})
