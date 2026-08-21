"""Kernel Inception Distance: the score to read when there are not many samples.

FID fits a Gaussian to each feature set and measures between the fits. The fit
is the problem. A covariance estimated from fewer samples than it has dimensions
is singular, and the error that introduces does not average out — it is a bias,
always upwards, and its size depends on the sample count. So a FID over 1,000
images is not a worse estimate of the same quantity as a FID over 50,000; it is
an estimate of a different one, and the two cannot be compared. Inception's
2048 features mean "not many" starts below about 10,000 images, which is most
of what a single-GPU run can afford between checkpoints.

KID (Binkowski et al., 2018) drops the Gaussian. It is the squared maximum mean
discrepancy between the two sets under a polynomial kernel, in the unbiased
form — the one that leaves the diagonal out of both within-set sums. Unbiased
means what it says: its expected value does not move with the sample count, so
a KID over 1,000 images and a KID over 50,000 estimate the same number and can
be put side by side. It also comes with an error bar, which FID cannot offer at
all, by being averaged over random subsets.

The cost is that it needs the feature vectors rather than two moments, and so
memory linear in the image count — see :class:`FeatureBank`. The numbers are
small (around 1e-3 at MNIST scale) and, like FID, carry no absolute meaning.
"""

from dataclasses import dataclass

import torch

from tinydiffusion.metrics.features import FeatureBank

__all__ = [
    "DEFAULT_KID_SUBSETS",
    "DEFAULT_KID_SUBSET_SIZE",
    "KidResult",
    "compute_kid",
    "kid_from_features",
]

DEFAULT_KID_SUBSET_SIZE = 1000
"""Images per subset. The published default, and the size the error bar refers to."""

DEFAULT_KID_SUBSETS = 100
"""Subsets averaged over. Enough that the reported mean is stable to its own last digit."""

_CHUNK = 4096
"""Rows of the kernel matrix built at once, so a large subset is not a large allocation."""


@dataclass(slots=True)
class KidResult:
    """A KID estimate and the spread of the subsets it was averaged over.

    Attributes:
        mean: the KID. Lower is better; zero is the two sets being
            indistinguishable to the kernel, and small negative values are
            ordinary — the estimator is unbiased, not non-negative.
        std: standard deviation across subsets. Not a confidence interval on
            the mean; it is how much one subset of `subset_size` images
            disagrees with another, which is what says whether two checkpoints'
            scores are far enough apart to mean anything.
        subsets: how many subsets were drawn.
        subset_size: images per subset, per side.
    """

    mean: float
    std: float
    subsets: int
    subset_size: int


def _polynomial_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The cubic polynomial kernel KID is conventionally measured under.

    ``k(x, y) = ((x . y) / d + 1)^3``, where ``d`` is the feature dimension.
    Dividing by ``d`` is what keeps the cubed term in range whatever the
    feature width, and is part of the published definition rather than a
    normalisation choice this project made.

    Args:
        x: ``(m, d)`` features.
        y: ``(n, d)`` features.

    Returns:
        The ``(m, n)`` kernel matrix.
    """
    return (x @ y.mT / x.shape[1] + 1.0).pow(3)


def _mean_offdiagonal(x: torch.Tensor) -> torch.Tensor:
    """Mean of a set's kernel matrix with its diagonal excluded.

    Leaving out ``k(x_i, x_i)`` is the whole of what makes the estimator
    unbiased: the diagonal is the one place a sample is compared with itself,
    and counting it inflates every within-set term by an amount that shrinks
    with the sample count — which is exactly the sample-count dependence KID
    exists to remove.

    Args:
        x: ``(m, d)`` features, ``m >= 2``.

    Returns:
        Scalar tensor, the mean over the ``m(m-1)`` off-diagonal entries.
    """
    m = x.shape[0]
    total = torch.zeros((), dtype=torch.float64, device=x.device)
    for start in range(0, m, _CHUNK):
        block = _polynomial_kernel(x[start : start + _CHUNK], x).double()
        # Subtract this block's share of the diagonal rather than materialising
        # the mask: rows [start, start+len) carry diagonal entries at the same
        # offset into the columns.
        rows = torch.arange(block.shape[0], device=x.device)
        total += block.sum() - block[rows, rows + start].sum()
    return total / (m * (m - 1))


def _mean_cross(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean of the kernel between two sets.

    Args:
        x: ``(m, d)`` features.
        y: ``(n, d)`` features.

    Returns:
        Scalar tensor, the mean over all ``m*n`` entries.
    """
    total = torch.zeros((), dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], _CHUNK):
        total += _polynomial_kernel(x[start : start + _CHUNK], y).double().sum()
    return total / (x.shape[0] * y.shape[0])


def _mmd2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Unbiased squared MMD between two feature sets under the polynomial kernel.

    Args:
        x: ``(m, d)`` features from one distribution, ``m >= 2``.
        y: ``(n, d)`` features from the other, ``n >= 2``.

    Returns:
        Scalar tensor in float64. May be slightly negative when the two sets
        come from the same distribution, which is what unbiasedness costs.
    """
    return _mean_offdiagonal(x) + _mean_offdiagonal(y) - 2 * _mean_cross(x, y)


@torch.no_grad()
def compute_kid(
    fake: torch.Tensor,
    real: torch.Tensor,
    *,
    subsets: int = DEFAULT_KID_SUBSETS,
    subset_size: int = DEFAULT_KID_SUBSET_SIZE,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> KidResult:
    """Score two feature sets against each other with KID.

    The estimate is averaged over `subsets` random subsets of `subset_size`
    images from each side, which is how the published implementation does it
    and what the reported standard deviation is a spread over. Sampling is
    without replacement within a subset and independent between them.

    Args:
        fake: ``(m, d)`` features of the generated images.
        real: ``(n, d)`` features of the real ones. Need not match `m`.
        subsets: how many subsets to average over.
        subset_size: images per subset, per side. Clamped down to the smaller
            of the two sets, since a subset cannot be larger than what it is
            drawn from.
        generator: RNG for the subset draws, for a reproducible score. None
            uses the global RNG.
        device: where to compute. None uses the features' own device; passing
            a GPU here is what makes a large `subset_size` affordable, and the
            subsets are moved a batch at a time rather than the whole set.

    Returns:
        The estimate and its spread.

    Raises:
        ValueError: if either set is not 2-D, they disagree on the feature
            dimension, `subsets` or `subset_size` is below 1, or a side holds
            fewer than two vectors, which leaves the unbiased estimator
            undefined.
    """
    for name, feats in (("fake", fake), ("real", real)):
        if feats.ndim != 2:
            raise ValueError(f"{name} features must be (n, dim), got {tuple(feats.shape)}")
        if feats.shape[0] < 2:
            raise ValueError(
                f"need at least 2 {name} features for an unbiased estimate, got {feats.shape[0]}"
            )
    if fake.shape[1] != real.shape[1]:
        raise ValueError(f"feature dimensions differ: {fake.shape[1]} and {real.shape[1]}")
    if subsets < 1:
        raise ValueError(f"subsets must be at least 1, got {subsets}")
    if subset_size < 2:
        raise ValueError(f"subset_size must be at least 2, got {subset_size}")

    size = min(subset_size, fake.shape[0], real.shape[0])
    target = torch.device(device) if device is not None else fake.device
    # float32 on the compute device: the kernel is a cube of an inner product,
    # so the sums are accumulated in float64 but the matmuls themselves are
    # what the feature vectors were stored in.
    fake = fake.to(device=target, dtype=torch.float32)
    real = real.to(device=target, dtype=torch.float32)

    # Drawn on the CPU whatever the compute device: torch.randperm takes a
    # generator only on the device it was created for, and a caller seeding one
    # for reproducibility has no reason to know where the kernel will be built.
    scores = [
        _mmd2(
            fake[torch.randperm(fake.shape[0], generator=generator)[:size].to(target)],
            real[torch.randperm(real.shape[0], generator=generator)[:size].to(target)],
        )
        for _ in range(subsets)
    ]
    stacked = torch.stack(scores)
    # Population rather than sample standard deviation for a single subset, so
    # one subset reports a spread of zero rather than a NaN.
    spread = stacked.std(correction=1 if stacked.numel() > 1 else 0)
    return KidResult(
        mean=float(stacked.mean()),
        std=float(spread),
        subsets=subsets,
        subset_size=size,
    )


def kid_from_features(
    fake: FeatureBank,
    real: FeatureBank,
    *,
    subsets: int = DEFAULT_KID_SUBSETS,
    subset_size: int = DEFAULT_KID_SUBSET_SIZE,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> KidResult:
    """Score two retained feature sets against each other with KID.

    Args:
        fake: retained features of the generated images.
        real: retained features of the real ones.
        subsets: how many subsets to average over.
        subset_size: images per subset, per side.
        generator: RNG for the subset draws.
        device: where to compute. See :func:`compute_kid`.

    Returns:
        The estimate and its spread.

    Raises:
        ValueError: if the two banks disagree on the feature dimension, or
            :func:`compute_kid` rejects their contents.
    """
    if fake.dim != real.dim:
        raise ValueError(f"feature dimensions differ: {fake.dim} and {real.dim}")
    return compute_kid(
        fake.features,
        real.features,
        subsets=subsets,
        subset_size=subset_size,
        generator=generator,
        device=device,
    )
