"""Frechet Inception Distance, and the streaming statistics it is computed from."""

from collections.abc import Mapping
from typing import Any

import torch


def _matrix_sqrt_psd(matrix: torch.Tensor) -> torch.Tensor:
    """Symmetric square root of a positive semi-definite matrix.

    Args:
        matrix: symmetric ``(d, d)`` matrix.

    Returns:
        The symmetric PSD matrix ``s`` with ``s @ s == matrix``.
    """
    matrix = 0.5 * (matrix + matrix.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    roots = eigenvalues.clamp_min(0).sqrt()
    return (eigenvectors * roots) @ eigenvectors.mT


def compute_fid(
    mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor
) -> torch.Tensor:
    """Compute the Frechet distance between two multivariate Gaussians.

    The Frechet Inception Distance between ``X_x ~ N(mu_1, sigma_1)`` and
    ``X_y ~ N(mu_2, sigma_2)`` is
    ``d^2 = ||mu_1 - mu_2||^2 + Tr(sigma_1 + sigma_2 - 2*sqrt(sigma_1*sigma_2))``.

    The cross term is evaluated as ``Tr(sqrt(A))`` with
    ``A = sqrt(sigma_1) @ sigma_2 @ sqrt(sigma_1)``, which is symmetric PSD and
    has the same eigenvalues as ``sigma_1 @ sigma_2``. Taking the eigenvalues of
    the raw product instead is cheaper but that matrix is not symmetric, so a
    general eigensolver hands back complex values whose real parts silently
    understate the trace.

    Args:
        mu1: mean of activations calculated on predicted (x) samples, ``(d,)``.
        sigma1: covariance matrix over activations calculated on predicted (x)
            samples, ``(d, d)``.
        mu2: mean of activations calculated on target (y) samples, ``(d,)``.
        sigma2: covariance matrix over activations calculated on target (y)
            samples, ``(d, d)``.

    Returns:
        Scalar tensor holding the distance between the two sets, in float64.

    Raises:
        ValueError: if the means are not 1-D, the covariances are not square, or
            the four arguments disagree on the feature dimension.
    """
    if mu1.ndim != 1 or mu2.ndim != 1:
        raise ValueError(f"means must be 1-D, got shapes {tuple(mu1.shape)} and {tuple(mu2.shape)}")
    dim = mu1.shape[0]
    for name, cov in (("sigma1", sigma1), ("sigma2", sigma2)):
        if cov.shape != (dim, dim):
            raise ValueError(f"{name} must be ({dim}, {dim}), got {tuple(cov.shape)}")
    if mu2.shape[0] != dim:
        raise ValueError(f"means disagree on dimension: {dim} and {mu2.shape[0]}")

    mu1, mu2 = mu1.double(), mu2.double()
    sigma1, sigma2 = sigma1.double(), sigma2.double()

    a = (mu1 - mu2).square().sum()
    b = sigma1.trace() + sigma2.trace()
    sqrt_sigma1 = _matrix_sqrt_psd(sigma1)
    inner = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    inner = 0.5 * (inner + inner.mT)
    c = torch.linalg.eigvalsh(inner).clamp_min(0).sqrt().sum()

    return ((a + b) - 2 * c).clamp_min(0)


class FeatureStats:
    """Streaming mean and covariance of a feature set.

    Accumulates the sum and the uncentred second moment of every feature vector
    it is shown, so a FID can be taken over a dataset far larger than memory
    without ever holding more than one batch at a time.

    Attributes:
        dim: feature dimension.
        n: how many feature vectors have been seen.
    """

    def __init__(self, dim: int, device: torch.device | str = "cpu") -> None:
        """Create empty statistics.

        Args:
            dim: feature dimension of the vectors that will be added.
            device: device to accumulate on. Keeping this next to the feature
                extractor avoids a sync per batch.

        Raises:
            ValueError: if ``dim`` is not positive.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.n = 0
        self.sum = torch.zeros(dim, dtype=torch.float64, device=device)
        self.outer = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    @property
    def device(self) -> torch.device:
        """Device the statistics are accumulated on."""
        return self.sum.device

    def __len__(self) -> int:
        """Number of feature vectors seen so far."""
        return self.n

    @torch.no_grad()
    def update(self, feats: torch.Tensor) -> None:
        """Fold a batch of features into the running statistics.

        Args:
            feats: ``(batch, dim)`` activations. Any dtype and device; they are
                cast to float64 on this object's device.

        Raises:
            ValueError: if ``feats`` is not 2-D or its width is not ``dim``.
        """
        if feats.ndim != 2:
            raise ValueError(f"expected (batch, dim) features, got shape {tuple(feats.shape)}")
        if feats.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim} features per row, got {feats.shape[1]}")
        if feats.shape[0] == 0:
            return

        f = feats.detach().to(device=self.device, dtype=torch.float64)
        self.n += f.shape[0]
        self.sum += f.sum(0)
        self.outer += f.mT @ f

    def merge(self, other: FeatureStats) -> None:
        """Fold another set of statistics into this one, in place.

        Lets several workers accumulate independently and combine at the end.

        Args:
            other: statistics over the same feature dimension.

        Raises:
            ValueError: if the two disagree on the feature dimension.
        """
        if other.dim != self.dim:
            raise ValueError(f"cannot merge {other.dim}-dim stats into {self.dim}-dim stats")
        self.n += other.n
        self.sum += other.sum.to(self.device)
        self.outer += other.outer.to(self.device)

    def state_dict(self) -> dict[str, torch.Tensor | int]:
        """The accumulator's contents, in a form that survives a round trip to disk.

        The raw sums rather than the mean and covariance they derive: they are
        what :meth:`merge` adds, so a restored set can still be extended, and
        the covariance stays a function of the accumulated moments rather than
        of a value that was rounded on the way out.

        Returns:
            A mapping of plain tensors and ints, ready for :func:`torch.save`
            under ``weights_only``.
        """
        return {"dim": self.dim, "n": self.n, "sum": self.sum.cpu(), "outer": self.outer.cpu()}

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> FeatureStats:
        """Rebuild statistics from a :meth:`state_dict`.

        Args:
            state: a mapping produced by :meth:`state_dict`.
            device: device to accumulate on from here on.

        Returns:
            The restored statistics.

        Raises:
            ValueError: if a key is missing, or the tensors do not have the
                shapes ``dim`` calls for. Both mean the payload is not one this
                class wrote, so it is rejected rather than half-loaded.
        """
        try:
            dim, n = int(state["dim"]), int(state["n"])
            total, outer = state["sum"], state["outer"]
        except KeyError as exc:
            raise ValueError(f"feature statistics are missing {exc} key") from None
        stats = cls(dim, device=device)
        if total.shape != (dim,) or outer.shape != (dim, dim):
            raise ValueError(
                f"feature statistics for dim={dim} must hold ({dim},) and ({dim}, {dim}) "
                f"tensors, got {tuple(total.shape)} and {tuple(outer.shape)}"
            )
        if n < 0:
            raise ValueError(f"feature count must not be negative, got {n}")
        stats.n = n
        stats.sum = total.to(device=stats.device, dtype=torch.float64)
        stats.outer = outer.to(device=stats.device, dtype=torch.float64)
        return stats

    def reset(self) -> None:
        """Drop everything seen so far, keeping the dimension and device."""
        self.n = 0
        self.sum.zero_()
        self.outer.zero_()

    def mean_cov(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and unbiased covariance of everything seen so far.

        Returns:
            ``(mu, cov)`` of shapes ``(dim,)`` and ``(dim, dim)``, in float64.

        Raises:
            ValueError: if fewer than two feature vectors have been added, which
                leaves the covariance undefined.
        """
        if self.n < 2:
            raise ValueError(f"need at least 2 samples for a covariance, got {self.n}")
        mu = self.sum / self.n
        cov = (self.outer - self.n * torch.outer(mu, mu)) / (self.n - 1)
        cov = 0.5 * (cov + cov.mT)
        return mu, cov


def fid_from_stats(fake: FeatureStats, real: FeatureStats) -> float:
    """Score two accumulated feature sets against each other.

    Args:
        fake: statistics over generated samples.
        real: statistics over reference samples.

    Returns:
        The FID, as a plain float.

    Raises:
        ValueError: if either set is too small, or the two disagree on the
            feature dimension.
    """
    if fake.dim != real.dim:
        raise ValueError(f"feature dimensions differ: {fake.dim} and {real.dim}")
    mu1, sigma1 = fake.mean_cov()
    mu2, sigma2 = real.mean_cov()
    return float(compute_fid(mu1, sigma1, mu2.to(mu1.device), sigma2.to(sigma1.device)))
