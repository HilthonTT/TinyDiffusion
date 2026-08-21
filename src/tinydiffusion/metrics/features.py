"""Retained feature vectors, for the metrics a mean and covariance cannot give.

FID reads a feature set through a Gaussian: two moments, accumulated in a fixed
amount of memory however many images go by (:class:`FeatureStats`). That is the
whole reason it scales, and also the reason it is biased — the Gaussian is an
assumption about a distribution nobody has checked, and the bias it carries
depends on how many samples were used, which is why two FIDs taken at different
image counts cannot be compared.

The metrics that fix that read the *pairwise* structure of the two sets — every
kernel evaluation for KID, every nearest-neighbour radius for precision and
recall — so they need the vectors themselves. That costs memory linear in the
image count rather than constant: at Inception's 2048 features, about 8 KB an
image, or 80 MB for the usual 10,000. This class is that trade, made explicit.

A bank can still produce a FID, from :attr:`FeatureBank.stats`: the moments are
a function of the vectors it is already holding. They are formed on demand
rather than alongside, so a score that never asks for a FID never pays the
``n x d^2`` accumulation, and neither does a bank restored from cache.
"""

from collections.abc import Mapping
from typing import Any

import torch

from tinydiffusion.metrics.fid import FeatureStats

__all__ = ["FeatureBank"]

_MOMENT_CHUNK = 4096
"""Rows folded into the moments at a time. Bounds the float64 cast, nothing more."""


class FeatureBank:
    """Every feature vector shown to it, and the moments they imply.

    Attributes:
        dim: feature dimension.
    """

    def __init__(self, dim: int, device: torch.device | str = "cpu") -> None:
        """Create an empty bank.

        Args:
            dim: feature dimension of the vectors that will be added.
            device: where to keep the retained vectors. The default is the
                host: the bank is the part that grows with the image count, and
                a scoring run would rather spend its device memory on the model
                drawing the samples. The metrics move it back a chunk at a
                time.

        Raises:
            ValueError: if `dim` is not positive.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self._device = torch.device(device)
        self._chunks: list[torch.Tensor] = []
        self._n = 0
        self._stats: FeatureStats | None = None

    @property
    def device(self) -> torch.device:
        """Device the retained vectors are kept on."""
        return self._device

    @property
    def n(self) -> int:
        """How many feature vectors have been added.

        Named to match :class:`FeatureStats`, so either can be handed to
        :func:`~tinydiffusion.metrics.evaluate.accumulate_features`.
        """
        return self._n

    def __len__(self) -> int:
        """Number of feature vectors seen so far."""
        return self._n

    @torch.no_grad()
    def update(self, feats: torch.Tensor) -> None:
        """Retain a batch of features.

        Args:
            feats: ``(batch, dim)`` activations, any dtype and device. Stored
                as float32, which is what the extractor produced; the moments
                :attr:`stats` derives from them are still taken in float64.

        Raises:
            ValueError: if `feats` is not 2-D or its width is not `dim`.
        """
        if feats.ndim != 2:
            raise ValueError(f"expected (batch, dim) features, got shape {tuple(feats.shape)}")
        if feats.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim} features per row, got {feats.shape[1]}")
        if feats.shape[0] == 0:
            return

        self._stats = None
        self._chunks.append(feats.detach().to(device=self._device, dtype=torch.float32))
        self._n += feats.shape[0]

    @property
    def features(self) -> torch.Tensor:
        """Every retained vector, as one ``(n, dim)`` float32 tensor.

        The batches are concatenated on first read and kept that way, so
        repeated reads — one per metric — do not each pay for the copy.
        """
        if len(self._chunks) != 1 or self._chunks[0].shape[0] != self._n:
            stacked = (
                torch.cat(self._chunks)
                if self._chunks
                else torch.zeros(0, self.dim, dtype=torch.float32, device=self._device)
            )
            self._chunks = [stacked]
        return self._chunks[0]

    @property
    def stats(self) -> FeatureStats:
        """Mean and covariance over the retained vectors, for FID.

        Formed on first read and kept, so the several metrics a single scoring
        run asks for do not each rebuild them. Accumulated in chunks rather
        than in one matmul: the second moment is ``dim x dim`` in float64, and
        the intermediate cast of the whole bank would be several times the
        bank itself.

        Returns:
            The moments, as :class:`FeatureStats`. They are what the streaming
            path would have produced up to float64 rounding — the bank sums
            over its own chunks rather than over the batches the images
            happened to arrive in, and addition in that order is not
            associative to the last bit. It is nowhere near a digit any score
            prints.
        """
        if self._stats is None:
            stats = FeatureStats(self.dim, device=self._device)
            features = self.features
            for start in range(0, self._n, _MOMENT_CHUNK):
                stats.update(features[start : start + _MOMENT_CHUNK])
            self._stats = stats
        return self._stats

    def to(self, device: torch.device | str) -> FeatureBank:
        """Move the retained vectors to another device, in place.

        Args:
            device: where to put them.

        Returns:
            This bank.
        """
        self._device = torch.device(device)
        self._chunks = [chunk.to(self._device) for chunk in self._chunks]
        self._stats = None
        return self

    def state_dict(self) -> dict[str, torch.Tensor | int]:
        """The retained vectors, in a form that survives a round trip to disk.

        The moments are not stored: they are a function of these vectors, and
        recomputing them on load is exact, where storing both invites the two
        to disagree.

        Returns:
            A mapping of plain tensors and ints, ready for :func:`torch.save`
            under ``weights_only``.
        """
        return {"dim": self.dim, "n": self._n, "features": self.features.cpu()}

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any], *, device: torch.device | str = "cpu"
    ) -> FeatureBank:
        """Rebuild a bank from a :meth:`state_dict`.

        Args:
            state: a mapping produced by :meth:`state_dict`.
            device: where to keep the restored vectors.

        Returns:
            The restored bank.

        Raises:
            ValueError: if a key is missing, or the payload does not have the
                shape ``dim`` and ``n`` call for. Both mean it is not one this
                class wrote, so it is rejected rather than half-loaded.
        """
        try:
            dim, n = int(state["dim"]), int(state["n"])
            features = state["features"]
        except KeyError as exc:
            raise ValueError(f"feature bank is missing {exc} key") from None
        if n < 0:
            raise ValueError(f"feature count must not be negative, got {n}")
        if features.shape != (n, dim):
            raise ValueError(
                f"feature bank for dim={dim}, n={n} must hold a ({n}, {dim}) tensor, "
                f"got {tuple(features.shape)}"
            )
        bank = cls(dim, device=device)
        bank.update(features)
        return bank
