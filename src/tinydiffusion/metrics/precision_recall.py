"""Improved precision and recall: the two ways a generator can be wrong, kept apart.

FID and KID each answer with one number, and one number cannot distinguish the
two failures that matter. A model that produces beautiful images of only three
digits and a model that produces all ten as smears can score the same, and the
fix for one is the opposite of the fix for the other.

Precision and recall (Kynkaanniemi et al., 2019) separate them by estimating
each set's *manifold* rather than fitting it a distribution. Around every real
feature vector, put a ball reaching its k-th nearest neighbour among the other
real vectors; their union is a rough outline of where real images live.

* **Precision** is the fraction of generated images that land inside it — how
  much of what the model produces is realistic.
* **Recall** is the same the other way round: the fraction of real images inside
  the generated manifold — how much of the real distribution the model reaches.

So low precision is a quality problem and low recall a coverage one. Guidance
trades one for the other directly, and watching a guidance sweep move them in
opposite directions is the clearest thing either number does.

Both are fractions in [0, 1], both need retained features (:class:`FeatureBank`)
and pairwise distances, and the cost is quadratic in the image count — an
``n x n`` distance matrix, computed in chunks so only a band of it exists at
once.
"""

from dataclasses import dataclass

import torch

from tinydiffusion.metrics.features import FeatureBank

__all__ = [
    "DEFAULT_NEIGHBOURS",
    "PrecisionRecall",
    "compute_precision_recall",
    "precision_recall_from_features",
]

DEFAULT_NEIGHBOURS = 3
"""Neighbours defining each manifold ball. The published value.

k trades the two errors against each other: k=1 gives balls so tight that
ordinary samples fall outside, and a large k inflates them until an outlier's
ball swallows the space between it and the rest of the set.
"""

_CHUNK = 1024
"""Rows of the distance matrix computed at once."""


@dataclass(slots=True)
class PrecisionRecall:
    """How much of each set the other one covers.

    Attributes:
        precision: fraction of generated features inside the real manifold, in
            [0, 1]. Higher is better, and reads as realism.
        recall: fraction of real features inside the generated manifold, in
            [0, 1]. Higher is better, and reads as coverage.
        neighbours: the k the manifolds were built with.
        num_generated: generated features the estimate used.
        num_real: real features the estimate used.
    """

    precision: float
    recall: float
    neighbours: int
    num_generated: int
    num_real: int


@torch.no_grad()
def _kth_neighbour_radius(features: torch.Tensor, k: int) -> torch.Tensor:
    """Distance from each vector to its k-th nearest neighbour in the same set.

    Args:
        features: ``(n, d)`` features.
        k: neighbours to reach, not counting the vector itself.

    Returns:
        ``(n,)`` radii.
    """
    radii = torch.empty(features.shape[0], dtype=features.dtype, device=features.device)
    for start in range(0, features.shape[0], _CHUNK):
        block = torch.cdist(features[start : start + _CHUNK], features)
        radii[start : start + _CHUNK] = block.topk(k + 1, largest=False).values[:, -1]
    return radii


@torch.no_grad()
def _fraction_inside(query: torch.Tensor, manifold: torch.Tensor, radii: torch.Tensor) -> float:
    """Fraction of `query` inside the union of balls around `manifold`.

    Args:
        query: ``(m, d)`` features to test.
        manifold: ``(n, d)`` features the balls are centred on.
        radii: ``(n,)`` ball radii, from :func:`_kth_neighbour_radius`.

    Returns:
        The fraction in [0, 1].
    """
    inside = 0
    for start in range(0, query.shape[0], _CHUNK):
        block = torch.cdist(query[start : start + _CHUNK], manifold)
        inside += int((block <= radii).any(dim=1).sum())
    return inside / query.shape[0]


@torch.no_grad()
def compute_precision_recall(
    fake: torch.Tensor,
    real: torch.Tensor,
    *,
    neighbours: int = DEFAULT_NEIGHBOURS,
    device: torch.device | str | None = None,
) -> PrecisionRecall:
    """Estimate precision and recall between two feature sets.

    Args:
        fake: ``(m, d)`` features of the generated images.
        real: ``(n, d)`` features of the real ones. Need not match `m`.
        neighbours: k, the neighbour each manifold ball reaches to.
        device: where to compute. None uses the features' own device. The work
            is ``m x n`` distances in `d` dimensions, so this is worth pointing
            at a GPU whenever there is one.

    Returns:
        The two fractions, with the counts they were taken over.

    Raises:
        ValueError: if either set is not 2-D, they disagree on the feature
            dimension, `neighbours` is below 1, or a set holds no more vectors
            than `neighbours`, which leaves its k-th neighbour undefined.
    """
    for name, feats in (("fake", fake), ("real", real)):
        if feats.ndim != 2:
            raise ValueError(f"{name} features must be (n, dim), got {tuple(feats.shape)}")
    if fake.shape[1] != real.shape[1]:
        raise ValueError(f"feature dimensions differ: {fake.shape[1]} and {real.shape[1]}")
    if neighbours < 1:
        raise ValueError(f"neighbours must be at least 1, got {neighbours}")
    for name, feats in (("fake", fake), ("real", real)):
        if feats.shape[0] <= neighbours:
            raise ValueError(
                f"need more than {neighbours} {name} features to reach a "
                f"{neighbours}-th neighbour, got {feats.shape[0]}"
            )

    target = torch.device(device) if device is not None else fake.device
    fake = fake.to(device=target, dtype=torch.float32)
    real = real.to(device=target, dtype=torch.float32)

    return PrecisionRecall(
        precision=_fraction_inside(fake, real, _kth_neighbour_radius(real, neighbours)),
        recall=_fraction_inside(real, fake, _kth_neighbour_radius(fake, neighbours)),
        neighbours=neighbours,
        num_generated=fake.shape[0],
        num_real=real.shape[0],
    )


def precision_recall_from_features(
    fake: FeatureBank,
    real: FeatureBank,
    *,
    neighbours: int = DEFAULT_NEIGHBOURS,
    device: torch.device | str | None = None,
) -> PrecisionRecall:
    """Estimate precision and recall between two retained feature sets.

    Args:
        fake: retained features of the generated images.
        real: retained features of the real ones.
        neighbours: k, the neighbour each manifold ball reaches to.
        device: where to compute. See :func:`compute_precision_recall`.

    Returns:
        The two fractions, with the counts they were taken over.

    Raises:
        ValueError: if the two banks disagree on the feature dimension, or
            :func:`compute_precision_recall` rejects their contents.
    """
    if fake.dim != real.dim:
        raise ValueError(f"feature dimensions differ: {fake.dim} and {real.dim}")
    return compute_precision_recall(
        fake.features, real.features, neighbours=neighbours, device=device
    )
