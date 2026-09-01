"""The Inception Score (Salimans et al. 2016, https://arxiv.org/abs/1606.03498).

Alone among the metrics here it never looks at a real image. It asks the
Inception classifier two questions about the generated set: is each sample
confidently *some* class, and do the samples between them cover *many* classes.
The score is the exponentiated KL between the two, so it rewards a set that is
individually decisive and collectively varied, and punishes both a blurry set
nothing classifies and a set that is a thousand copies of one dog.

Which is also its limitation, and the reason it is reported alongside FID rather
than instead of it. Not looking at the real data means it cannot tell you
whether the samples resemble *your* dataset — only whether they resemble
ImageNet. On MNIST that is close to meaningless: handwritten digits are not an
ImageNet class, and a model that has learned them perfectly still scores
whatever Inception happens to think a 7 is. It earns its keep on natural images,
and it is here because it costs nothing on top of a score that is already
running Inception over every sample.

The convention it is quoted under is a mean and a standard deviation over
``splits`` disjoint chunks of the sample set. That spread is a spread over
subsets of one run, not over runs, so it says whether the number is stable
rather than whether two models differ — the same caveat KID's spread carries.
"""

from dataclasses import dataclass

import torch

__all__ = [
    "DEFAULT_IS_SPLITS",
    "InceptionScoreResult",
    "inception_score_from_probs",
]

DEFAULT_IS_SPLITS = 10
"""Chunks the score is averaged over. Ten is what the paper used and everyone quotes."""

_EPS = 1e-12
"""Floor under a probability before its log. p log p is 0 at p = 0; log is not."""


@dataclass(frozen=True, slots=True)
class InceptionScoreResult:
    """An Inception Score and the spread it was measured with.

    Attributes:
        mean: the score, averaged over the splits. Higher is better, and it is
            bounded above by the number of classes Inception knows.
        std: standard deviation across the splits. A spread over subsets of one
            sample set, so it says how stable this number is, not whether two
            models differ.
        splits: how many chunks it was averaged over.
        split_size: images per chunk. The score is biased by this — a smaller
            chunk covers fewer classes and scores lower — so two runs are only
            comparable at the same one.
    """

    mean: float
    std: float
    splits: int
    split_size: int


def inception_score_from_probs(
    probs: torch.Tensor, *, splits: int = DEFAULT_IS_SPLITS
) -> InceptionScoreResult:
    """Score a set of samples from their Inception class probabilities.

    Args:
        probs: ``(N, num_classes)`` softmax outputs, one row per generated
            image. Rows are assumed already normalised; nothing here renormalises
            them, so a set of logits would produce a number rather than an error.
        splits: how many disjoint chunks to average over. The chunks are taken
            in order rather than shuffled, which is what makes the result a
            function of the samples alone — the caller's generation order
            already cycles the classes.

    Returns:
        The score and its spread.

    Raises:
        ValueError: if `probs` is not 2-D, holds no rows, or `splits` is not
            positive or exceeds the number of rows.
    """
    if probs.ndim != 2:
        raise ValueError(f"expected (N, classes) probabilities, got shape {tuple(probs.shape)}")
    total = probs.shape[0]
    if total == 0:
        raise ValueError("no probabilities to score")
    if splits < 1:
        raise ValueError(f"splits must be positive, got {splits}")
    if splits > total:
        raise ValueError(f"cannot split {total} samples into {splits} chunks")

    working = probs.double()
    size = total // splits
    scores = []
    for index in range(splits):
        chunk = working[index * size : (index + 1) * size]
        marginal = chunk.mean(dim=0, keepdim=True)
        kl = (chunk * (chunk.clamp_min(_EPS).log() - marginal.clamp_min(_EPS).log())).sum(dim=1)
        scores.append(kl.mean().exp())

    stacked = torch.stack(scores)
    return InceptionScoreResult(
        mean=float(stacked.mean()),
        std=float(stacked.std(unbiased=True)) if splits > 1 else 0.0,
        splits=splits,
        split_size=size,
    )
