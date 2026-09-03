"""Inception-v3 activations, the feature space FID is conventionally measured in.

One network, three readings, and the metrics that want them differ.

- The **pooled** 2048-d activation is what FID, KID and precision/recall are
  measured in. It is spatially averaged, which is what makes it a summary of
  *what* is in an image and blind to *where*.
- The **spatial** reading keeps that geometry: the first few channels of an
  intermediate feature map, unpooled. That is the space sFID is measured in,
  and it is why sFID notices the spatial incoherence — a face with its features
  rearranged — that a pooled score marks as fine.
- The **class probabilities** are what the Inception Score reads. Alone among
  the three it needs no real images at all, which is its appeal and also its
  limitation: it measures whether samples look like confident ImageNet classes,
  not whether they look like your data.

All three come out of one forward pass, through :meth:`InceptionFeatures.analyse`.
That matters because the alternative is running Inception once per metric over
every image on both sides of a score.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinydiffusion.data.datasets import denormalize

INCEPTION_DIM = 2048
"""Width of the final average-pooled Inception-v3 activation."""

INCEPTION_CLASSES = 1000
"""ImageNet classes the Inception classifier scores, which the IS reads."""

SFID_CHANNELS = 7
"""Intermediate channels sFID keeps.

Nash et al. 2021 (https://arxiv.org/abs/2103.03841) take the first seven
channels of the mixed 6/conv feature map rather than all 768, which is what
keeps the feature width comparable to FID's 2048 and the covariance the same
size of problem. Seven is theirs; there is nothing special about the number
beyond the width it lands on.
"""

SFID_SPATIAL = 17
"""Height and width of that feature map at Inception's own input resolution."""

SFID_DIM = SFID_CHANNELS * SFID_SPATIAL * SFID_SPATIAL
"""Width of the spatial feature vector: 7 x 17 x 17, or 2023."""

INCEPTION_SIZE = 299
"""Resolution Inception-v3 was trained at."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class InceptionOutputs:
    """Everything one Inception pass produces, for the metrics that read them.

    Attributes:
        pool: ``(B, 2048)`` pooled activations — FID, KID, precision/recall.
        spatial: ``(B, 2023)`` flattened intermediate feature map — sFID.
        probs: ``(B, 1000)`` softmax class probabilities — Inception Score.
    """

    pool: torch.Tensor
    spatial: torch.Tensor
    probs: torch.Tensor


@runtime_checkable
class FeatureExtractor(Protocol):
    """A network mapping model-space images to a fixed-width feature vector.

    Anything with a feature width and a ``(B, C, H, W) -> (B, dim)`` call can
    stand in here, which keeps the FID plumbing testable without downloading
    Inception weights.
    """

    dim: int

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Map a batch of images in [-1, 1] to features."""
        ...


class InceptionFeatures(nn.Module):
    """Pool-3 activations of a pretrained Inception-v3.

    Scores from this extractor are comparable *between runs of this project*,
    which is what a training curve needs. They are not directly comparable with
    published FIDs: the canonical numbers come from the original TensorFlow
    Inception graph, and torchvision's port differs enough in weights and
    preprocessing to shift the absolute value. The ordering it induces over
    checkpoints is the part that carries over.

    Attributes:
        dim: feature width, always :data:`INCEPTION_DIM`.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, *, weights: str | None = "DEFAULT") -> None:
        """Load pretrained Inception-v3 and strip its classifier.

        Args:
            weights: torchvision weights enum name, or None to build the
                network untrained — only useful in tests, since untrained
                features make the score meaningless.

        Raises:
            ImportError: if torchvision is not installed.
        """
        super().__init__()
        try:
            from torchvision.models import inception_v3
        except ImportError as exc:  # pragma: no cover
            raise ImportError("computing FID needs torchvision installed") from exc

        # torchvision's ported Google weights expect input scaled to [-1, 1];
        # transform_input=True maps ImageNet-normalised input to that convention,
        # which is what torchvision itself does whenever weights are loaded.
        net = inception_v3(weights=weights, transform_input=True, init_weights=False)
        self.classifier = net.fc
        net.fc = nn.Identity()
        net.eval()
        self.net = net
        self.dim = INCEPTION_DIM

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Bring model-space images into Inception's input convention.

        Args:
            images: ``(B, C, H, W)`` in [-1, 1], with ``C`` of 1 or 3.

        Returns:
            ``(B, 3, 299, 299)`` ImageNet-normalised images.

        Raises:
            ValueError: if the batch is not 4-D or has an unusable channel count.
        """
        if images.ndim != 4:
            raise ValueError(f"expected (B, C, H, W) images, got shape {tuple(images.shape)}")
        channels = images.shape[1]
        if channels not in (1, 3):
            raise ValueError(f"expected 1 or 3 channels, got {channels}")

        x = denormalize(images.to(self.mean.dtype))
        if channels == 1:
            x = x.expand(-1, 3, -1, -1)
        if x.shape[-2:] != (INCEPTION_SIZE, INCEPTION_SIZE):
            x = F.interpolate(
                x,
                size=(INCEPTION_SIZE, INCEPTION_SIZE),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of images.

        Args:
            images: ``(B, C, H, W)`` in [-1, 1].

        Returns:
            ``(B, 2048)`` activations.
        """
        return self.net(self.preprocess(images.to(self.mean.device)))

    @torch.no_grad()
    def analyse(self, images: torch.Tensor) -> InceptionOutputs:
        """Run one pass and return every reading the metrics take from it.

        The spatial map is captured with a forward hook on the block that
        produces it rather than by re-implementing Inception's forward, which
        would be a copy of torchvision's to keep in step with torchvision's.

        Args:
            images: ``(B, C, H, W)`` in [-1, 1].

        Returns:
            The pooled features, the flattened spatial ones, and the class
            probabilities, all for the same images in the same order.
        """
        captured: list[torch.Tensor] = []
        handle = self.net.Mixed_6e.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        try:
            pool = self.net(self.preprocess(images.to(self.mean.device)))
        finally:
            handle.remove()

        spatial = captured[0][:, :SFID_CHANNELS].flatten(1)
        probs = self.classifier(pool).softmax(dim=-1)
        return InceptionOutputs(pool=pool, spatial=spatial, probs=probs)
