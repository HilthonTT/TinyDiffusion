"""Inception-v3 activations, the feature space FID is conventionally measured in."""

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinydiffusion.data.mnist import denormalize

INCEPTION_DIM = 2048
"""Width of the final average-pooled Inception-v3 activation."""

INCEPTION_SIZE = 299
"""Resolution Inception-v3 was trained at."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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

    # Declared so the registered buffers type as tensors rather than as the
    # Tensor | Module union `nn.Module.__getattr__` is annotated with.
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
        except ImportError as exc:  # pragma: no cover - torchvision is a hard dep
            raise ImportError("computing FID needs torchvision installed") from exc

        # transform_input applies the original TF-graph rescaling on top of
        # ImageNet normalisation; we normalise ourselves, so it stays off.
        net = inception_v3(weights=weights, transform_input=False, init_weights=False)
        # The 1000-way classifier is the part FID explicitly does not want: the
        # score is over the 2048-d pooled features feeding it.
        net.fc = nn.Identity()
        net.eval()
        self.net = net
        self.dim = INCEPTION_DIM

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        # Buffers rather than plain tensors so `.to(device)` moves them with the
        # rest of the module.
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
            # Inception has no greyscale input; replicating is what every FID
            # implementation does for single-channel data.
            x = x.expand(-1, 3, -1, -1)
        if x.shape[-2:] != (INCEPTION_SIZE, INCEPTION_SIZE):
            # antialias matters here: MNIST at 32px is being upsampled ~9x, and
            # the two resize paths otherwise disagree enough to move the score.
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
