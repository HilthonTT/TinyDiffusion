"""Score a checkpoint's samples against real data with FID."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from tinydiffusion.data.mnist import MNIST_CHANNELS, mnist_dataloader
from tinydiffusion.diffusion.ddim import ddim_sample
from tinydiffusion.diffusion.gaussian_diffusion import Diffusion
from tinydiffusion.diffusion.guidance import conditioned
from tinydiffusion.metrics.fid import FeatureStats, fid_from_stats
from tinydiffusion.metrics.inception import FeatureExtractor
from tinydiffusion.sampling import load_for_sampling, resolve_labels
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.utils.seed import seed_everything

DEFAULT_FID_IMAGES = 10_000
"""Samples per side. Below a few thousand the score is dominated by its own bias."""


@dataclass(slots=True)
class FidResult:
    """The outcome of scoring one checkpoint with FID.

    Attributes:
        checkpoint: the file that was scored.
        split: the real split the samples were compared against.
        fid: the distance. Lower is better; it has no upper bound and no
            absolute meaning, only ordering between comparable runs.
        num_generated: how many samples were drawn.
        num_real: how many real images they were compared against.
        feature_dim: width of the feature space the score was taken in.
        num_steps: DDIM steps used to draw the samples.
        guidance: the guidance scale used, or None if unconditional.
        used_ema: whether the EMA weights were sampled.
    """

    checkpoint: Path
    split: str
    fid: float
    num_generated: int
    num_real: int
    feature_dim: int
    num_steps: int
    guidance: float | None
    used_ema: bool

    @property
    def undersampled(self) -> bool:
        """Whether either side has too few images for a stable covariance.

        A ``d``-dimensional covariance needs more than ``d`` samples to be full
        rank. Below that the estimate is singular and FID is biased upwards by
        an amount that depends on the sample count, so two runs scored with
        different counts are not comparable.
        """
        return min(self.num_generated, self.num_real) <= self.feature_dim

    def format(self) -> str:
        """Render the result as a short report.

        Returns:
            A multi-line string: the headline score, then how it was produced.
        """
        weights = "ema" if self.used_ema else "raw"
        lines = [
            f"{self.checkpoint} | {self.split} split | {weights} weights",
            f"fid {self.fid:.3f}",
            "",
            f"{self.num_generated} generated vs {self.num_real} real images",
            f"{self.num_steps} ddim steps"
            + (f" | guidance {self.guidance:g}" if self.guidance is not None else ""),
        ]
        if self.undersampled:
            lines += [
                "",
                f"warning: fewer than {self.feature_dim} images per side, so the "
                "covariance is singular and this score is biased upwards. Compare "
                "it only with scores taken at the same image count.",
            ]
        return "\n".join(lines)


def accumulate_features(
    images: Iterable[torch.Tensor],
    extractor: FeatureExtractor,
    *,
    stats: FeatureStats | None = None,
    limit: int | None = None,
) -> FeatureStats:
    """Fold batches of images into feature statistics.

    Args:
        images: an iterable of ``(B, C, H, W)`` batches in [-1, 1].
        extractor: the feature network to run them through.
        stats: accumulator to add to, or None to start a fresh one.
        limit: stop once this many images have been seen. The batch that
            crosses the limit is truncated, so the count lands exactly.

    Returns:
        The accumulator, for chaining.
    """
    stats = stats if stats is not None else FeatureStats(extractor.dim)
    for batch in images:
        if limit is not None:
            room = limit - stats.n
            if room <= 0:
                break
            batch = batch[:room]
        stats.update(extractor(batch))
    return stats


def generate_images(
    diffusion: Diffusion,
    net: torch.nn.Module,
    cfg: TrainConfig,
    *,
    num_images: int,
    batch_size: int,
    num_steps: int,
    eta: float,
    guidance: float,
) -> Iterable[torch.Tensor]:
    """Draw samples in batches, yielding each as it is produced.

    Generating lazily keeps peak memory at one batch however many images the
    score covers.

    Args:
        diffusion: the loaded process.
        net: the network to sample from, typically the EMA weights.
        cfg: the checkpoint's config, for the schedule and image geometry.
        num_images: total images to draw.
        batch_size: images per batch.
        num_steps: DDIM steps per batch.
        eta: 0.0 is deterministic DDIM; 1.0 reproduces ancestral DDPM.
        guidance: classifier-free guidance scale, ignored when unconditional.

    Yields:
        ``(b, C, image_size, image_size)`` batches in [-1, 1].
    """
    remaining = num_images
    while remaining > 0:
        batch = min(batch_size, remaining)
        # Cycled labels rather than random ones: MNIST is near-uniform over its
        # classes, and cycling removes the class-imbalance noise that random
        # draws would add to the score at small sample counts.
        y = resolve_labels(None, num_images=batch, num_classes=cfg.num_classes, device=cfg.device)
        yield ddim_sample(
            diffusion,
            batch,
            (MNIST_CHANNELS, cfg.image_size, cfg.image_size),
            cfg.device,
            num_steps=num_steps,
            eta=eta,
            model=conditioned(net, y, num_classes=cfg.num_classes, scale=guidance),
        )
        remaining -= batch


@torch.no_grad()
def fid_for_checkpoint(
    checkpoint: Path,
    *,
    num_images: int = DEFAULT_FID_IMAGES,
    split: str = "train",
    batch_size: int | None = None,
    data_root: Path | None = None,
    num_steps: int | None = None,
    eta: float = 0.0,
    guidance: float | None = None,
    use_ema: bool = True,
    seed: int = 0,
    device: str | None = None,
    extractor: FeatureExtractor | None = None,
    progress: bool = True,
) -> FidResult:
    """Sample a checkpoint and score the samples against real images.

    Unlike the held-out loss, this measures the thing you actually care about:
    how close the distribution of generated images is to the real one, in the
    feature space of a pretrained classifier. It is slow — every score draws
    ``num_images`` samples through the full DDIM chain — so it belongs at the
    end of a run rather than inside the training loop.

    The number is only meaningful in comparison. Hold ``num_images``,
    ``split``, ``num_steps`` and the extractor fixed across the checkpoints
    being compared, since each of them moves the score on its own.

    Args:
        checkpoint: file to score.
        num_images: how many samples to draw, and how many real images to
            compare them against.
        split: ``"train"`` or ``"test"``. The convention is the training split,
            which is both larger and the distribution the model was fit to.
        batch_size: images per batch, or None to reuse the checkpoint's.
        data_root: dataset directory, or None to reuse the checkpoint's.
        num_steps: DDIM steps per sample. Defaults to the checkpoint's.
        eta: 0.0 is deterministic DDIM; 1.0 reproduces ancestral DDPM.
        guidance: classifier-free guidance scale, or None for the checkpoint's.
            Worth sweeping: guidance trades diversity for fidelity, and FID
            usually has an interior minimum somewhere above 1.
        use_ema: sample the EMA weights, which is what ``sample`` uses.
        seed: seed applied before generating, making the samples reproducible.
        device: device to score on. Defaults to CUDA when available.
        extractor: feature network, or None to load Inception-v3, downloading
            the weights on first use.
        progress: draw progress bars.

    Returns:
        The scored result.

    Raises:
        ValueError: if ``num_images`` is below 2, leaving the covariance
            undefined, or ``split`` is not ``"train"`` or ``"test"``.
    """
    if num_images < 2:
        raise ValueError(f"num_images must be at least 2 for a covariance, got {num_images}")
    if split not in ("test", "train"):
        raise ValueError(f"unknown split {split!r}, expected 'test' or 'train'")

    diffusion, ema, cfg = load_for_sampling(checkpoint, device)
    net = ema.module if use_ema else diffusion.net
    steps = num_steps if num_steps is not None else cfg.sample_steps
    scale = cfg.guidance if guidance is None else guidance
    batch = batch_size if batch_size is not None else cfg.batch_size

    if extractor is None:
        # Imported here so the module stays importable, and the rest of the CLI
        # keeps working, on a machine that cannot reach the weights.
        from tinydiffusion.metrics.inception import InceptionFeatures

        extractor = InceptionFeatures().to(cfg.device)

    loader = mnist_dataloader(
        data_root if data_root is not None else cfg.data_root,
        batch_size=batch,
        train=split == "train",
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
        # Fixed order and no dropped tail, so the reference side of the score
        # depends only on the split and the image count.
        shuffle=False,
        drop_last=False,
    )
    real_batches = (x.to(cfg.device, non_blocking=True) for x, _ in loader)

    real = accumulate_features(
        tqdm(
            real_batches,
            desc=f"fid real ({split})",
            total=-(-num_images // batch),
            disable=not progress,
        ),
        extractor,
        limit=num_images,
    )

    seed_everything(seed)
    generated = accumulate_features(
        tqdm(
            generate_images(
                diffusion,
                net,
                cfg,
                num_images=num_images,
                batch_size=batch,
                num_steps=steps,
                eta=eta,
                guidance=scale,
            ),
            desc="fid generated",
            total=-(-num_images // batch),
            disable=not progress,
        ),
        extractor,
    )

    if real.n < num_images:
        # The split ran out first; scoring uneven sides is legitimate but the
        # caller should know the count they asked for is not what they got.
        print(f"only {real.n} images in the {split} split, scoring against those")

    return FidResult(
        checkpoint=checkpoint,
        split=split,
        fid=fid_from_stats(generated, real),
        num_generated=generated.n,
        num_real=real.n,
        feature_dim=extractor.dim,
        num_steps=steps,
        guidance=scale if cfg.num_classes is not None else None,
        used_ema=use_ema,
    )
