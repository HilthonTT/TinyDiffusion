"""Score a checkpoint's samples against real data.

FID is the headline number and always computed. KID and precision/recall are
opt-in, because they are the ones that need the feature vectors kept rather
than summarised — see :mod:`tinydiffusion.metrics.features` for what that
costs. They answer questions FID cannot: whether a score taken over a few
thousand images means anything (:mod:`tinydiffusion.metrics.kid`), and whether
a bad score is a quality or a coverage failure
(:mod:`tinydiffusion.metrics.precision_recall`).
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from tinydiffusion.data.datasets import image_dataloader
from tinydiffusion.diffusion.ddim import DEFAULT_SPACING
from tinydiffusion.diffusion.gaussian_diffusion import Diffusion
from tinydiffusion.diffusion.guidance import conditioned
from tinydiffusion.diffusion.samplers import DEFAULT_SAMPLER, get_sampler
from tinydiffusion.metrics.cache import (
    load_reference_features,
    load_reference_stats,
    reference_features_path,
    reference_stats_path,
    save_reference_features,
    save_reference_stats,
)
from tinydiffusion.metrics.features import FeatureBank
from tinydiffusion.metrics.fid import FeatureStats, fid_from_stats
from tinydiffusion.metrics.inception import FeatureExtractor
from tinydiffusion.metrics.kid import (
    DEFAULT_KID_SUBSET_SIZE,
    DEFAULT_KID_SUBSETS,
    KidResult,
    kid_from_features,
)
from tinydiffusion.metrics.precision_recall import (
    DEFAULT_NEIGHBOURS,
    PrecisionRecall,
    precision_recall_from_features,
)
from tinydiffusion.sampling import load_for_sampling
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.utils.seed import seed_everything

DEFAULT_FID_IMAGES = 10_000
"""Samples per side. Below a few thousand the score is dominated by its own bias."""

type FeatureSink = FeatureStats | FeatureBank
"""What a pass over images can fold its features into.

The moments alone, which is all FID needs and all that fits in constant memory,
or the vectors themselves, which KID and precision/recall need. The two share
the ``update``-and-``n`` interface :func:`accumulate_features` uses, and nothing
else."""


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
        sampler: which sampler drew them.
        spacing: which timestep spacing they were drawn over. Like the sampler
            it moves the score without changing the model, so it is recorded
            with the count of steps it placed.
        guidance: the guidance scale used, or None if unconditional.
        guidance_rescale: the guidance rescale factor used. 0 is plain
            guidance, and it moves the score like any other sampling setting,
            so it is recorded alongside the scale it corrects.
        used_ema: whether the EMA weights were sampled.
        kid: the Kernel Inception Distance and its spread, or None if it was
            not asked for. Unlike `fid` it is unbiased, so it stays comparable
            between scores taken over different image counts.
        precision_recall: the manifold precision and recall, or None if they
            were not asked for. They split a bad score into its two causes,
            which neither `fid` nor `kid` can do.
    """

    checkpoint: Path
    split: str
    fid: float
    num_generated: int
    num_real: int
    feature_dim: int
    num_steps: int
    guidance: float | None
    guidance_rescale: float
    used_ema: bool
    sampler: str = DEFAULT_SAMPLER
    spacing: str = DEFAULT_SPACING
    kid: KidResult | None = None
    precision_recall: PrecisionRecall | None = None

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
        ]
        if self.kid is not None:
            # The spread is the point of reporting KID at all: it says whether
            # the gap between two checkpoints is bigger than the noise in
            # either measurement.
            lines.append(
                f"kid {self.kid.mean:.5f} +- {self.kid.std:.5f} "
                f"({self.kid.subsets} subsets of {self.kid.subset_size})"
            )
        if self.precision_recall is not None:
            lines.append(
                f"precision {self.precision_recall.precision:.3f} | "
                f"recall {self.precision_recall.recall:.3f} "
                f"(k={self.precision_recall.neighbours})"
            )
        lines += [
            "",
            f"{self.num_generated} generated vs {self.num_real} real images",
            f"{self.num_steps} {self.sampler} steps ({self.spacing} spacing)"
            + (f" | guidance {self.guidance:g}" if self.guidance is not None else "")
            + (
                f" | rescale {self.guidance_rescale:g}"
                if self.guidance is not None and self.guidance_rescale > 0
                else ""
            ),
        ]
        if self.undersampled:
            biased = "this score is" if self.kid is None else "the fid is"
            lines += [
                "",
                f"warning: fewer than {self.feature_dim} images per side, so the "
                f"covariance is singular and {biased} biased upwards. Compare "
                "it only with scores taken at the same image count.",
            ]
            if self.kid is None:
                lines.append("--kid is unbiased at this count and does not have that problem.")
        return "\n".join(lines)


def accumulate_features(
    images: Iterable[torch.Tensor],
    extractor: FeatureExtractor,
    *,
    stats: FeatureSink | None = None,
    limit: int | None = None,
) -> FeatureSink:
    """Fold batches of images into a feature accumulator.

    Args:
        images: an iterable of ``(B, C, H, W)`` batches in [-1, 1].
        extractor: the feature network to run them through.
        stats: accumulator to add to, or None to start fresh moments. A
            :class:`~tinydiffusion.metrics.features.FeatureBank` goes here
            when the vectors themselves are wanted and not only their moments;
            the two share the interface this needs.
        limit: stop once this many images have been seen. The batch that
            crosses the limit is truncated, so the count lands exactly.

    Returns:
        The accumulator, for chaining.
    """
    sink: FeatureSink = stats if stats is not None else FeatureStats(extractor.dim)
    for batch in images:
        if limit is not None:
            room = limit - sink.n
            if room <= 0:
                break
            batch = batch[:room]
        sink.update(extractor(batch))
    return sink


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
    guidance_rescale: float = 0.0,
    sampler: str = DEFAULT_SAMPLER,
    spacing: str = DEFAULT_SPACING,
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
        num_steps: denoising steps per batch.
        eta: 0.0 is deterministic DDIM; 1.0 reproduces ancestral DDPM.
        guidance: classifier-free guidance scale, ignored when unconditional.
        guidance_rescale: guidance rescale factor; see
            :func:`~tinydiffusion.diffusion.guidance.rescale_guided`.
        sampler: which sampler to draw with; a key of
            :data:`~tinydiffusion.diffusion.samplers.SAMPLERS`. Defaults to
            DDIM, which is what every caller wanted before there was a choice.
        spacing: which subsequence of the training schedule to visit; a key of
            :data:`~tinydiffusion.diffusion.ddim.SPACINGS`.

    Yields:
        ``(b, C, image_size, image_size)`` batches in [-1, 1].
    """
    draw = get_sampler(sampler)
    produced = 0
    while produced < num_images:
        batch = min(batch_size, num_images - produced)
        # Cycled labels rather than random ones, so the class mix is fixed
        # instead of being one more source of noise in the score. The cycle
        # continues across batch boundaries rather than restarting: at 10k
        # images in batches of 128, restarting leaves classes 8 and 9 with 937
        # samples against 1016 for class 0, an 8% skew the score would read as
        # the model's own.
        y = (
            None
            if cfg.num_classes is None
            else torch.arange(produced, produced + batch, device=cfg.device) % cfg.num_classes
        )
        yield draw(
            diffusion,
            batch,
            (cfg.dataset_spec().channels, cfg.image_size, cfg.image_size),
            cfg.device,
            num_steps=num_steps,
            eta=eta,
            model=conditioned(
                net,
                y,
                num_classes=cfg.num_classes,
                scale=guidance,
                rescale=guidance_rescale,
            ),
            spacing=spacing,
        )
        produced += batch


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
    sampler: str | None = None,
    spacing: str | None = None,
    guidance: float | None = None,
    guidance_rescale: float | None = None,
    use_ema: bool = True,
    seed: int = 0,
    device: str | None = None,
    extractor: FeatureExtractor | None = None,
    cache: bool = True,
    progress: bool = True,
    kid: bool = False,
    kid_subsets: int = DEFAULT_KID_SUBSETS,
    kid_subset_size: int = DEFAULT_KID_SUBSET_SIZE,
    precision_recall: bool = False,
    neighbours: int = DEFAULT_NEIGHBOURS,
) -> FidResult:
    """Sample a checkpoint and score the samples against real images.

    Unlike the held-out loss, this measures the thing you actually care about:
    how close the distribution of generated images is to the real one, in the
    feature space of a pretrained classifier. It is slow — every score draws
    ``num_images`` samples through the full DDIM chain — so it belongs at the
    end of a run rather than inside the training loop.

    The number is only meaningful in comparison. Hold ``num_images``,
    ``split``, ``num_steps``, ``sampler`` and the extractor fixed across the checkpoints
    being compared, since each of them moves the score on its own.

    Args:
        checkpoint: file to score.
        num_images: how many samples to draw, and how many real images to
            compare them against.
        split: ``"train"`` or ``"test"``. The convention is the training split,
            which is both larger and the distribution the model was fit to.
        batch_size: images per batch, or None to reuse the checkpoint's.
        data_root: dataset directory, or None to reuse the checkpoint's.
        num_steps: denoising steps per sample. Defaults to the checkpoint's.
        eta: 0.0 is deterministic DDIM; 1.0 reproduces ancestral DDPM.
        sampler: which sampler to draw with, or None for the checkpoint's own.
            It moves the score like any other sampling setting, so hold it
            fixed across the checkpoints being compared.
        spacing: which timestep spacing to draw over, or None for the
            checkpoint's own. Also a sampling setting, and held fixed for the
            same reason.
        guidance: classifier-free guidance scale, or None for the checkpoint's.
            Worth sweeping: guidance trades diversity for fidelity, and FID
            usually has an interior minimum somewhere above 1.
        guidance_rescale: guidance rescale factor, or None for the
            checkpoint's. Worth sweeping jointly with `guidance`: correcting
            the scale is what usually lets the interior minimum sit at a
            higher scale than it otherwise could.
        use_ema: sample the EMA weights, which is what ``sample`` uses.
        seed: seed applied before generating, making the samples reproducible.
        device: device to score on. Defaults to CUDA when available.
        extractor: feature network, or None to load Inception-v3, downloading
            the weights on first use.
        cache: reuse the real side's features from disk when an entry for this
            exact reference set exists, and write one when it does not. The
            real half of the score does not depend on the checkpoint or on any
            sampling setting, so a sweep over ``guidance`` or ``num_steps``
            otherwise recomputes the identical statistics once per point. See
            :mod:`tinydiffusion.metrics.cache`; False forces the recomputation.
        progress: draw progress bars.
        kid: also compute the Kernel Inception Distance. It is unbiased, so it
            is the number to compare when `num_images` is in the low
            thousands, where FID is mostly reporting its own sample count.
        kid_subsets: subsets to average the KID over.
        kid_subset_size: images per KID subset, per side. Clamped down to the
            smaller of the two sets.
        precision_recall: also estimate manifold precision and recall, which
            split a bad score into "the samples are not realistic" and "the
            samples do not cover the data" — the two failures every single
            number conflates. Quadratic in `num_images`.
        neighbours: k for the precision/recall manifolds.

    Returns:
        The scored result. `kid` and `precision_recall` on it are None unless
        they were asked for.

    Raises:
        ValueError: if ``num_images`` is below 2, leaving the covariance
            undefined, ``split`` is not ``"train"`` or ``"test"``, or no
            sampler or spacing goes by the name given.
    """
    # Both of the opt-in metrics read pairwise structure, so both need the
    # vectors kept rather than folded into moments as they go by.
    retain = kid or precision_recall
    if num_images < 2:
        raise ValueError(f"num_images must be at least 2 for a covariance, got {num_images}")
    if split not in ("test", "train"):
        raise ValueError(f"unknown split {split!r}, expected 'test' or 'train'")

    diffusion, ema, cfg = load_for_sampling(checkpoint, device)
    net = ema.module if use_ema else diffusion.net
    steps = num_steps if num_steps is not None else cfg.sample_steps
    draw_with = cfg.sampler if sampler is None else sampler
    space_with = cfg.sample_spacing if spacing is None else spacing
    scale = cfg.guidance if guidance is None else guidance
    rescale = cfg.guidance_rescale if guidance_rescale is None else guidance_rescale
    batch = batch_size if batch_size is not None else cfg.batch_size

    if extractor is None:
        # Imported here so the module stays importable, and the rest of the CLI
        # keeps working, on a machine that cannot reach the weights.
        from tinydiffusion.metrics.inception import InceptionFeatures

        extractor = InceptionFeatures().to(cfg.device)
    elif isinstance(extractor, nn.Module):
        # A supplied extractor was built without knowing where this would run —
        # `device=None` resolves to CUDA when there is a GPU, whatever device
        # the checkpoint was trained on. Left alone it fails several layers
        # down, as a matmul complaining about mat2 rather than about a device.
        # Only for a Module: FeatureExtractor promises `dim` and a call, so
        # anything else is the caller's to place.
        extractor = extractor.to(cfg.device)

    root = data_root if data_root is not None else cfg.data_root
    # Every input the reference statistics depend on is in the path, so a hit
    # is the same set of images through the same network — and a change to any
    # of them is a miss rather than a stale read.
    cache_path = reference_stats_path(
        root,
        dataset=cfg.dataset,
        split=split,
        num_images=num_images,
        image_size=cfg.image_size,
        extractor=extractor,
    )
    features_path = reference_features_path(
        root,
        dataset=cfg.dataset,
        split=split,
        num_images=num_images,
        image_size=cfg.image_size,
        extractor=extractor,
    )

    real_bank: FeatureBank | None = None
    real: FeatureStats | None = None
    if cache:
        if retain:
            # A bank answers everything a stats entry does and more, so it is
            # the one to look for; the moments entry cannot stand in for it.
            real_bank = load_reference_features(features_path, dim=extractor.dim)
            real = real_bank.stats if real_bank is not None else None
        else:
            real = load_reference_stats(cache_path, dim=extractor.dim)
    if real is None:
        loader = image_dataloader(
            cfg.dataset_spec(),
            root,
            batch_size=batch,
            train=split == "train",
            image_size=cfg.image_size,
            num_workers=cfg.num_workers,
            # Fixed order and no dropped tail, so the reference side of the
            # score depends only on the split and the image count — which is
            # also what makes it safe to cache.
            shuffle=False,
            drop_last=False,
        )
        real_batches = (x.to(cfg.device, non_blocking=True) for x, _ in loader)

        real_sink: FeatureSink = (
            FeatureBank(extractor.dim) if retain else FeatureStats(extractor.dim)
        )
        accumulate_features(
            tqdm(
                real_batches,
                desc=f"fid real ({split})",
                total=-(-num_images // batch),
                disable=not progress,
            ),
            extractor,
            stats=real_sink,
            limit=num_images,
        )
        real_bank = real_sink if isinstance(real_sink, FeatureBank) else None
        real = real_sink.stats if isinstance(real_sink, FeatureBank) else real_sink
        if cache:
            # Both entries when the features were computed: the moments are a
            # free by-product of a pass that has already happened, and writing
            # them means a later FID-only sweep need not read the larger file.
            save_reference_stats(cache_path, real)
            if real_bank is not None:
                save_reference_features(features_path, real_bank)

    seed_everything(seed)
    generated: FeatureSink = FeatureBank(extractor.dim) if retain else FeatureStats(extractor.dim)
    accumulate_features(
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
                guidance_rescale=rescale,
                sampler=draw_with,
                spacing=space_with,
            ),
            desc="fid generated",
            total=-(-num_images // batch),
            disable=not progress,
        ),
        extractor,
        stats=generated,
    )
    generated_bank = generated if isinstance(generated, FeatureBank) else None
    generated_stats = generated.stats if isinstance(generated, FeatureBank) else generated

    kid_score: KidResult | None = None
    if kid and generated_bank is not None and real_bank is not None:
        kid_score = kid_from_features(
            generated_bank,
            real_bank,
            subsets=kid_subsets,
            subset_size=kid_subset_size,
            # Seeded from the run's own seed rather than left to the global
            # RNG, which sampling has advanced by an amount that depends on
            # how many batches it drew.
            generator=torch.Generator().manual_seed(seed),
            device=cfg.device,
        )

    pr_score: PrecisionRecall | None = None
    if precision_recall and generated_bank is not None and real_bank is not None:
        pr_score = precision_recall_from_features(
            generated_bank, real_bank, neighbours=neighbours, device=cfg.device
        )

    if real.n < num_images:
        # The split ran out first; scoring uneven sides is legitimate but the
        # caller should know the count they asked for is not what they got.
        print(f"only {real.n} images in the {split} split, scoring against those")

    return FidResult(
        checkpoint=checkpoint,
        split=split,
        fid=fid_from_stats(generated_stats, real),
        num_generated=generated.n,
        num_real=real.n,
        feature_dim=extractor.dim,
        num_steps=steps,
        guidance=scale if cfg.num_classes is not None else None,
        guidance_rescale=rescale,
        used_ema=use_ema,
        sampler=draw_with,
        spacing=space_with,
        kid=kid_score,
        precision_recall=pr_score,
    )
