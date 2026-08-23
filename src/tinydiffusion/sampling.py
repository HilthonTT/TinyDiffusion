"""Generate images from a trained checkpoint."""

from collections.abc import Sequence
from pathlib import Path

import torch
from torchvision.utils import save_image

from tinydiffusion.data.datasets import denormalize
from tinydiffusion.diffusion.gaussian_diffusion import Diffusion
from tinydiffusion.diffusion.guidance import conditioned, cycled_labels
from tinydiffusion.diffusion.samplers import get_sampler
from tinydiffusion.training.checkpoints import read_checkpoint, restore_checkpoint
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.ema import EMA
from tinydiffusion.training.model import build_model
from tinydiffusion.utils.device import resolve_device
from tinydiffusion.utils.precision import DEFAULT_PRECISION, apply_precision, resolve_precision
from tinydiffusion.utils.seed import seed_everything


def load_for_sampling(
    checkpoint: Path, device: str | None = None
) -> tuple[Diffusion, EMA, TrainConfig]:
    """Rebuild a model from a checkpoint and its embedded config.

    The architecture is taken from the config stored alongside the weights, so
    a checkpoint stays loadable without the TOML file it was trained from.

    Args:
        checkpoint: file written by
            :func:`~tinydiffusion.training.checkpoints.save_checkpoint`.
        device: device to load onto. Defaults to CUDA when available.

    Returns:
        The model, its EMA weights, and the config it was trained with. The
        config's ``device`` reflects where it was just loaded, not where it was
        trained.

    Raises:
        KeyError: if the checkpoint predates config provenance.
    """
    resolved = resolve_device(device)
    ckpt = read_checkpoint(checkpoint, device=resolved)
    if "config" not in ckpt:
        raise KeyError(f"{checkpoint} stores no config; cannot infer the architecture")

    cfg = TrainConfig.from_mapping({**ckpt["config"], "device": resolved})
    diffusion = build_model(cfg).to(resolved)
    ema = EMA(diffusion.net, decay=cfg.ema_decay, warmup=cfg.ema_warmup)
    restore_checkpoint(ckpt, diffusion=diffusion, ema=ema)
    return diffusion, ema, cfg


def resolve_labels(
    labels: Sequence[int] | None,
    *,
    num_images: int,
    num_classes: int | None,
    device: torch.device | str,
) -> torch.Tensor | None:
    """Turn the requested classes into one label per image.

    Args:
        labels: the classes asked for, repeated in order until every image has
            one — so ``[3]`` fills the grid with 3s and ``[0, 1]`` alternates.
            None asks for the default, which is one image per class, cycling.
        num_images: how many images will be generated.
        num_classes: the checkpoint's class count, or None if it is
            unconditional.
        device: device to build the tensor on.

    Returns:
        A ``(num_images,)`` long tensor, or None for an unconditional model.

    Raises:
        ValueError: if labels are asked of an unconditional checkpoint, the
            sequence is empty, or a label names no class.
    """
    if num_classes is None:
        if labels is not None:
            raise ValueError("this checkpoint is unconditional, so it cannot be given labels")
        return None

    if labels is None:
        return cycled_labels(num_images, num_classes, device)
    if not labels:
        raise ValueError("no labels given")

    out_of_range = sorted({label for label in labels if not 0 <= label < num_classes})
    if out_of_range:
        raise ValueError(
            f"label(s) {', '.join(map(str, out_of_range))} outside "
            f"[0, {num_classes - 1}] for this checkpoint"
        )

    asked = torch.tensor(labels, dtype=torch.long, device=device)
    return asked[torch.arange(num_images, device=device) % asked.shape[0]]


def sample_from_checkpoint(
    checkpoint: Path,
    out: Path,
    *,
    num_images: int = 8,
    num_steps: int | None = None,
    eta: float = 0.0,
    sampler: str | None = None,
    spacing: str | None = None,
    labels: Sequence[int] | None = None,
    guidance: float | None = None,
    guidance_rescale: float | None = None,
    seed: int | None = None,
    device: str | None = None,
    precision: str = DEFAULT_PRECISION,
) -> Path:
    """Draw images from a checkpoint's EMA weights and write them as a grid.

    Args:
        checkpoint: file to sample from.
        out: image path to write.
        num_images: how many images to generate.
        num_steps: denoising steps. Defaults to the checkpoint's
            ``sample_steps``.
        eta: 0.0 is deterministic DDIM; 1.0 reproduces DDPM ancestral
            sampling. Only ``ddim`` accepts a non-zero value.
        sampler: which sampler to draw with, or None for the checkpoint's own.
            See :data:`~tinydiffusion.diffusion.samplers.SAMPLERS`.
        spacing: which subsequence of the training schedule to visit, or None
            for the checkpoint's own. ``quadratic`` packs the steps towards
            ``t = 0`` and is worth trying at low `num_steps`; see
            :data:`~tinydiffusion.diffusion.ddim.SPACINGS`.
        labels: classes to generate, cycled over the grid. Conditional
            checkpoints only; see :func:`resolve_labels` for the default.
        guidance: classifier-free guidance scale, or None to use the
            checkpoint's. 1.0 is the plain conditional prediction; higher
            sharpens class identity and costs a second forward pass per step.
        guidance_rescale: how much of the scale inflation guidance causes to
            correct, or None for the checkpoint's. See
            :func:`~tinydiffusion.diffusion.guidance.rescale_guided`; 0.7 is
            the published value, and it is worth reaching for whenever
            `guidance` is above about 3.
        seed: seed applied before sampling, or None to leave the RNG alone.
        device: device to sample on. Defaults to CUDA when available.
        precision: what to run the network in; see
            :mod:`tinydiffusion.utils.precision`. The default is float32, which
            is both the slowest and the only one that does not depend on the
            hardware it ran on.

    Returns:
        The path that was written.

    Raises:
        ValueError: if ``num_images`` is not positive, no sampler, spacing or
            precision goes by that name, or the conditioning arguments do not
            match the checkpoint.
    """
    if num_images < 1:
        raise ValueError(f"num_images must be positive, got {num_images}")
    if seed is not None:
        seed_everything(seed)

    diffusion, ema, cfg = load_for_sampling(checkpoint, device)
    if guidance is not None and cfg.num_classes is None and guidance != 1.0:
        raise ValueError("this checkpoint is unconditional, so guidance does not apply")

    y = resolve_labels(
        labels, num_images=num_images, num_classes=cfg.num_classes, device=cfg.device
    )
    scale = cfg.guidance if guidance is None else guidance
    rescale = cfg.guidance_rescale if guidance_rescale is None else guidance_rescale
    # Under the conditioning wrapper, not over it: guidance extrapolates and
    # rescales in float32 whatever the network itself runs in.
    net = apply_precision(ema.module, resolve_precision(precision, cfg.device), cfg.device)

    images = get_sampler(cfg.sampler if sampler is None else sampler)(
        diffusion,
        num_images,
        (cfg.dataset_spec().channels, cfg.image_size, cfg.image_size),
        cfg.device,
        num_steps=num_steps if num_steps is not None else cfg.sample_steps,
        eta=eta,
        model=conditioned(net, y, num_classes=cfg.num_classes, scale=scale, rescale=rescale),
        spacing=cfg.sample_spacing if spacing is None else spacing,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(images), out, nrow=grid_width(num_images, cfg.num_classes, labels))
    return out


def grid_width(num_images: int, num_classes: int | None, labels: Sequence[int] | None) -> int:
    """Images per row in the saved grid.

    A default conditional grid cycles the classes, so laying it out one class
    per column makes the rows repeats of the same label sequence and the
    columns directly comparable. Everything else keeps the usual eight.

    Args:
        num_images: how many images the grid holds.
        num_classes: the checkpoint's class count, or None if unconditional.
        labels: the classes the caller asked for, if any.

    Returns:
        A positive row width.
    """
    if labels is None and num_classes is not None and num_classes <= 16:
        return min(num_classes, num_images)
    return min(8, num_images)
