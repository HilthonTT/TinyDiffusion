"""Generate images from a trained checkpoint."""

from pathlib import Path

import torch
from torchvision.utils import save_image

from tinydiffusion.data.mnist import MNIST_CHANNELS, denormalize
from tinydiffusion.diffusion.ddim import ddim_sample
from tinydiffusion.diffusion.ddpm import DDPM
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.ema import EMA
from tinydiffusion.training.train_mnist import build_model, read_checkpoint, restore_checkpoint
from tinydiffusion.utils.seed import seed_everything


def default_device() -> str:
    """Return ``"cuda"`` when a GPU is visible, else ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_for_sampling(checkpoint: Path, device: str | None = None) -> tuple[DDPM, EMA, TrainConfig]:
    """Rebuild a model from a checkpoint and its embedded config.

    The architecture is taken from the config stored alongside the weights, so
    a checkpoint stays loadable without the TOML file it was trained from.

    Args:
        checkpoint: file written by
            :func:`~tinydiffusion.training.train_mnist.save_checkpoint`.
        device: device to load onto. Defaults to CUDA when available.

    Returns:
        The model, its EMA weights, and the config it was trained with. The
        config's ``device`` reflects where it was just loaded, not where it was
        trained.

    Raises:
        KeyError: if the checkpoint predates config provenance.
    """
    resolved = device or default_device()
    ckpt = read_checkpoint(checkpoint, device=resolved)
    if "config" not in ckpt:
        raise KeyError(f"{checkpoint} stores no config; cannot infer the architecture")

    cfg = TrainConfig.from_mapping({**ckpt["config"], "device": resolved})
    ddpm = build_model(cfg).to(resolved)
    ema = EMA(ddpm.eps_model, decay=cfg.ema_decay, warmup=cfg.ema_warmup)
    restore_checkpoint(ckpt, ddpm=ddpm, ema=ema)
    return ddpm, ema, cfg


def sample_from_checkpoint(
    checkpoint: Path,
    out: Path,
    *,
    num_images: int = 8,
    num_steps: int | None = None,
    eta: float = 0.0,
    seed: int | None = None,
    device: str | None = None,
) -> Path:
    """Draw images from a checkpoint's EMA weights and write them as a grid.

    Args:
        checkpoint: file to sample from.
        out: image path to write.
        num_images: how many images to generate.
        num_steps: DDIM steps. Defaults to the checkpoint's ``sample_steps``.
        eta: 0.0 is deterministic DDIM; 1.0 reproduces DDPM ancestral sampling.
        seed: seed applied before sampling, or None to leave the RNG alone.
        device: device to sample on. Defaults to CUDA when available.

    Returns:
        The path that was written.

    Raises:
        ValueError: if ``num_images`` is not positive.
    """
    if num_images < 1:
        raise ValueError(f"num_images must be positive, got {num_images}")
    if seed is not None:
        seed_everything(seed)

    ddpm, ema, cfg = load_for_sampling(checkpoint, device)
    images = ddim_sample(
        ddpm,
        num_images,
        (MNIST_CHANNELS, cfg.image_size, cfg.image_size),
        cfg.device,
        num_steps=num_steps if num_steps is not None else cfg.sample_steps,
        eta=eta,
        model=ema.module,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(images), out, nrow=min(8, num_images))
    return out
