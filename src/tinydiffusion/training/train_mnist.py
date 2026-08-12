"""MNIST training loop for TinyDiffusion."""

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm

from tinydiffusion.data.mnist import MNIST_CHANNELS, denormalize, mnist_dataloader
from tinydiffusion.diffusion.ddim import ddim_sample
from tinydiffusion.diffusion.ddpm import DDPM
from tinydiffusion.diffusion.schedules import cosine_beta_schedule, linear_beta_schedule
from tinydiffusion.models.unet import UNet
from tinydiffusion.training.ema import EMA


@dataclass(slots=True)
class TrainConfig:
    """Hyperparameters for an MNIST training run.

    Defaults target a single consumer GPU and reach recognisable digits within
    a handful of epochs.
    """

    # data
    data_root: Path = Path("data")
    image_size: int = 32
    batch_size: int = 128
    num_workers: int = 4

    # model
    base: int = 64
    ch_mult: tuple[int, ...] = (1, 2, 2)
    n_res: int = 2
    attn_resolutions: tuple[int, ...] = (16,)
    dropout: float = 0.1

    # diffusion
    n_T: int = 1000
    schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # optimisation
    n_epoch: int = 30
    lr: float = 2e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    ema_warmup: int = 2000

    # bookkeeping
    seed: int = 0
    amp: bool = True
    sample_every: int = 1
    n_sample: int = 16
    sample_steps: int = 50
    out_dir: Path = Path("contents")
    ckpt_dir: Path = Path("checkpoints")
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy-free torch CPU and all CUDA devices.

    Args:
        seed: value to seed every generator with.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: TrainConfig) -> DDPM:
    """Construct the U-Net and wrap it in the diffusion process.

    Args:
        cfg: run configuration.

    Returns:
        An untrained :class:`DDPM` on the CPU.
    """
    net = UNet(
        in_channels=MNIST_CHANNELS,
        out_channels=MNIST_CHANNELS,
        base=cfg.base,
        ch_mult=cfg.ch_mult,
        n_res=cfg.n_res,
        attn_resolutions=cfg.attn_resolutions,
        dropout=cfg.dropout,
        image_size=cfg.image_size,
    )
    if cfg.schedule == "cosine":
        betas: torch.Tensor = cosine_beta_schedule(cfg.n_T)
    elif cfg.schedule == "linear":
        betas = linear_beta_schedule(cfg.beta_start, cfg.beta_end, cfg.n_T)
    else:
        raise ValueError(f"unknown schedule {cfg.schedule!r}, expected 'cosine' or 'linear'")

    return DDPM(eps_model=net, betas=betas, n_T=cfg.n_T)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    ddpm: DDPM,
    ema: EMA,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: TrainConfig,
) -> None:
    """Write a resumable checkpoint.

    Saving only ``model.state_dict()`` makes a run unresumable: the optimiser
    moments and the EMA shadow weights are both training state.

    Args:
        path: destination file.
        epoch: index of the epoch just completed.
        ddpm: the diffusion model.
        ema: exponential moving average of the network weights.
        optim: optimiser whose moments should be preserved.
        scaler: AMP gradient scaler.
        cfg: run configuration, stored for provenance.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": ddpm.eps_model.state_dict(),
            "ema": ema.module.state_dict(),
            "ema_step": ema.step,
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        },
        tmp,
    )
    # Rename last so an interrupted save cannot corrupt a good checkpoint.
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    *,
    ddpm: DDPM,
    ema: EMA,
    optim: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: str = "cpu",
) -> int:
    """Restore a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: checkpoint file.
        ddpm: model to load weights into.
        ema: EMA wrapper to load shadow weights into.
        optim: optimiser to restore, or None to skip (e.g. sampling only).
        scaler: AMP scaler to restore, or None to skip.
        device: device to map tensors onto.

    Returns:
        The epoch index to resume from.
    """
    ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    ddpm.eps_model.load_state_dict(ckpt["model"])
    ema.module.load_state_dict(ckpt["ema"])
    ema.step = ckpt.get("ema_step", 0)
    if optim is not None and "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["epoch"]) + 1


@torch.no_grad()
def save_samples(
    ddpm: DDPM,
    ema: EMA,
    real: torch.Tensor,
    cfg: TrainConfig,
    epoch: int,
) -> None:
    """Render a grid of EMA samples above a strip of real images.

    Putting real data in the same grid makes contrast and stroke weight
    directly comparable, which is what tells you whether the model has
    learned the data distribution or merely something digit-shaped.

    Args:
        ddpm: the diffusion model, used for its schedule.
        ema: EMA weights to sample from.
        real: a batch of real images in [-1, 1] to show alongside.
        cfg: run configuration.
        epoch: epoch index, used in the filename.
    """
    shape = (MNIST_CHANNELS, cfg.image_size, cfg.image_size)
    fake = ddim_sample(
        ddpm,
        cfg.n_sample,
        shape,
        cfg.device,
        n_steps=cfg.sample_steps,
        eta=0.0,
        model=ema.module,
    )
    n_real = min(cfg.n_sample, real.shape[0])
    grid = torch.cat([denormalize(fake), denormalize(real[:n_real].to(cfg.device))], dim=0)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    save_image(grid, cfg.out_dir / f"sample_{epoch:04d}.png", nrow=min(8, cfg.n_sample))


def train_mnist(cfg: TrainConfig | None = None, resume: Path | None = None) -> DDPM:
    """Train a DDPM on MNIST.

    Args:
        cfg: run configuration. Defaults are used when omitted.
        resume: checkpoint to continue from, or None to start fresh.

    Returns:
        The trained model, with EMA weights already swapped in.
    """
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)

    ddpm = build_model(cfg).to(cfg.device)
    ema = EMA(ddpm.eps_model, decay=cfg.ema_decay, warmup=cfg.ema_warmup)
    ema.module.to(cfg.device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=cfg.lr)
    use_amp = cfg.amp and cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    if resume is not None:
        start_epoch = load_checkpoint(
            resume, ddpm=ddpm, ema=ema, optim=optim, scaler=scaler, device=cfg.device
        )
        print(f"resumed from {resume} at epoch {start_epoch}")

    loader = mnist_dataloader(
        cfg.data_root,
        batch_size=cfg.batch_size,
        train=True,
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
    )

    n_params = sum(p.numel() for p in ddpm.eps_model.parameters())
    print(f"{n_params / 1e6:.2f}M parameters | device {cfg.device} | amp {use_amp}")

    last_batch = torch.zeros(1, MNIST_CHANNELS, cfg.image_size, cfg.image_size)

    for epoch in range(start_epoch, cfg.n_epoch):
        ddpm.train()
        loss_ema: float | None = None
        pbar = tqdm(loader, desc=f"epoch {epoch}")

        for x, _ in pbar:
            x = x.to(cfg.device, non_blocking=True)
            last_batch = x

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = ddpm(x)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                # Unscale first, or the clip threshold is applied to scaled grads.
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(ddpm.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()
            ema.update(ddpm.eps_model)

            value = loss.item()
            loss_ema = value if loss_ema is None else 0.9 * loss_ema + 0.1 * value
            pbar.set_postfix(loss=f"{loss_ema:.4f}")

        if cfg.sample_every > 0 and (epoch + 1) % cfg.sample_every == 0:
            save_samples(ddpm, ema, last_batch, cfg, epoch)

        save_checkpoint(
            cfg.ckpt_dir / "last.pt",
            epoch=epoch,
            ddpm=ddpm,
            ema=ema,
            optim=optim,
            scaler=scaler,
            cfg=cfg,
        )

    # Ship the EMA weights: they are what the sample grids were drawn from.
    ddpm.eps_model.load_state_dict(ema.module.state_dict())
    return ddpm


if __name__ == "__main__":
    train_mnist()
