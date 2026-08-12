"""MNIST training loop for TinyDiffusion."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

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
from tinydiffusion.utils.seed import seed_everything


@dataclass(slots=True)
class TrainConfig:
    """Hyperparameters for an MNIST training run.

    Defaults target a single consumer GPU and reach recognisable digits within
    a handful of epochs. Field names mirror the constructor arguments of
    :class:`~tinydiffusion.models.unet.UNet` and :class:`DDPM` so the wiring
    below stays a rename-free pass-through.
    """

    # data
    data_root: Path = Path("data")
    image_size: int = 32
    batch_size: int = 128
    num_workers: int = 4

    # model
    base_channels: int = 64
    channel_mult: tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = (16,)
    dropout: float = 0.1

    # diffusion
    num_timesteps: int = 1000
    schedule: Literal["cosine", "linear"] = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # optimisation
    num_epochs: int = 30
    lr: float = 2e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    ema_warmup: int = 2000

    # bookkeeping
    seed: int = 0
    amp: bool = True
    sample_every: int = 1
    num_samples: int = 16
    sample_steps: int = 50
    out_dir: Path = Path("contents")
    ckpt_dir: Path = Path("checkpoints")
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self) -> None:
        """Reject configurations that would only fail an epoch into the run.

        Raises:
            ValueError: if the schedule is unknown or the sampling step count
                cannot index the training schedule.
        """
        if self.schedule not in ("cosine", "linear"):
            raise ValueError(f"unknown schedule {self.schedule!r}, expected 'cosine' or 'linear'")
        if not 1 <= self.sample_steps <= self.num_timesteps:
            raise ValueError(
                f"sample_steps must lie in [1, {self.num_timesteps}], got {self.sample_steps}"
            )
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")


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
        base_channels=cfg.base_channels,
        channel_mult=cfg.channel_mult,
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=cfg.attn_resolutions,
        dropout=cfg.dropout,
        image_size=cfg.image_size,
    )
    if cfg.schedule == "cosine":
        betas = cosine_beta_schedule(cfg.num_timesteps)
    else:
        betas = linear_beta_schedule(cfg.beta_start, cfg.beta_end, cfg.num_timesteps)

    return DDPM(eps_model=net, betas=betas, num_timesteps=cfg.num_timesteps)


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
        cfg.num_samples,
        shape,
        cfg.device,
        num_steps=cfg.sample_steps,
        eta=0.0,
        model=ema.module,
    )
    reference = real[: cfg.num_samples].to(cfg.device)
    grid = torch.cat([denormalize(fake), denormalize(reference)], dim=0)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    save_image(grid, cfg.out_dir / f"sample_{epoch:04d}.png", nrow=min(8, cfg.num_samples))


def train_mnist(cfg: TrainConfig | None = None, resume: Path | None = None) -> DDPM:
    """Train a DDPM on MNIST.

    Args:
        cfg: run configuration. Defaults are used when omitted.
        resume: checkpoint to continue from, or None to start fresh.

    Returns:
        The trained model, with EMA weights already swapped in.
    """
    cfg = cfg or TrainConfig()
    seed_everything(cfg.seed)

    device_type = torch.device(cfg.device).type
    use_amp = cfg.amp and device_type == "cuda"
    if device_type == "cuda":
        # Input shapes are fixed for the whole run, so autotuning pays off once.
        torch.backends.cudnn.benchmark = True

    ddpm = build_model(cfg).to(cfg.device)
    ema = EMA(ddpm.eps_model, decay=cfg.ema_decay, warmup=cfg.ema_warmup)

    optim = torch.optim.Adam(ddpm.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

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

    # Kept on the CPU so the sample grid does not pin a training batch in VRAM.
    reference: torch.Tensor | None = None

    for epoch in range(start_epoch, cfg.num_epochs):
        ddpm.train()
        loss_ema: float | None = None

        with tqdm(loader, desc=f"epoch {epoch}") as pbar:
            for x, _ in pbar:
                x = x.to(cfg.device, non_blocking=True)
                if reference is None:
                    reference = x[: cfg.num_samples].detach().cpu()

                optim.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type, enabled=use_amp):
                    loss = ddpm(x)

                scaler.scale(loss).backward()
                if cfg.grad_clip > 0:
                    # Unscale first, or the clip threshold is applied to scaled grads.
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(ddpm.parameters(), cfg.grad_clip)

                scale_before = scaler.get_scale()
                scaler.step(optim)
                scaler.update()
                # A shrinking scale means inf/NaN grads and a skipped optimiser
                # step; folding the unchanged weights in would still burn a step
                # of the EMA warmup.
                if scaler.get_scale() >= scale_before:
                    ema.update(ddpm.eps_model)

                value = loss.item()
                loss_ema = value if loss_ema is None else 0.9 * loss_ema + 0.1 * value
                pbar.set_postfix(loss=f"{loss_ema:.4f}")

        if cfg.sample_every > 0 and (epoch + 1) % cfg.sample_every == 0 and reference is not None:
            save_samples(ddpm, ema, reference, cfg, epoch)

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
