"""MNIST training loop for TinyDiffusion."""

import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm

from tinydiffusion.data.mnist import MNIST_CHANNELS, denormalize, mnist_dataloader
from tinydiffusion.diffusion.ddim import ddim_sample
from tinydiffusion.diffusion.ddpm import DDPM
from tinydiffusion.diffusion.gaussian_diffusion import (
    Diffusion,
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from tinydiffusion.diffusion.guidance import Conditioned, conditioned, drop_labels
from tinydiffusion.diffusion.schedules import cosine_beta_schedule, linear_beta_schedule
from tinydiffusion.models.unet import UNet
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.ema import EMA
from tinydiffusion.training.interrupt import interrupt_guard
from tinydiffusion.utils.device import describe_device, enable_tf32, resolve_device
from tinydiffusion.utils.seed import seed_everything
from tinydiffusion.utils.tracking import RunLogger, timestep_quartile_losses

__all__ = [
    "QUARTILE_EVERY",
    "TrainConfig",
    "build_model",
    "load_checkpoint",
    "read_checkpoint",
    "restore_checkpoint",
    "save_checkpoint",
    "save_samples",
    "train_mnist",
]


QUARTILE_EVERY = 8
"""Batches between timestep-quartile samples.

Slicing the loss by timestep costs a device sync per quartile, which is real
money on a GPU when the loop is otherwise asynchronous. The quartiles are only
ever read as an epoch mean, and every batch draws its timesteps independently,
so sampling one batch in eight measures the same thing for an eighth of the
overhead.
"""


def _epochs(count: int) -> str:
    """Render an epoch count, pluralised.

    Args:
        count: number of epochs.

    Returns:
        e.g. ``"1 epoch"`` or ``"30 epochs"``.
    """
    return f"{count} epoch" if count == 1 else f"{count} epochs"


def build_model(cfg: TrainConfig) -> Diffusion:
    """Construct the U-Net and wrap it in the diffusion process.

    The parameterisation fields pick the process:
    :class:`~tinydiffusion.diffusion.ddpm.DDPM` for the default
    epsilon/fixed-small/MSE combination it was written for, and
    :class:`~tinydiffusion.diffusion.gaussian_diffusion.GaussianDiffusion` for
    anything else. A learned variance also doubles the network's output
    channels, since it emits the variance parameters alongside the mean.
    ``num_classes`` gives the U-Net a label embedding; the process itself is
    unchanged, since conditioning is carried by the network alone.

    Args:
        cfg: run configuration.

    Returns:
        An untrained diffusion process on the CPU.
    """
    mean_type, var_type, loss_type = cfg.diffusion_types()
    net = UNet(
        in_channels=MNIST_CHANNELS,
        out_channels=MNIST_CHANNELS * (2 if var_type.is_learned else 1),
        base_channels=cfg.base_channels,
        channel_mult=cfg.channel_mult,
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=cfg.attn_resolutions,
        dropout=cfg.dropout,
        image_size=cfg.image_size,
        num_classes=cfg.num_classes,
    )
    if cfg.schedule == "cosine":
        betas = cosine_beta_schedule(cfg.num_timesteps)
    else:
        betas = linear_beta_schedule(cfg.beta_start, cfg.beta_end, cfg.num_timesteps)

    if (mean_type, var_type, loss_type) == (
        ModelMeanType.EPSILON,
        ModelVarType.FIXED_SMALL,
        LossType.MSE,
    ):
        return DDPM(eps_model=net, betas=betas, num_timesteps=cfg.num_timesteps)

    return GaussianDiffusion(
        net,
        betas=betas,
        num_timesteps=cfg.num_timesteps,
        model_mean_type=mean_type,
        model_var_type=var_type,
        loss_type=loss_type,
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    diffusion: Diffusion,
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
        diffusion: the diffusion model.
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
            "model": diffusion.net.state_dict(),
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
    diffusion: Diffusion,
    ema: EMA,
    optim: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: str = "cpu",
) -> int:
    """Read and restore a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: checkpoint file.
        diffusion: model to load weights into.
        ema: EMA wrapper to load shadow weights into.
        optim: optimiser to restore, or None to skip (e.g. sampling only).
        scaler: AMP scaler to restore, or None to skip.
        device: device to map tensors onto.

    Returns:
        The epoch index to resume from.
    """
    ckpt = read_checkpoint(path, device=device)
    return restore_checkpoint(ckpt, diffusion=diffusion, ema=ema, optim=optim, scaler=scaler)


def read_checkpoint(path: Path, *, device: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint file into memory without applying it.

    Callers that need the stored config before they can build the model — the
    sampling entry point, for one — read once through here and then apply the
    result with :func:`restore_checkpoint`.

    Args:
        path: checkpoint file.
        device: device to map tensors onto.

    Returns:
        The raw checkpoint mapping.
    """
    ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    return ckpt


def restore_checkpoint(
    ckpt: dict[str, Any],
    *,
    diffusion: Diffusion,
    ema: EMA,
    optim: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> int:
    """Apply an already-loaded checkpoint to the training objects.

    Args:
        ckpt: mapping returned by :func:`read_checkpoint`.
        diffusion: model to load weights into.
        ema: EMA wrapper to load shadow weights into.
        optim: optimiser to restore, or None to skip (e.g. sampling only).
        scaler: AMP scaler to restore, or None to skip.

    Returns:
        The epoch index to resume from.
    """
    diffusion.net.load_state_dict(ckpt["model"])
    ema.module.load_state_dict(ckpt["ema"])
    ema.step = ckpt.get("ema_step", 0)
    if optim is not None and "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["epoch"]) + 1


@torch.no_grad()
def save_samples(
    diffusion: Diffusion,
    ema: EMA,
    real: torch.Tensor,
    cfg: TrainConfig,
    epoch: int,
    labels: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> None:
    """Render a grid of EMA samples above a strip of real images.

    Putting real data in the same grid makes contrast and stroke weight
    directly comparable, which is what tells you whether the model has
    learned the data distribution or merely something digit-shaped.

    A conditional run generates on the *real* strip's own labels, so the two
    halves line up column by column and the comparison becomes per class: a
    generated 4 sits directly above a real 4.

    Args:
        diffusion: the diffusion model, used for its schedule.
        ema: EMA weights to sample from.
        real: a batch of real images in [-1, 1] to show alongside.
        cfg: run configuration.
        epoch: zero-based epoch index. The filename is one-based, matching the
            progress bar.
        labels: the real strip's class labels, or None when unconditional.
        noise: the starting x_T to redraw from, or None for a fresh draw. Held
            fixed across a run, it makes the grids a flipbook of one set of
            images sharpening rather than an unrelated sample each epoch.
    """
    shape = (MNIST_CHANNELS, cfg.image_size, cfg.image_size)
    if labels is not None:
        # A batch smaller than num_samples leaves the strip short; repeat it so
        # there is a label per generated image either way.
        index = torch.arange(cfg.num_samples) % labels.shape[0]
        labels = labels[index].to(cfg.device)

    fake = ddim_sample(
        diffusion,
        cfg.num_samples,
        shape,
        cfg.device,
        num_steps=cfg.sample_steps,
        eta=0.0,
        model=conditioned(ema.module, labels, num_classes=cfg.num_classes, scale=cfg.guidance),
        noise=noise,
    )
    reference = real[: cfg.num_samples].to(cfg.device)
    grid = torch.cat([denormalize(fake), denormalize(reference)], dim=0)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    save_image(grid, cfg.out_dir / f"sample_{epoch + 1:04d}.png", nrow=min(8, cfg.num_samples))


def _save_and_report(
    cfg: TrainConfig,
    *,
    epoch: int,
    diffusion: Diffusion,
    ema: EMA,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> None:
    """Checkpoint a cancelled run and tell the user how to pick it up again.

    Args:
        cfg: run configuration.
        epoch: index of the last fully completed epoch, or -1 if none.
        diffusion: the diffusion model.
        ema: exponential moving average of the network weights.
        optim: optimiser whose moments should be preserved.
        scaler: AMP gradient scaler.
    """
    path = cfg.ckpt_dir / "last.pt"
    save_checkpoint(
        path, epoch=epoch, diffusion=diffusion, ema=ema, optim=optim, scaler=scaler, cfg=cfg
    )
    done = max(epoch + 1, 0)
    print(f"saved {path} ({_epochs(done)} complete)")
    print(f"resume with: tinydiffusion train --resume {path}")


def train_mnist(cfg: TrainConfig | None = None, resume: Path | None = None) -> Diffusion:
    """Train a diffusion model on MNIST.

    Which process is trained follows from the config; see :func:`build_model`.

    Ctrl+C does not tear the run down: it is caught at the next batch boundary
    and turned into a confirmation prompt, with the option to checkpoint first
    so ``--resume`` can pick the run up later.

    Args:
        cfg: run configuration. Defaults are used when omitted.
        resume: checkpoint to continue from, or None to start fresh.

    Returns:
        The trained model, with EMA weights already swapped in. A run cancelled
        part way through returns the model as it stood at that point.
    """
    cfg = cfg or TrainConfig()
    cfg = replace(cfg, device=resolve_device(cfg.device))
    seed_everything(cfg.seed)

    device_type = torch.device(cfg.device).type
    use_amp = cfg.amp and device_type == "cuda"
    if device_type == "cuda":
        # Input shapes are fixed for the whole run, so autotuning pays off once.
        torch.backends.cudnn.benchmark = True
        enable_tf32()

    diffusion = build_model(cfg).to(cfg.device)
    ema = EMA(diffusion.net, decay=cfg.ema_decay, warmup=cfg.ema_warmup)

    optim = torch.optim.Adam(diffusion.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    start_epoch = 0
    if resume is not None:
        start_epoch = load_checkpoint(
            resume, diffusion=diffusion, ema=ema, optim=optim, scaler=scaler, device=cfg.device
        )
        print(f"resumed from {resume}, {_epochs(start_epoch)} already done")

    loader = mnist_dataloader(
        cfg.data_root,
        batch_size=cfg.batch_size,
        train=True,
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
    )

    n_params = sum(p.numel() for p in diffusion.net.parameters())
    remaining = cfg.num_epochs - start_epoch
    if start_epoch == 0:
        plan = _epochs(cfg.num_epochs)
    elif remaining > 0:
        plan = f"epochs {start_epoch + 1}-{cfg.num_epochs} ({remaining} to go)"
    else:
        # A checkpoint past num_epochs would otherwise render as "epochs 4-2".
        plan = f"nothing to run (checkpoint is at {_epochs(start_epoch)})"
    conditioning = (
        f"{cfg.num_classes} classes, {cfg.class_dropout:g} label dropout"
        if cfg.num_classes is not None
        else "unconditional"
    )
    print(
        f"{n_params / 1e6:.2f}M parameters | device {describe_device(cfg.device)} | "
        f"amp {use_amp} | {conditioning} | {plan} | {len(loader)} steps/epoch"
    )
    if remaining <= 0:
        print(f"nothing to do: the checkpoint already covers all {_epochs(cfg.num_epochs)}")

    # Kept on the CPU so the sample grid does not pin a training batch in VRAM.
    reference: torch.Tensor | None = None
    reference_labels: torch.Tensor | None = None

    # Fixed latents, so each epoch's grid redraws the same x_T and the sequence of
    # PNGs reads as one set of digits sharpening rather than a fresh draw each time.
    # Seeded off cfg.seed rather than the live RNG, so a --resume continues the
    # same grid instead of starting a new one.
    sample_noise = torch.randn(
        cfg.num_samples,
        MNIST_CHANNELS,
        cfg.image_size,
        cfg.image_size,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    cancelled = False

    logger = RunLogger.for_run(
        cfg.log_dir,
        console=cfg.log_console,
        jsonl=cfg.log_jsonl,
        tensorboard=cfg.tensorboard,
    )

    with logger, interrupt_guard() as interrupts:
        for epoch in range(start_epoch, cfg.num_epochs):
            diffusion.train()
            loss_ema: float | None = None
            epoch_start = time.perf_counter()
            images = 0

            with tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.num_epochs}") as pbar:
                for batch, (x, y) in enumerate(pbar):
                    x = x.to(cfg.device, non_blocking=True)
                    if reference is None:
                        reference = x[: cfg.num_samples].detach().cpu()
                        if cfg.num_classes is not None:
                            reference_labels = y[: cfg.num_samples].detach().cpu()

                    model: nn.Module | None = None
                    if cfg.num_classes is not None:
                        # Dropping a fraction of the labels to the null token is
                        # the only thing training does differently: it is what
                        # teaches the one network the unconditional prediction
                        # that guidance extrapolates away from at sample time.
                        labels = drop_labels(
                            y.to(cfg.device, non_blocking=True),
                            cfg.num_classes,
                            cfg.class_dropout,
                        )
                        model = Conditioned(diffusion.net, labels)

                    optim.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type, enabled=use_amp):
                        terms = diffusion.loss_terms(x, model=model)
                    loss = terms.loss

                    # torch ships `Tensor.backward` unannotated, so now that the
                    # loss is a real Tensor rather than the Any that
                    # `nn.Module.__call__` returns, strict mypy calls this an
                    # untyped call.
                    scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                    grad_norm: float | None = None
                    if cfg.grad_clip > 0:
                        # Unscale first, or the clip threshold is applied to scaled grads.
                        scaler.unscale_(optim)
                        # The pre-clip norm comes back for free; it is the first
                        # thing to look at when a loss curve goes flat or spikes.
                        grad_norm = float(
                            nn.utils.clip_grad_norm_(diffusion.parameters(), cfg.grad_clip)
                        )

                    scale_before = scaler.get_scale()
                    scaler.step(optim)
                    scaler.update()
                    # A shrinking scale means inf/NaN grads and a skipped optimiser
                    # step; folding the unchanged weights in would still burn a step
                    # of the EMA warmup.
                    stepped = scaler.get_scale() >= scale_before
                    if stepped:
                        ema.update(diffusion.net)

                    value = loss.item()
                    loss_ema = value if loss_ema is None else 0.9 * loss_ema + 0.1 * value
                    pbar.set_postfix(loss=f"{loss_ema:.4f}")

                    images += x.shape[0]
                    batch_metrics = {"train/loss": value, "train/skipped_step": float(not stepped)}
                    if grad_norm is not None:
                        batch_metrics["train/grad_norm"] = grad_norm
                    if batch % QUARTILE_EVERY == 0:
                        batch_metrics |= {
                            f"train/{name}": quartile_loss
                            for name, quartile_loss in timestep_quartile_losses(
                                terms.per_sample.float(), terms.timesteps, cfg.num_timesteps
                            ).items()
                        }
                    logger.accumulate(**batch_metrics)

                    if interrupts.requested:
                        # Batch boundary: model, optimiser and EMA all agree, so
                        # a checkpoint written here resumes cleanly.
                        with tqdm.external_write_mode():
                            choice = interrupts.resolve()
                        if not choice.stop:
                            continue
                        if choice.save:
                            # The last *completed* epoch is the one before this
                            # partial one, so resuming replays it in full.
                            _save_and_report(
                                cfg,
                                epoch=epoch - 1,
                                diffusion=diffusion,
                                ema=ema,
                                optim=optim,
                                scaler=scaler,
                            )
                        else:
                            print("cancelled without saving")
                        cancelled = True
                        break

            elapsed = time.perf_counter() - epoch_start
            logger.set(
                **{
                    "train/lr": float(optim.param_groups[0]["lr"]),
                    "train/ema_decay": ema.current_decay,
                    "train/amp_scale": float(scaler.get_scale()) if use_amp else 1.0,
                    "time/epoch_seconds": elapsed,
                    "time/images_per_second": images / elapsed if elapsed > 0 else 0.0,
                }
            )
            # Flushed even for the partial epoch a Ctrl+C ends on: those batches
            # were still work, and the record explains where the run stopped.
            logger.flush(step=epoch)

            if cancelled:
                break

            if (
                cfg.sample_every > 0
                and (epoch + 1) % cfg.sample_every == 0
                and reference is not None
            ):
                save_samples(
                    diffusion,
                    ema,
                    reference,
                    cfg,
                    epoch,
                    labels=reference_labels,
                    noise=sample_noise,
                )

            save_checkpoint(
                cfg.ckpt_dir / "last.pt",
                epoch=epoch,
                diffusion=diffusion,
                ema=ema,
                optim=optim,
                scaler=scaler,
                cfg=cfg,
            )

    # Ship the EMA weights: they are what the sample grids were drawn from.
    diffusion.net.load_state_dict(ema.module.state_dict())
    return diffusion


if __name__ == "__main__":
    train_mnist()
