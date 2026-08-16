"""MNIST training loop for TinyDiffusion."""

import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torchvision.utils import save_image
from tqdm import tqdm

from tinydiffusion.data.datasets import denormalize, image_dataloader
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
from tinydiffusion.training.validation import validation_loss
from tinydiffusion.utils.device import describe_device, enable_tf32, resolve_device
from tinydiffusion.utils.seed import seed_everything
from tinydiffusion.utils.tracking import RunLogger, timestep_quartile_losses

__all__ = [
    "ARCHITECTURE_FIELDS",
    "BEST_CHECKPOINT",
    "INTERRUPTED_CHECKPOINT",
    "LAST_CHECKPOINT",
    "QUARTILE_EVERY",
    "TrainConfig",
    "build_model",
    "check_resume_compatible",
    "epoch_seed",
    "load_checkpoint",
    "read_checkpoint",
    "restore_checkpoint",
    "save_checkpoint",
    "save_samples",
    "train_mnist",
    "validation_batches",
]

LAST_CHECKPOINT = "last.pt"
"""Rewritten after every completed epoch. What ``--resume`` normally wants."""

BEST_CHECKPOINT = "best.pt"
"""The epoch with the lowest held-out loss so far. Written when ``keep_best``."""

INTERRUPTED_CHECKPOINT = "interrupted.pt"
"""Where a cancelled run saves.

Deliberately not ``last.pt``: a Ctrl+C lands mid-epoch, so its weights are
worse than the ones the previous epoch left behind. Writing them over
``last.pt`` — under the *previous* epoch's number, which is what makes the
resume replay correct — would quietly replace a good checkpoint with a worse
one bearing the same label.
"""


ARCHITECTURE_FIELDS = (
    "dataset",
    "image_size",
    "base_channels",
    "channel_mult",
    "num_res_blocks",
    "attn_resolutions",
    "num_classes",
    "num_timesteps",
    "schedule",
    "beta_start",
    "beta_end",
    "predict",
    "variance",
    "objective",
)
"""Config fields a checkpoint's weights are tied to.

The first seven decide the shape of every tensor in the state dict — ``dataset``
by way of its channel count, which is the U-Net's input and output width — and
the rest
decide the schedule buffers and what the network's output means. Neither kind
survives being changed under a ``--resume``, and only the first kind fails
loudly on its own.
"""


QUARTILE_EVERY = 8
"""Batches between timestep-quartile samples.

Slicing the loss by timestep costs a device sync per quartile, which is real
money on a GPU when the loop is otherwise asynchronous. The quartiles are only
ever read as an epoch mean, and every batch draws its timesteps independently,
so sampling one batch in eight measures the same thing for an eighth of the
overhead.
"""


def _warmup_lr(step: int, warmup: int) -> float:
    """Linear LR ramp-in factor, for LambdaLR.

    Stepped per optimiser step, not per epoch.

    Args:
        step: optimiser steps completed.
        warmup: steps to ramp over. 0 disables the ramp.

    Returns:
        A multiplier on ``cfg.lr`` in [0, 1].
    """
    if warmup <= 0:
        return 1.0
    return min(step, warmup) / warmup


def epoch_seed(seed: int, epoch: int) -> int:
    """Seed for one epoch's shuffle order.

    A function of ``(seed, epoch)`` alone, deliberately: seeding the loader
    once at startup makes the order depend on how many epochs have already run
    in *this process*, so a run resumed at epoch 5 replays the ordering a fresh
    run used for epoch 0, and every later epoch follows suit. Deriving it here
    means epoch 5 draws epoch 5's batches whether it was reached by resuming or
    by training straight through.

    Args:
        seed: the run's seed.
        epoch: zero-based epoch index.

    Returns:
        A seed inside the range ``torch.Generator.manual_seed`` accepts.
    """
    # An odd multiplier so consecutive epochs land far apart in the stream,
    # masked to 63 bits because manual_seed rejects anything wider.
    return (seed * 1_000_003 + epoch) & ((1 << 63) - 1)


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
    channels = cfg.dataset_spec().channels
    net = UNet(
        in_channels=channels,
        out_channels=channels * (2 if var_type.is_learned else 1),
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
    sched: LRScheduler | None = None,
    best_val: float | None = None,
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
        sched: LR schedule, or None if the run has none. Its step count is what
            the warmup ramp is a function of, so a resume that dropped it would
            replay the ramp from zero on already-trained weights.
        best_val: lowest held-out loss seen so far, or None if the run is not
            validating. Stored so a ``--resume`` does not restart the
            comparison and overwrite a better ``best.pt``.
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
            "sched": sched.state_dict() if sched is not None else None,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
            "best_val": best_val,
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
    sched: LRScheduler | None = None,
    device: str = "cpu",
) -> int:
    """Read and restore a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: checkpoint file.
        diffusion: model to load weights into.
        ema: EMA wrapper to load shadow weights into.
        optim: optimiser to restore, or None to skip (e.g. sampling only).
        scaler: AMP scaler to restore, or None to skip.
        sched: LR schedule to restore, or None to skip.
        device: device to map tensors onto.

    Returns:
        The epoch index to resume from.
    """
    ckpt = read_checkpoint(path, device=device)
    return restore_checkpoint(
        ckpt, diffusion=diffusion, ema=ema, optim=optim, scaler=scaler, sched=sched
    )


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
    sched: LRScheduler | None = None,
) -> int:
    """Apply an already-loaded checkpoint to the training objects.

    Args:
        ckpt: mapping returned by :func:`read_checkpoint`.
        diffusion: model to load weights into.
        ema: EMA wrapper to load shadow weights into.
        optim: optimiser to restore, or None to skip (e.g. sampling only).
        scaler: AMP scaler to restore, or None to skip.
        sched: LR schedule to restore, or None to skip. Checkpoints written
            before schedules existed carry no entry, and leave it at step 0.

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
    if sched is not None and ckpt.get("sched") is not None:
        sched.load_state_dict(ckpt["sched"])
    return int(ckpt["epoch"]) + 1


def check_resume_compatible(
    ckpt: dict[str, Any], cfg: TrainConfig, *, path: Path | None = None
) -> None:
    """Refuse a ``--resume`` whose weights do not fit the model being built.

    ``train --config X --resume Y`` builds the network from `X` and then loads
    `Y` into it. When the two disagree the failure is a raw ``load_state_dict``
    size-mismatch dump listing every tensor, which says nothing about which
    setting was changed — and for a differing schedule or parameterisation
    there is no failure at all, just a run that quietly optimises something
    other than what the checkpoint was trained on.

    Args:
        ckpt: mapping returned by :func:`read_checkpoint`.
        cfg: the config the model was built from.
        path: the checkpoint's path, for the message.

    Raises:
        ValueError: if any of :data:`ARCHITECTURE_FIELDS` differs.
    """
    stored = ckpt.get("config")
    if stored is None:
        # Predates config provenance. Nothing to compare, so let load_state_dict
        # have its say rather than refusing a checkpoint that may well fit.
        return

    # Round-tripped through the config so lists become tuples and strings
    # become paths; comparing the raw dict would flag `[1, 2]` against `(1, 2)`.
    reference = TrainConfig.from_mapping({**stored, "device": cfg.device})
    changed = [
        (name, getattr(reference, name), getattr(cfg, name))
        for name in ARCHITECTURE_FIELDS
        if getattr(reference, name) != getattr(cfg, name)
    ]
    if not changed:
        return

    detail = "\n".join(
        f"  {name}: checkpoint {was!r}, config {now!r}" for name, was, now in changed
    )
    where = path if path is not None else "the checkpoint"
    raise ValueError(
        f"{where} was trained with a different model, so it cannot resume into this "
        f"config:\n{detail}\nmatch the config to the checkpoint, or start a fresh run "
        f"without --resume"
    )


def validation_batches(cfg: TrainConfig) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialise the held-out slice the run is scored on each epoch.

    Read once and kept in host memory rather than reloaded per epoch: the slice
    is small, and it has to be *the same images every time* for the epoch-to-
    epoch comparison — and so for ``best.pt`` — to mean anything.

    Args:
        cfg: run configuration. ``val_batches`` bounds the slice; 0 takes the
            whole test split.

    Returns:
        ``(images, labels)`` pairs on the CPU, empty if ``val_every`` is off.
    """
    if cfg.val_every <= 0:
        return []

    loader = image_dataloader(
        cfg.dataset_spec(),
        cfg.data_root,
        batch_size=cfg.batch_size,
        train=False,
        image_size=cfg.image_size,
        # Read once, so worker processes would cost more to spawn than they save.
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, (x, y) in enumerate(loader):
        if cfg.val_batches and index >= cfg.val_batches:
            break
        batches.append((x, y))
    return batches


def _snapshot_epoch(ckpt_dir: Path, source: Path, *, epoch: int, keep: int) -> None:
    """Keep a numbered copy of this epoch's checkpoint, pruning old ones.

    A copy of the file just written rather than a second :func:`save_checkpoint`
    call: the bytes are identical by construction, and serialising a U-Net
    twice per epoch is not free.

    Args:
        ckpt_dir: directory holding the run's checkpoints.
        source: the checkpoint just written, copied under an epoch name.
        epoch: zero-based epoch index. The filename is one-based.
        keep: how many snapshots to retain, newest first.
    """
    if keep <= 0 or not source.is_file():
        return
    shutil.copy2(source, ckpt_dir / f"epoch_{epoch + 1:04d}.pt")
    # Zero-padded, so lexical order is epoch order.
    snapshots = sorted(ckpt_dir.glob("epoch_*.pt"))
    for stale in snapshots[: max(len(snapshots) - keep, 0)]:
        stale.unlink(missing_ok=True)


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
    shape = (cfg.dataset_spec().channels, cfg.image_size, cfg.image_size)
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
    sched: LRScheduler | None = None,
    best_val: float | None = None,
) -> None:
    """Checkpoint a cancelled run and tell the user how to pick it up again.

    Written to :data:`INTERRUPTED_CHECKPOINT`, never over ``last.pt``; see that
    constant for why.

    Args:
        cfg: run configuration.
        epoch: index of the last fully completed epoch, or -1 if none.
        diffusion: the diffusion model.
        ema: exponential moving average of the network weights.
        optim: optimiser whose moments should be preserved.
        scaler: AMP gradient scaler.
        sched: LR schedule to restore, or None to skip.
        best_val: lowest held-out loss seen so far, carried through so a resume
            keeps comparing against it.
    """
    path = cfg.ckpt_dir / INTERRUPTED_CHECKPOINT
    save_checkpoint(
        path,
        epoch=epoch,
        diffusion=diffusion,
        ema=ema,
        optim=optim,
        scaler=scaler,
        sched=sched,
        cfg=cfg,
        best_val=best_val,
    )
    done = max(epoch + 1, 0)
    print(f"saved {path} ({_epochs(done)} complete, plus a partial epoch)")
    print(f"resume with: tinydiffusion train --resume {path}")


def train_mnist(cfg: TrainConfig | None = None, resume: Path | None = None) -> Diffusion:
    """Train a diffusion model on MNIST.

    Which process is trained follows from the config; see :func:`build_model`.

    Ctrl+C does not tear the run down: it is caught at the next batch boundary
    and turned into a confirmation prompt, with the option to checkpoint first
    so ``--resume`` can pick the run up later. That save goes to
    :data:`INTERRUPTED_CHECKPOINT`, leaving ``last.pt`` as the newest *complete*
    epoch.

    Each epoch ends by scoring a fixed held-out slice, and the best-scoring
    epoch is kept as :data:`BEST_CHECKPOINT`. Diffusion runs do not improve
    monotonically, so the last epoch is not reliably the one worth sampling.

    Args:
        cfg: run configuration. Defaults are used when omitted.
        resume: checkpoint to continue from, or None to start fresh.

    Returns:
        The trained model, with EMA weights already swapped in. A run cancelled
        part way through returns the model as it stood at that point.

    Raises:
        ValueError: if `resume` names a checkpoint trained with a different
            model than `cfg` describes.
    """
    cfg = cfg or TrainConfig()
    cfg = replace(cfg, device=resolve_device(cfg.device))
    spec = cfg.dataset_spec()
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    device_type = torch.device(cfg.device).type
    use_amp = cfg.amp and device_type == "cuda"
    if device_type == "cuda":
        # Input shapes are fixed for the whole run, so autotuning pays off once
        # — but the autotuner picks whichever kernel wins on the day, so it is
        # exactly what `deterministic` is asking us not to do. seed_everything
        # has already cleared the flag; setting it back here unconditionally
        # would quietly undo the setting that was just applied.
        torch.backends.cudnn.benchmark = not cfg.deterministic
        enable_tf32()

    diffusion = build_model(cfg).to(cfg.device)
    ema = EMA(diffusion.net, decay=cfg.ema_decay, warmup=cfg.ema_warmup)

    optim = torch.optim.Adam(diffusion.parameters(), lr=cfg.lr)
    # Stepped once per *applied* optimiser step below, so the ramp counts real
    # updates rather than batches AMP threw away.
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: _warmup_lr(step, cfg.lr_warmup)
    )
    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

    start_epoch = 0
    best_val: float | None = None
    if resume is not None:
        ckpt = read_checkpoint(resume, device=cfg.device)
        # Before load_state_dict, so a mismatch names the setting that changed
        # instead of listing every tensor that no longer fits.
        check_resume_compatible(ckpt, cfg, path=resume)
        start_epoch = restore_checkpoint(
            ckpt, diffusion=diffusion, ema=ema, optim=optim, scaler=scaler, sched=sched
        )
        best_val = ckpt.get("best_val")
        print(f"resumed from {resume}, {_epochs(start_epoch)} already done")

    # Re-seeded per epoch below, so the batch order is a function of the epoch
    # index rather than of how many epochs this process has already run.
    loader_rng = torch.Generator()
    loader = image_dataloader(
        spec,
        cfg.data_root,
        batch_size=cfg.batch_size,
        train=True,
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
        # Only the training split is augmented, and only where the spec says a
        # flip preserves the label.
        augment=True,
        generator=loader_rng,
    )

    held_out = validation_batches(cfg)

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
    scoring = (
        f"{sum(x.shape[0] for x, _ in held_out)} held-out images every {_epochs(cfg.val_every)}"
        if held_out
        else "no validation"
    )
    print(
        f"{n_params / 1e6:.2f}M parameters | {spec.name} {cfg.image_size}px x{spec.channels} | "
        f"device {describe_device(cfg.device)} | amp {use_amp} | {conditioning} | {plan} | "
        f"{len(loader)} steps/epoch | {scoring}"
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
        spec.channels,
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
            loader_rng.manual_seed(epoch_seed(cfg.seed, epoch))
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
                        sched.step()

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
                                best_val=best_val,
                                sched=sched,
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
            new_best: float | None = None
            if held_out and not cancelled and (epoch + 1) % cfg.val_every == 0:
                # The EMA weights, because they are what the sample grids and
                # every downstream command draw from. Scoring the live weights
                # would pick a "best" epoch nobody ever samples.
                val = validation_loss(
                    diffusion,
                    held_out,
                    model=ema.module,
                    num_classes=cfg.num_classes,
                    device=cfg.device,
                    num_steps=cfg.val_steps,
                    seed=cfg.seed,
                )
                logger.set(**{"val/loss": val})
                if best_val is None or val < best_val:
                    best_val = new_best = val
                logger.set(**{"val/best_loss": best_val})

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

            last = cfg.ckpt_dir / LAST_CHECKPOINT
            save_checkpoint(
                last,
                epoch=epoch,
                diffusion=diffusion,
                ema=ema,
                optim=optim,
                scaler=scaler,
                sched=sched,
                cfg=cfg,
                best_val=best_val,
            )
            _snapshot_epoch(cfg.ckpt_dir, last, epoch=epoch, keep=cfg.keep_last)

            if new_best is not None and cfg.keep_best:
                best = cfg.ckpt_dir / BEST_CHECKPOINT
                # Copied rather than re-serialised: identical bytes, half the I/O.
                shutil.copy2(last, best)
                print(f"val/loss {new_best:.5f} is a new best; wrote {best}")

    # Ship the EMA weights: they are what the sample grids were drawn from.
    diffusion.net.load_state_dict(ema.module.state_dict())
    return diffusion


if __name__ == "__main__":
    train_mnist()
