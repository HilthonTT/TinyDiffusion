"""What a run leaves on disk: sample grids, and the checkpoints beside them.

Kept apart from the loop that triggers them because none of it is part of
training -- a run that wrote nothing would train identically -- and because the
grid in particular is worth drawing on its own, after the fact, from a
checkpoint.
"""

import shutil
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torchvision.utils import save_image

from tinydiffusion.data.datasets import denormalize
from tinydiffusion.diffusion.gaussian_diffusion import Diffusion
from tinydiffusion.diffusion.guidance import conditioned
from tinydiffusion.diffusion.samplers import get_sampler
from tinydiffusion.training.checkpoints import INTERRUPTED_CHECKPOINT, save_checkpoint
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.ema import EMA
from tinydiffusion.training.plan import epochs_phrase
from tinydiffusion.utils.fp16 import master_params_to_state_dict

__all__ = [
    "checkpoint_state",
    "save_and_report",
    "save_samples",
    "snapshot_epoch",
]


def snapshot_epoch(ckpt_dir: Path, source: Path, *, epoch: int, keep: int) -> None:
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
) -> Path:
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

    Returns:
        The path written, so a caller watching the run can pick the grid up
        without reconstructing the filename.
    """
    shape = (cfg.dataset_spec().channels, cfg.image_size, cfg.image_size)
    if labels is not None:
        index = torch.arange(cfg.num_samples) % labels.shape[0]
        labels = labels[index].to(cfg.device)

    fake = get_sampler(cfg.sampler)(
        diffusion,
        cfg.num_samples,
        shape,
        cfg.device,
        num_steps=cfg.sample_steps,
        eta=0.0,
        model=conditioned(
            ema.module,
            labels,
            num_classes=cfg.num_classes,
            scale=cfg.guidance,
            rescale=cfg.guidance_rescale,
        ),
        noise=noise,
        spacing=cfg.sample_spacing,
    )
    reference = real[: cfg.num_samples].to(cfg.device)
    grid = torch.cat([denormalize(fake), denormalize(reference)], dim=0)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.out_dir / f"sample_{epoch + 1:04d}.png"
    save_image(grid, path, nrow=min(8, cfg.num_samples))
    return path


def checkpoint_state(
    diffusion: Diffusion, master_params: list[nn.Parameter] | None
) -> dict[str, torch.Tensor] | None:
    """The weights to checkpoint, in float32 whichever precision the run uses.

    Args:
        diffusion: the diffusion model.
        master_params: the run's float32 master copy, or None when the network
            holds its own weights at full precision.

    Returns:
        A float32 state dict when there is a master copy to read, and None to
        leave :func:`~tinydiffusion.training.checkpoints.save_checkpoint` with
        its default of the network's own.
    """
    if master_params is None:
        return None
    return master_params_to_state_dict(diffusion.net, master_params)


def save_and_report(
    cfg: TrainConfig,
    *,
    epoch: int,
    diffusion: Diffusion,
    ema: EMA,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sched: LRScheduler | None = None,
    best_val: float | None = None,
    model_state: dict[str, torch.Tensor] | None = None,
    say: Callable[[str], None] = print,
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
        model_state: float32 weights to write in place of the network's own;
            see :func:`~tinydiffusion.training.checkpoints.save_checkpoint`.
        say: where the two lines go. Defaults to printing them.
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
        model_state=model_state,
    )
    done = max(epoch + 1, 0)
    say(f"saved {path} ({epochs_phrase(done)} complete, plus a partial epoch)")
    say(f"resume with: tinydiffusion train --resume {path}")
