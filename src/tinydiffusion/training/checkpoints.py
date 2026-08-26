"""Reading and writing training checkpoints.

A checkpoint holds more than weights: the optimiser moments, the EMA shadow
weights, the AMP scaler and the LR schedule's step count are all training state,
and a resume that dropped any of them would continue from something other than
where it left off. The config travels along too, which is what lets
:func:`check_resume_compatible` refuse a ``--resume`` before ``load_state_dict``
gets a chance to fail obscurely.

Everything here is loop-free, so sampling, evaluation and the server can read a
checkpoint without importing :mod:`tinydiffusion.training.train`.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import LRScheduler

from tinydiffusion.diffusion.gaussian_diffusion import Diffusion
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.ema import EMA

__all__ = [
    "ARCHITECTURE_FIELDS",
    "BEST_CHECKPOINT",
    "INTERRUPTED_CHECKPOINT",
    "LAST_CHECKPOINT",
    "check_resume_compatible",
    "config_from_checkpoint",
    "load_checkpoint",
    "read_checkpoint",
    "restore_checkpoint",
    "restore_rng_state",
    "rng_state",
    "save_checkpoint",
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
    "folder_channels",
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
    "zero_snr",
    "predict",
    "variance",
    "objective",
)
"""Config fields a checkpoint's weights are tied to.

The first eight decide the shape of every tensor in the state dict — ``dataset``
by way of its channel count, which is the U-Net's input and output width, and
``folder_channels``, which is where that count comes from when the dataset is a
folder — and the rest
decide the schedule buffers and what the network's output means. ``zero_snr``
belongs to the second kind: it rescales the betas the whole schedule is derived
from, so flipping it under a ``--resume`` continues the run against a different
forward process while every tensor still fits. Neither kind survives being
changed under a ``--resume``, and only the first kind fails loudly on its own.
"""


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
    model_state: dict[str, torch.Tensor] | None = None,
) -> None:
    """Write a resumable checkpoint.

    Saving only ``model.state_dict()`` makes a run unresumable: the optimiser
    moments and the EMA shadow weights are both training state. So is the
    global RNG, which travels along under ``"rng"``; see :func:`rng_state`.

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
        model_state: the weights to write, when ``diffusion.net``'s own are not
            the ones to keep. That is the
            :attr:`~tinydiffusion.training.config.TrainConfig.full_fp16` case,
            where the network holds a float16 rounding of the float32 master
            copy the optimiser actually steps; see
            :func:`~tinydiffusion.utils.fp16.master_params_to_state_dict`.
            Defaults to ``diffusion.net.state_dict()``, so every checkpoint
            this project writes is a float32 one however it was trained.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": diffusion.net.state_dict() if model_state is None else model_state,
            "ema": ema.module.state_dict(),
            "ema_step": ema.step,
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "sched": sched.state_dict() if sched is not None else None,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
            "best_val": best_val,
            "rng": rng_state(),
        },
        tmp,
    )
    # Rename last so an interrupted save cannot corrupt a good checkpoint.
    tmp.replace(path)


def rng_state() -> dict[str, Any]:
    """Snapshot the global RNG, so a resume continues the same random stream.

    The loader's shuffle order is already a function of the epoch index alone
    (see :func:`~tinydiffusion.training.train.epoch_seed`), but everything else
    a step draws — the diffusion noise, the timesteps, dropout, and the label
    dropout that classifier-free guidance is trained on — comes from the global
    generator, which is seeded once at startup. Without this, epoch 5 of a
    resumed run sees different noise than epoch 5 of a run trained straight
    through, and ``deterministic`` does not make the two agree.

    Returns:
        The CPU generator's state, plus every CUDA device's, under ``"cuda"``.
    """
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(ckpt: dict[str, Any]) -> bool:
    """Put the global RNG back where the checkpoint left it.

    Deliberately not part of :func:`restore_checkpoint`: sampling, evaluation
    and the server all read checkpoints too, and each seeds the generator for
    itself. Reaching into the process-wide RNG is the training loop's business
    alone, so it asks for it by name.

    Note that this is exact only for a checkpoint written at an epoch boundary.
    :data:`INTERRUPTED_CHECKPOINT` is written mid-epoch but resumes from the
    start of that epoch, so its stream is a valid one that no straight run
    would have followed — still reproducible, just not identical.

    Args:
        ckpt: mapping returned by :func:`read_checkpoint`. One written before
            this was stored leaves the RNG untouched.

    Returns:
        Whether a state was found and applied.
    """
    state = ckpt.get("rng")
    if not state:
        return False
    torch.set_rng_state(state["cpu"].cpu().to(torch.uint8))
    cuda = state.get("cuda")
    # A run moved between machines can have a different device count, and
    # set_rng_state_all raises on a mismatch. The CPU state is the one that
    # matters for the data path either way, so a partial restore beats none.
    if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in cuda])
    return True


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
        scaler: AMP scaler to restore, or None to skip. A checkpoint from a run
            that had no scaler enabled leaves it at its own initial scale.
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
    if scaler is not None and ckpt.get("scaler"):
        # Truthiness rather than presence: a run whose scaler was disabled —
        # bf16, CPU, amp off — stores an empty dict, and GradScaler refuses to
        # load one rather than treating it as "nothing to restore". That would
        # make an fp16 run unable to pick up a bf16 run's checkpoint even
        # though its weights fit perfectly, so the scale simply starts fresh.
        scaler.load_state_dict(ckpt["scaler"])
    if sched is not None and ckpt.get("sched") is not None:
        sched.load_state_dict(ckpt["sched"])
    return int(ckpt["epoch"]) + 1


def config_from_checkpoint(path: Path) -> TrainConfig:
    """Recover the config a checkpoint was trained with.

    What ``--resume`` on its own continues from. The alternative — defaulting
    to :class:`TrainConfig` and letting :func:`check_resume_compatible` object
    — makes resuming any run that was not trained on the defaults require the
    original TOML file, which the checkpoint has been carrying all along.

    Args:
        path: checkpoint file.

    Returns:
        The stored configuration, including the paths and the device the run
        used. A caller layering overrides on top should apply them afterwards.

    Raises:
        ValueError: if the checkpoint predates config provenance, and so has
            nothing to rebuild.
    """
    stored = read_checkpoint(path).get("config")
    if stored is None:
        raise ValueError(
            f"{path} stores no config, so --resume cannot infer the run's settings; "
            f"pass --config with the file it was trained from"
        )
    return TrainConfig.from_mapping(stored)


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
