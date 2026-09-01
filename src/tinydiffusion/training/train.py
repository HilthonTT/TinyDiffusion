"""The training loop for TinyDiffusion.

What a run trains on follows from its config's ``dataset``; see
:mod:`tinydiffusion.data.datasets` for the registry it names.

:func:`train` is the assembly: it resolves the run, builds what the run needs
and drives the epochs. The pieces themselves live next door --
:mod:`~tinydiffusion.training.setup` makes the decisions,
:mod:`~tinydiffusion.training.plan` announces them,
:mod:`~tinydiffusion.training.batches` fixes the batches a run reuses,
:mod:`~tinydiffusion.training.loop` runs a batch and an epoch,
:mod:`~tinydiffusion.training.artifacts` writes the grids and checkpoints, and
:mod:`~tinydiffusion.training.reporting` carries the numbers out.
"""

from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch

from tinydiffusion.data.datasets import image_dataloader, set_loader_epoch
from tinydiffusion.diffusion.gaussian_diffusion import Diffusion
from tinydiffusion.models.unet import UNet
from tinydiffusion.training.artifacts import save_samples
from tinydiffusion.training.batches import epoch_seed, reference_batch, validation_batches
from tinydiffusion.training.checkpoints import check_resume_compatible, read_checkpoint
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.distributed import barrier
from tinydiffusion.training.distributed import (
    setup as distributed_setup,
)
from tinydiffusion.training.distributed import (
    shutdown as distributed_shutdown,
)
from tinydiffusion.training.interrupt import interrupt_guard
from tinydiffusion.training.loop import Run, finish_epoch, log_epoch, run_epoch, score_epoch
from tinydiffusion.training.lr import lr_factor
from tinydiffusion.training.observer import TrainObserver
from tinydiffusion.training.plan import announce_plan
from tinydiffusion.training.reporting import DRAIN_EVERY, QUARTILE_EVERY, ObserverBackend, silent
from tinydiffusion.training.setup import (
    build_network,
    configure_backends,
    parameter_sets,
    resolve_precision,
    restore_run,
)
from tinydiffusion.utils.fp16 import model_params_to_master_params
from tinydiffusion.utils.seed import seed_everything
from tinydiffusion.utils.tracking import LoggerBackend, RunLogger

__all__ = [
    "DRAIN_EVERY",
    "QUARTILE_EVERY",
    "epoch_seed",
    "reference_batch",
    "save_samples",
    "train",
    "validation_batches",
]


def train(
    cfg: TrainConfig | None = None,
    resume: Path | None = None,
    observer: TrainObserver | None = None,
) -> Diffusion:
    """Train a diffusion model on the dataset the config names.

    Which process is trained follows from the config too; see
    :func:`~tinydiffusion.training.model.build_model`.

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
        observer: something watching the run from outside the loop; see
            :mod:`tinydiffusion.training.observer`. Passing one redirects every
            line this would have printed, replaces the tqdm bar with
            :meth:`~tinydiffusion.training.observer.TrainObserver.on_batch`,
            and lets the watcher stop the run without the Ctrl+C prompt — which
            has nobody to answer it when the terminal belongs to a display.
            None leaves all three exactly as they were.

    Returns:
        The trained model, with EMA weights already swapped in. A run cancelled
        part way through returns the model as it stood at that point.

    Raises:
        ValueError: if `resume` names a checkpoint trained with a different
            model than `cfg` describes.
    """
    cfg = cfg or TrainConfig()
    say: Callable[[str], None] = observer.on_message if observer is not None else print
    group, device = distributed_setup(cfg.device, say=say)
    if observer is None and not group.is_main:
        say = silent
    cfg = replace(cfg, device=device)
    spec = cfg.dataset_spec()
    seed_everything(cfg.seed, deterministic=cfg.deterministic)

    precision = resolve_precision(cfg, say)
    configure_backends(cfg, precision.device_type)

    diffusion, ema, train_net, ddp = build_network(cfg, group, precision, say)
    model_params, master_params, step_params = parameter_sets(
        diffusion, full_fp16=precision.full_fp16
    )

    optim = torch.optim.AdamW(
        step_params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )

    ckpt: dict[str, Any] | None = None
    if resume is not None:
        ckpt = read_checkpoint(resume, device=cfg.device)
        check_resume_compatible(ckpt, cfg, path=resume)

    loader_rng = torch.Generator()
    loader = image_dataloader(
        spec,
        cfg.data_root,
        batch_size=cfg.batch_size,
        train=True,
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
        augment=True,
        generator=loader_rng,
        num_replicas=group.world_size if group.enabled else None,
        rank=group.rank if group.enabled else None,
        seed=cfg.seed,
    )

    steps_per_epoch = -(-len(loader) // cfg.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: lr_factor(
            step,
            warmup=cfg.lr_warmup,
            total=steps_per_epoch * cfg.num_epochs,
            schedule=cfg.lr_schedule,
        ),
    )
    scaler = precision.grad_scaler()

    start_epoch = 0
    best_val: float | None = None
    if ckpt is not None:
        start_epoch, best_val = restore_run(
            ckpt,
            resume=resume,
            diffusion=diffusion,
            ema=ema,
            optim=optim,
            scaler=scaler,
            sched=sched,
            full_fp16=precision.full_fp16,
            say=say,
        )

    if master_params is not None:
        model_params_to_master_params(model_params, master_params)
        cast(UNet, diffusion.net).convert_to_fp16()

    held_out = validation_batches(cfg)

    announce_plan(
        cfg,
        spec,
        group,
        observer=observer,
        say=say,
        n_params=sum(p.numel() for p in diffusion.net.parameters()),
        precision=precision.label,
        start_epoch=start_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_images=sum(x.shape[0] for x, _ in held_out),
    )

    reference, reference_labels = reference_batch(cfg)

    sample_noise = torch.randn(
        cfg.num_samples,
        spec.channels,
        cfg.image_size,
        cfg.image_size,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    extra: list[LoggerBackend] = [] if observer is None else [ObserverBackend(observer)]
    logger = RunLogger.for_run(
        cfg.log_dir,
        console=cfg.log_console and group.is_main,
        jsonl=cfg.log_jsonl and group.is_main,
        tensorboard=cfg.tensorboard and group.is_main,
        wandb=cfg.wandb and group.is_main,
        wandb_project=cfg.wandb_project,
        wandb_config=asdict(cfg),
        extra=extra,
    )

    run = Run(
        cfg=cfg,
        group=group,
        precision=precision,
        diffusion=diffusion,
        ema=ema,
        train_net=train_net,
        ddp=ddp,
        optim=optim,
        sched=sched,
        scaler=scaler,
        model_params=model_params,
        master_params=master_params,
        step_params=step_params,
        say=say,
    )

    try:
        with logger, interrupt_guard() as interrupts:
            for epoch in range(start_epoch, cfg.num_epochs):
                loader_rng.manual_seed(epoch_seed(cfg.seed, epoch))
                set_loader_epoch(loader, epoch)

                outcome = run_epoch(
                    run,
                    loader,
                    logger,
                    epoch=epoch,
                    interrupts=interrupts,
                    observer=observer,
                    best_val=best_val,
                )
                log_epoch(
                    run,
                    logger,
                    quartile_sums=outcome.quartile_sums,
                    quartile_counts=outcome.quartile_counts,
                    elapsed=outcome.elapsed,
                    images=outcome.images,
                )
                new_best: float | None = None
                if held_out and not outcome.cancelled and (epoch + 1) % cfg.val_every == 0:
                    best_val, new_best = score_epoch(run, logger, held_out, best_val=best_val)

                logger.flush(step=epoch)

                if outcome.cancelled:
                    break

                finish_epoch(
                    run,
                    epoch=epoch,
                    best_val=best_val,
                    new_best=new_best,
                    reference=reference,
                    reference_labels=reference_labels,
                    sample_noise=sample_noise,
                    observer=observer,
                )

        if master_params is not None:
            cast(UNet, diffusion.net).convert_to_fp32()

        diffusion.net.load_state_dict(ema.module.state_dict())
        barrier(group)
    finally:
        distributed_shutdown()
    return diffusion


if __name__ == "__main__":
    train()
