"""Score a trained checkpoint on held-out data.

Two numbers, measuring different things. The held-out *loss* is whatever the run
was trained on, pinned to a fixed timestep grid so two checkpoints can be
compared by it; it is cheap, and it is a proxy. The *variational bound* — bits
per dimension — is the likelihood the model actually assigns to real images, is
comparable against published figures, and costs one network evaluation per
timestep per image, so it is opt-in and scored on a slice rather than a split.

Neither is sample quality. :mod:`tinydiffusion.metrics` is where that lives, and
the three routinely disagree: a model can win on bits-per-dim and lose on FID.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from tinydiffusion.data.datasets import image_dataloader
from tinydiffusion.diffusion.gaussian_diffusion import GaussianDiffusion
from tinydiffusion.diffusion.guidance import Conditioned
from tinydiffusion.sampling import load_for_sampling
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.validation import DEFAULT_VAL_STEPS, eval_timesteps
from tinydiffusion.utils.modules import eval_mode
from tinydiffusion.utils.seed import seed_everything

__all__ = [
    "DEFAULT_BPD_IMAGES",
    "DEFAULT_EVAL_STEPS",
    "EvalResult",
    "eval_timesteps",
    "evaluate_checkpoint",
]

DEFAULT_EVAL_STEPS = DEFAULT_VAL_STEPS
"""Timesteps to score at. Enough to cover the schedule without being slow.

The same grid the training loop's per-epoch validation uses, so the two
numbers are on one scale.
"""

DEFAULT_BPD_IMAGES = 128
"""Images the variational bound is estimated over.

The bound walks the whole training schedule, so it costs ``num_timesteps``
network evaluations per image — a thousand of them at the default schedule,
against the couple of dozen the held-out loss spends. A few hundred images is
enough for the third decimal of a bits-per-dim figure and is minutes rather
than hours; the whole split is neither affordable nor necessary.
"""


@dataclass(slots=True)
class EvalResult:
    """The outcome of scoring one checkpoint.

    Attributes:
        checkpoint: the file that was scored.
        split: ``"test"`` or ``"train"``.
        num_images: how many images the average covers.
        loss: mean training objective over every timestep and image. That is
            noise-prediction MSE for the default parameterisation, and whatever
            the checkpoint was trained on otherwise — so two checkpoints are
            only comparable when they share one.
        per_timestep: ``(t, loss)`` pairs, ascending in ``t``.
        used_ema: whether the EMA weights were scored.
        bpd: the full variational bound in bits per dimension, or None if it
            was not asked for. Unlike `loss` this is comparable against
            published likelihoods and across parameterisations, since every
            model that defines a bound defines the same one.
        prior_bpd: the part of `bpd` contributed by the gap between q(x_T | x_0)
            and the standard normal prior. It depends on the schedule alone and
            no amount of training moves it, so a large share of the total means
            the forward process has not finished destroying the signal by x_T —
            which is what ``zero_snr`` exists to fix.
        num_bpd_images: how many images the bound covers. Smaller than
            `num_images`, since the bound costs a network evaluation per
            timestep.
    """

    checkpoint: Path
    split: str
    num_images: int
    loss: float
    per_timestep: list[tuple[int, float]]
    used_ema: bool
    bpd: float | None = None
    prior_bpd: float | None = None
    num_bpd_images: int = 0

    def format(self) -> str:
        """Render the result as a short report.

        Returns:
            A multi-line string: the headline loss, the bound where one was
            computed, then a per-timestep table.
        """
        weights = "ema" if self.used_ema else "raw"
        lines = [
            f"{self.checkpoint} | {self.split} split | "
            f"{self.num_images} images | {weights} weights",
            f"loss {self.loss:.5f}",
        ]
        if self.bpd is not None:
            # The prior term is printed beside the total rather than folded
            # into it: it is the one part of the bound training cannot improve,
            # so a reader comparing two checkpoints needs to see how much of
            # the difference was ever theirs to move.
            lines.append(
                f"bpd {self.bpd:.5f} (prior {self.prior_bpd:.5f}) over {self.num_bpd_images} images"
            )
        lines += ["", "     t     loss"]
        lines += [f"{t:6d}   {loss:.5f}" for t, loss in self.per_timestep]
        return "\n".join(lines)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: Path,
    *,
    split: str = "test",
    num_steps: int = DEFAULT_EVAL_STEPS,
    batch_size: int | None = None,
    data_root: Path | None = None,
    use_ema: bool = True,
    seed: int = 0,
    device: str | None = None,
    progress: bool = True,
    bpd: bool = False,
    bpd_images: int = DEFAULT_BPD_IMAGES,
) -> EvalResult:
    """Measure noise-prediction loss on a held-out split.

    The training loss is drawn at a random timestep per image, so two
    checkpoints cannot be compared by it directly. This pins the timesteps to a
    fixed grid and reseeds before each batch, so the only thing that varies
    between runs is the weights.

    Note that the number is a proxy: a lower held-out loss means the network
    predicts noise better, which correlates with sample quality but does not
    measure it. Look at the grids for that.

    Args:
        checkpoint: file to score.
        split: ``"test"`` for the held-out split, ``"train"`` for the training
            one — scoring both is how you see overfitting. Neither is
            augmented, so the training score measures the split rather than a
            random flip of it.
        num_steps: how many timesteps to score at.
        batch_size: images per batch, or None to reuse the checkpoint's.
        data_root: dataset directory, or None to reuse the checkpoint's.
        use_ema: score the EMA weights, which are what sampling uses.
        seed: base seed for the noise. Each batch draws from ``seed`` offset by
            its index, so the score is reproducible without every batch being
            scored against one and the same draw.
        device: device to score on. Defaults to CUDA when available.
        progress: draw a progress bar.
        bpd: also evaluate the full variational bound, in bits per dimension.
            It walks every timestep of the training schedule rather than the
            `num_steps` grid the loss uses, so it costs `num_timesteps` network
            evaluations per image — normally a thousand — and is scored over
            `bpd_images` rather than the whole split. Unclipped, since clamping
            the implied x_0 makes the number no longer a bound.
        bpd_images: how many images to estimate the bound over. Rounded up to a
            whole batch, and capped by the split.

    Returns:
        The scored result. `bpd` on it is None unless it was asked for.

    Raises:
        ValueError: if ``split`` is not ``"test"`` or ``"train"``, ``bpd`` is
            asked of a checkpoint whose process does not define one, or
            ``bpd_images`` is not positive.
    """
    if split not in ("test", "train"):
        raise ValueError(f"unknown split {split!r}, expected 'test' or 'train'")
    if bpd and bpd_images < 1:
        raise ValueError(f"bpd_images must be positive, got {bpd_images}")

    diffusion, ema, cfg = load_for_sampling(checkpoint, device)
    net = ema.module if use_ema else diffusion.net

    if bpd and not isinstance(diffusion, GaussianDiffusion):
        # DDPM is the fixed-variance special case, built directly rather than
        # through GaussianDiffusion, and has no variational bound to walk. The
        # weights are fine; it is the process wrapper that cannot answer.
        raise ValueError(
            # ASCII only: this reaches a terminal, and a Windows console on the
            # default code page renders an em dash as mojibake.
            "the variational bound needs the generalised process, and this "
            "checkpoint's parameterisation is served by the plain DDPM one. "
            "Train with a non-default `predict`, `variance` or `objective`; "
            '`variance = "learned_range"` with `objective = "rescaled_mse"` '
            "is the configuration the bound is normally quoted for."
        )

    loader = image_dataloader(
        cfg.dataset_spec(),
        data_root if data_root is not None else cfg.data_root,
        batch_size=batch_size if batch_size is not None else cfg.batch_size,
        train=split == "train",
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
        # A score has to cover the whole split in a fixed order. The train
        # split's own defaults would shuffle it and drop the ragged last
        # batch, quietly leaving up to batch_size-1 images out of the average.
        shuffle=False,
        drop_last=False,
    )

    steps = eval_timesteps(cfg.num_timesteps, num_steps).to(cfg.device)
    totals = torch.zeros(len(steps), dtype=torch.float64, device=cfg.device)
    num_images = 0

    with eval_mode(net):
        batches = tqdm(loader, desc=f"eval {split}", disable=not progress)
        for index, (x, y) in enumerate(batches):
            x = x.to(cfg.device, non_blocking=True)
            # A conditional model is scored on the true labels, and never with
            # guidance: the objective it was trained on is the conditional
            # prediction, and scoring it against a null or extrapolated one
            # would measure something the run never optimised.
            scored = (
                Conditioned(net, y.to(cfg.device, non_blocking=True))
                if cfg.num_classes is not None
                else net
            )
            # Reseed per batch so the noise depends only on position in the
            # split, never on batch count or how many timesteps were scored.
            # Offset by the batch index, or every batch would be scored against
            # the identical draw: the average would then cover num_images
            # images but only one noise sample per slot, and carry whatever
            # bias that single draw happens to have.
            seed_everything(seed + index)
            for i, step in enumerate(steps):
                t = step.expand(x.shape[0])
                totals[i] += diffusion.loss_at(x, t, model=scored).double() * x.shape[0]
            num_images += x.shape[0]

    bound = prior = None
    scored_for_bpd = 0
    if bpd:
        assert isinstance(diffusion, GaussianDiffusion)
        bound, prior, scored_for_bpd = _bound_bits_per_dim(
            diffusion,
            net,
            loader,
            cfg=cfg,
            num_images=bpd_images,
            seed=seed,
            progress=progress,
        )

    per_step = (totals / max(num_images, 1)).tolist()
    return EvalResult(
        checkpoint=checkpoint,
        split=split,
        num_images=num_images,
        loss=float(sum(per_step) / len(per_step)),
        per_timestep=[
            (int(t), float(loss)) for t, loss in zip(steps.tolist(), per_step, strict=True)
        ],
        used_ema=use_ema,
        bpd=bound,
        prior_bpd=prior,
        num_bpd_images=scored_for_bpd,
    )


@torch.no_grad()
def _bound_bits_per_dim(
    diffusion: GaussianDiffusion,
    net: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    cfg: TrainConfig,
    num_images: int,
    seed: int,
    progress: bool,
) -> tuple[float, float, int]:
    """Average the full variational bound over the first `num_images` images.

    Args:
        diffusion: the loaded process, which is what knows the bound.
        net: the network to evaluate, EMA or raw.
        loader: the same loader the loss was scored over, re-iterated. Its
            order is fixed and its tail is not dropped, so "the first N images"
            names the same set every time.
        cfg: the checkpoint's config, for the device and the label space.
        num_images: how many images to cover. The batch that crosses the limit
            is scored whole, so the realised count can overshoot by less than
            one batch; it is returned rather than assumed.
        seed: reseeded per batch, exactly as the loss pass does, so the forward
            noise the bound is estimated over depends only on position in the
            split.
        progress: draw a progress bar.

    Returns:
        Tuple of ``(total_bpd, prior_bpd, images_scored)``, the first two
        averaged over the images actually seen.
    """
    totals = torch.zeros((), dtype=torch.float64, device=cfg.device)
    priors = torch.zeros((), dtype=torch.float64, device=cfg.device)
    seen = 0

    batches = tqdm(
        loader,
        desc="eval bpd",
        total=-(-num_images // cfg.batch_size),
        disable=not progress,
    )
    with eval_mode(net):
        for index, (x, y) in enumerate(batches):
            if seen >= num_images:
                break
            x = x.to(cfg.device, non_blocking=True)
            scored = (
                Conditioned(net, y.to(cfg.device, non_blocking=True))
                if cfg.num_classes is not None
                else net
            )
            seed_everything(seed + index)
            # Unclipped: clamping the implied x_0 at each step tightens the
            # number without it still being an upper bound on the negative
            # log-likelihood, which is the only thing that makes it comparable
            # with anyone else's.
            terms = diffusion.calc_bpd_loop(x, model=scored, clip_denoised=False)
            totals += terms["total_bpd"].double().sum()
            priors += terms["prior_bpd"].double().sum()
            seen += x.shape[0]

    # tqdm holds the iterator open, and the loader's workers with it.
    batches.close()
    divisor = max(seen, 1)
    return float(totals / divisor), float(priors / divisor), seen
