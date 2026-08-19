"""Score a trained checkpoint on held-out data."""

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from tinydiffusion.data.datasets import image_dataloader
from tinydiffusion.diffusion.guidance import Conditioned
from tinydiffusion.sampling import load_for_sampling
from tinydiffusion.training.validation import DEFAULT_VAL_STEPS, eval_timesteps
from tinydiffusion.utils.modules import eval_mode
from tinydiffusion.utils.seed import seed_everything

__all__ = ["DEFAULT_EVAL_STEPS", "EvalResult", "eval_timesteps", "evaluate_checkpoint"]

DEFAULT_EVAL_STEPS = DEFAULT_VAL_STEPS
"""Timesteps to score at. Enough to cover the schedule without being slow.

The same grid the training loop's per-epoch validation uses, so the two
numbers are on one scale.
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
    """

    checkpoint: Path
    split: str
    num_images: int
    loss: float
    per_timestep: list[tuple[int, float]]
    used_ema: bool

    def format(self) -> str:
        """Render the result as a short report.

        Returns:
            A multi-line string: the headline loss, then a per-timestep table.
        """
        weights = "ema" if self.used_ema else "raw"
        lines = [
            f"{self.checkpoint} | {self.split} split | "
            f"{self.num_images} images | {weights} weights",
            f"loss {self.loss:.5f}",
            "",
            "     t     loss",
        ]
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

    Returns:
        The scored result.

    Raises:
        ValueError: if ``split`` is not ``"test"`` or ``"train"``.
    """
    if split not in ("test", "train"):
        raise ValueError(f"unknown split {split!r}, expected 'test' or 'train'")

    diffusion, ema, cfg = load_for_sampling(checkpoint, device)
    net = ema.module if use_ema else diffusion.net

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
    )
