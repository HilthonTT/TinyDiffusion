"""Run configuration for MNIST training, and its TOML/checkpoint round-trip."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Self

import torch

from tinydiffusion.data.datasets import DEFAULT_DATASET, DatasetSpec, dataset_spec
from tinydiffusion.diffusion.gaussian_diffusion import (
    LossType,
    LossWeighting,
    ModelMeanType,
    ModelVarType,
)
from tinydiffusion.diffusion.samplers import DEFAULT_SAMPLER, get_sampler
from tinydiffusion.diffusion.timesteps import timestep_sampler

# Fields whose declared type is not what TOML (or a checkpoint's provenance
# dict, which stringifies Paths) hands back, so they need coercing on the way in.
_PATH_FIELDS = frozenset({"data_root", "out_dir", "ckpt_dir", "log_dir"})
_TUPLE_FIELDS = frozenset({"channel_mult", "attn_resolutions", "betas"})


@dataclass(slots=True)
class TrainConfig:
    """Hyperparameters for an MNIST training run.

    Defaults target a single consumer GPU and reach recognisable digits within
    a handful of epochs. Field names mirror the constructor arguments of
    :class:`~tinydiffusion.models.unet.UNet` and
    :class:`~tinydiffusion.diffusion.ddpm.DDPM` so the wiring in
    :func:`~tinydiffusion.training.model.build_model` stays a
    rename-free pass-through.

    ``predict``, ``variance`` and ``objective`` select the diffusion
    parameterisation. At their defaults they describe exactly what
    :class:`~tinydiffusion.diffusion.ddpm.DDPM` implements, and that is what
    gets built; any other combination is served by
    :class:`~tinydiffusion.diffusion.gaussian_diffusion.GaussianDiffusion`.

    ``predict = "v"`` is the velocity parameterisation, and the one to pair
    with ``zero_snr = true``: the published schedules leave a little signal in
    x_T, which the model learns to lean on and pure noise does not have, and
    rescaling that away costs epsilon prediction its target. Together they are
    what make a short sampling chain — ``sampler = "dpmpp"`` at 15 or 20
    ``sample_steps`` — hold up.

    ``loss_weighting = "min_snr"`` clamps each timestep's weight at
    ``min_snr_gamma`` (Hang et al. 2023). Uniform weighting quietly favours the
    low-noise timesteps, which are nearly solved already; the clamp stops them
    drowning out the rest and typically reaches a given loss in far fewer
    epochs. ``timestep_sampler = "loss_second_moment"`` is the other half of
    the same problem, attacking the variance of *which* timesteps get drawn
    rather than what they are worth once drawn; it earns its keep on the
    variational objectives and does little for plain MSE.

    ``dataset`` names an entry in
    :data:`~tinydiffusion.data.datasets.DATASETS`, which is where the channel
    count, the label space and the augmentation come from. Nothing downstream
    hard-codes any of them, so switching datasets is this field plus whatever
    ``num_classes`` and ``image_size`` the new one implies.

    ``num_classes`` opts into class-conditional training — 10 for MNIST's
    digits. ``class_dropout`` is the fraction of training labels replaced by
    the null token, which is what gives the same network an unconditional
    prediction for ``guidance`` to extrapolate from; see
    :mod:`~tinydiffusion.diffusion.guidance`.

    ``guidance_rescale`` corrects the scale that extrapolation inflates (Lin et
    al. 2023 §3.4). Guidance travels along ``cond - uncond`` without regard for
    distance, so past a scale of about 3 the prediction's standard deviation
    outgrows anything the model was trained on and the images come back
    flat and over-saturated; 0.7 is the published blend, and 0 is plain
    guidance. It is worth most on exactly the configuration ``predict = "v"``
    with ``zero_snr = true`` produces, where the terminal step carries no
    signal to anchor the scale, and it does nothing at ``guidance = 1.0``,
    where there is no extrapolation to correct.

    ``lr_warmup`` ramps the learning rate linearly from zero over that many
    optimiser steps before holding it at ``lr``. Diffusion training is unstable
    in the first few hundred steps at full LR; 0 turns the ramp off.
    ``lr_schedule`` decides what happens after the ramp: ``constant`` holds
    ``lr`` for the rest of the run, and ``cosine`` decays it to zero over the
    remaining optimiser steps, which usually buys a little final quality. The
    decay is a function of ``num_epochs``, so resuming with a different value
    resumes onto a different curve.

    ``grad_accum`` runs that many micro-batches per optimiser step, for an
    effective batch of ``batch_size * grad_accum`` in the memory of one
    ``batch_size``. It multiplies neither the epoch nor the logged loss: what
    it changes is how many batches go into each update, so ``lr_warmup`` and
    the LR schedule — which count updates — cover proportionally more data.

    ``amp_dtype`` picks the autocast type on CUDA. ``fp16`` needs the gradient
    scaler and can skip a step when it overflows; ``bf16`` has the range not to
    and so runs unscaled, but wants Ampere or newer. ``compile`` wraps the
    network in :func:`torch.compile` for training only — the checkpoint, the
    EMA and every sampler keep the eager module, so a compiled run's
    checkpoints stay ordinary ones. ``channels_last`` is worth measuring rather
    than assuming: it wins on wide convolutional stacks under AMP and can lose
    on a small one.

    ``grad_checkpoint`` drops the U-Net's intermediate activations and
    recomputes them in the backward pass, which is what makes a wider model or
    a larger batch fit on a card it otherwise would not. Roughly a third more
    compute per step, so reach for it when memory rather than speed is the
    binding constraint. It changes no weights and no result: a checkpoint
    trained with it on resumes with it off and vice versa. Sampling is
    unaffected either way, since there is no backward pass to save for.

    ``deterministic`` trades throughput for bit-reproducibility: it forces
    deterministic cuDNN/cuBLAS kernels and turns off the cuDNN autotuner, whose
    kernel choice is otherwise free to vary between runs on identical inputs.
    The batch order is a function of ``seed`` and the epoch index either way,
    so it is only the kernels this changes.

    ``val_every`` scores a fixed slice of the held-out split after each epoch,
    on a pinned timestep grid and pinned noise, so the number moves only with
    the weights. ``keep_best`` uses it to maintain ``best.pt`` alongside
    ``last.pt``: training loss alone cannot tell you which epoch to sample
    from, and the last one is not reliably the best.
    """

    # data
    dataset: str = DEFAULT_DATASET
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

    # conditioning
    num_classes: int | None = None
    class_dropout: float = 0.1
    guidance: float = 1.0
    guidance_rescale: float = 0.0

    # diffusion
    num_timesteps: int = 1000
    schedule: Literal["cosine", "linear"] = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    predict: Literal["epsilon", "start_x", "v", "previous_x"] = "epsilon"
    variance: Literal["fixed_small", "fixed_large", "learned", "learned_range"] = "fixed_small"
    objective: Literal["mse", "rescaled_mse", "kl", "rescaled_kl"] = "mse"
    zero_snr: bool = False
    loss_weighting: Literal["uniform", "min_snr"] = "uniform"
    min_snr_gamma: float = 5.0
    timestep_sampler: Literal["uniform", "loss_second_moment"] = "uniform"

    # optimisation
    num_epochs: int = 30
    lr: float = 2e-4
    lr_warmup: int = 500  # optimiser steps to ramp the LR over; 0 disables it
    lr_schedule: Literal["constant", "cosine"] = "constant"
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_accum: int = 1  # micro-batches per optimiser step
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    ema_warmup: int = 2000

    # validation
    val_every: int = 1
    val_steps: int = 10
    val_batches: int = 4

    # bookkeeping
    seed: int = 0
    deterministic: bool = False
    amp: bool = True
    amp_dtype: Literal["fp16", "bf16"] = "fp16"
    compile: bool = False
    channels_last: bool = False
    grad_checkpoint: bool = False
    sample_every: int = 1
    num_samples: int = 16
    sampler: str = DEFAULT_SAMPLER
    sample_steps: int = 50
    out_dir: Path = Path("contents")
    ckpt_dir: Path = Path("checkpoints")
    keep_best: bool = True
    keep_last: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # tracking
    log_dir: Path = Path("runs/mnist")
    log_console: bool = True
    log_jsonl: bool = True
    tensorboard: bool = False

    def __post_init__(self) -> None:
        """Reject configurations that would only fail an epoch into the run.

        Every field that has a range is checked here rather than left to the
        first thing that trips over it: some of them fail late and obscurely —
        a batch size of 0 inside the dataloader, an empty ``channel_mult``
        inside the U-Net — and ``ema_decay`` outside [0, 1] does not fail at
        all, it quietly ships diverging weights.

        Raises:
            ValueError: if the dataset is unregistered, the schedule is
                unknown, a step count cannot index the training schedule, a
                size or rate falls outside the range it has to lie in, or the
                conditioning settings do not describe a trainable model.
        """
        # Raises on an unregistered name, and is what every downstream shape
        # is read from, so it is checked before anything else can use it.
        spec = self.dataset_spec()
        if self.image_size < 1:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must not be negative, got {self.num_workers}")
        if self.base_channels < 1:
            raise ValueError(f"base_channels must be positive, got {self.base_channels}")
        if not self.channel_mult or any(mult < 1 for mult in self.channel_mult):
            raise ValueError(
                f"channel_mult must hold at least one positive multiplier, got {self.channel_mult}"
            )
        if self.num_res_blocks < 1:
            raise ValueError(f"num_res_blocks must be positive, got {self.num_res_blocks}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {self.dropout}")
        if self.num_timesteps < 1:
            # Checked before the two step counts below, which are bounded by it
            # and would otherwise report an empty range as the problem.
            raise ValueError(f"num_timesteps must be positive, got {self.num_timesteps}")
        if self.schedule not in ("cosine", "linear"):
            raise ValueError(f"unknown schedule {self.schedule!r}, expected 'cosine' or 'linear'")
        if not 1 <= self.sample_steps <= self.num_timesteps:
            raise ValueError(
                f"sample_steps must lie in [1, {self.num_timesteps}], got {self.sample_steps}"
            )
        # Raises on an unregistered name; checked here so a typo costs nothing
        # rather than being found by the first per-epoch grid.
        get_sampler(self.sampler)
        timestep_sampler(self.timestep_sampler, self.num_timesteps)
        if self.min_snr_gamma <= 0:
            raise ValueError(f"min_snr_gamma must be positive, got {self.min_snr_gamma}")
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        if not 1 <= self.val_steps <= self.num_timesteps:
            raise ValueError(
                f"val_steps must lie in [1, {self.num_timesteps}], got {self.val_steps}"
            )
        if self.val_batches < 0:
            raise ValueError(f"val_batches must not be negative, got {self.val_batches}")
        if self.num_epochs < 0:
            raise ValueError(f"num_epochs must not be negative, got {self.num_epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.lr_warmup < 0:
            raise ValueError(f"lr_warmup must not be negative, got {self.lr_warmup}")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(
                f"unknown lr_schedule {self.lr_schedule!r}, expected 'constant' or 'cosine'"
            )
        if self.grad_accum < 1:
            raise ValueError(f"grad_accum must be positive, got {self.grad_accum}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must not be negative, got {self.weight_decay}")
        if self.grad_clip < 0:
            raise ValueError(f"grad_clip must not be negative, got {self.grad_clip}")
        if len(self.betas) != 2 or not all(0.0 <= b < 1.0 for b in self.betas):
            raise ValueError(f"betas must be two values in [0, 1), got {self.betas}")
        if not 0.0 <= self.ema_decay <= 1.0:
            # Outside [0, 1] the average extrapolates away from the weights it
            # is meant to follow, and nothing downstream ever says so: the loss
            # keeps falling while every sample, every best.pt comparison and
            # every shipped checkpoint comes from weights that are diverging.
            raise ValueError(f"ema_decay must lie in [0, 1], got {self.ema_decay}")
        if self.ema_warmup < 0:
            raise ValueError(f"ema_warmup must not be negative, got {self.ema_warmup}")
        if self.amp_dtype not in ("fp16", "bf16"):
            raise ValueError(f"unknown amp_dtype {self.amp_dtype!r}, expected 'fp16' or 'bf16'")
        if self.keep_last < 0:
            raise ValueError(f"keep_last must not be negative, got {self.keep_last}")
        self._check_conditioning(spec)
        self.diffusion_types()

    def _check_conditioning(self, spec: DatasetSpec) -> None:
        """Reject conditioning settings that cannot produce what they promise.

        Args:
            spec: the dataset being trained on, whose label space
                ``num_classes`` has to match.

        Raises:
            ValueError: if the class count does not match the dataset's, the
                dropout rate is out of range, or guidance is asked for from a
                model that will not learn the unconditional prediction it
                needs.
        """
        if self.num_classes is not None and self.num_classes != spec.num_classes:
            # The labels come from the dataset, so a count that disagrees with
            # it either indexes past the embedding table or leaves rows that
            # nothing ever trains.
            raise ValueError(
                f"num_classes={self.num_classes} does not match {spec.name}, which has "
                f"{spec.num_classes} classes; use num_classes={spec.num_classes} or "
                "leave it unset to train unconditionally"
            )
        if not 0.0 <= self.class_dropout < 1.0:
            raise ValueError(f"class_dropout must lie in [0, 1), got {self.class_dropout}")
        if self.guidance < 0.0:
            raise ValueError(f"guidance must not be negative, got {self.guidance}")
        if not 0.0 <= self.guidance_rescale <= 1.0:
            raise ValueError(f"guidance_rescale must lie in [0, 1], got {self.guidance_rescale}")
        if self.guidance_rescale > 0.0 and self.guidance == 1.0:
            # Silently a no-op rather than an error otherwise: at scale 1 the
            # guided prediction is the conditional one, so the correction is
            # the identity and the config promises something it cannot do.
            raise ValueError(
                f"guidance_rescale={self.guidance_rescale} has nothing to correct at "
                "guidance=1.0; raise guidance, or leave guidance_rescale at 0.0"
            )
        if self.guidance != 1.0 and self.num_classes is None:
            raise ValueError(
                f"guidance={self.guidance} needs a conditional model; set num_classes "
                "(10 for MNIST) or leave guidance at 1.0"
            )
        if self.guidance != 1.0 and self.class_dropout == 0.0:
            # The null embedding would never be trained, so extrapolating away
            # from it produces noise rather than a sharper digit.
            raise ValueError(
                "guidance needs class_dropout > 0 so the null token gets trained; "
                "use class_dropout=0.1, or guidance=1.0 for plain conditional sampling"
            )

    def dataset_spec(self) -> DatasetSpec:
        """Resolve ``dataset`` to the registry entry it names.

        Called from :meth:`__post_init__`, so a config that names nothing fails
        while it is being read rather than once a loader is built, and again by
        everything that needs the channel count or the label space.

        Returns:
            The spec for :attr:`dataset`.

        Raises:
            ValueError: if no dataset is registered under that name.
        """
        return dataset_spec(self.dataset)

    def diffusion_types(self) -> tuple[ModelMeanType, ModelVarType, LossType, LossWeighting]:
        """Resolve the parameterisation fields to their enums.

        Called from :meth:`__post_init__` so a bad combination fails while the
        config is being read rather than after the dataset has downloaded, and
        again by :func:`~tinydiffusion.training.model.build_model`, which
        is what keeps validation and construction from drifting apart.

        Returns:
            Tuple of ``(mean_type, var_type, loss_type, weighting)``.

        Raises:
            ValueError: if a field names no such option, or the combination
                cannot be trained.
        """
        try:
            mean_type = ModelMeanType(self.predict)
            var_type = ModelVarType(self.variance)
            loss_type = LossType(self.objective)
            weighting = LossWeighting(self.loss_weighting)
        except ValueError as exc:
            raise ValueError(f"bad diffusion parameterisation: {exc}") from exc

        if self.zero_snr and mean_type is ModelMeanType.EPSILON:
            # At a zero terminal SNR, x_T holds no signal at all, so an epsilon
            # prediction there says nothing about x_0 and the recovery divides
            # by a vanishing sqrt(alphabar). v prediction is the pairing the
            # rescaling was published with.
            raise ValueError(
                "zero_snr leaves the last timestep with no signal, which epsilon "
                "prediction cannot invert; use predict='v' (or 'start_x')"
            )
        if weighting is LossWeighting.MIN_SNR:
            if loss_type.is_variational:
                raise ValueError(
                    f"loss_weighting='min_snr' weights the MSE term, which "
                    f"objective={self.objective!r} does not have; use an MSE objective"
                )
            if mean_type is ModelMeanType.PREVIOUS_X:
                raise ValueError("loss_weighting='min_snr' is not defined for predict='previous_x'")

        if var_type.is_learned and loss_type is LossType.MSE:
            # Nothing in L_simple touches the variance head, so it would keep
            # its initialisation and be sampled from anyway.
            raise ValueError(
                f"variance={self.variance!r} needs an objective that trains it; "
                "use objective='rescaled_mse' (the hybrid loss) or a KL objective"
            )
        if loss_type is LossType.RESCALED_MSE and not var_type.is_learned:
            raise ValueError(
                f"objective='rescaled_mse' adds a variational term to train a learned "
                f"variance, but variance={self.variance!r}; use objective='mse' instead"
            )
        return mean_type, var_type, loss_type, weighting

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Self:
        """Build a config from a plain mapping, coercing paths and tuples.

        Used for both TOML files and the provenance dict inside a checkpoint,
        neither of which can represent :class:`~pathlib.Path` or ``tuple``.

        Args:
            values: field name to value. Every key must name a real field.

        Returns:
            The constructed config.

        Raises:
            ValueError: if the mapping contains keys that are not fields.
        """
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config field(s): {', '.join(sorted(unknown))}")

        coerced: dict[str, Any] = {}
        for name, value in values.items():
            if name in _PATH_FIELDS:
                coerced[name] = Path(value)
            elif name in _TUPLE_FIELDS:
                coerced[name] = tuple(value)
            else:
                coerced[name] = value
        return cls(**coerced)


def load_config(path: Path) -> TrainConfig:
    """Read a training config from a TOML file.

    Tables are cosmetic grouping only: every key is flattened into the flat
    :class:`TrainConfig` namespace, so ``[model] base_channels`` and a
    top-level ``base_channels`` mean the same thing. Unknown or repeated keys
    are errors rather than silent no-ops, which is how config typos usually
    survive to waste a training run.

    Args:
        path: TOML file to read.

    Returns:
        The parsed configuration.

    Raises:
        ValueError: if a key appears in more than one table, or names no field.
    """
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    flat: dict[str, Any] = {}
    origin: dict[str, str] = {}
    for key, value in raw.items():
        section = value if isinstance(value, dict) else {key: value}
        where = key if isinstance(value, dict) else "<top level>"
        for name, item in section.items():
            if name in flat:
                raise ValueError(f"config key {name!r} set in both [{origin[name]}] and [{where}]")
            flat[name] = item
            origin[name] = where

    return TrainConfig.from_mapping(flat)
