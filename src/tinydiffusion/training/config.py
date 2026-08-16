"""Run configuration for MNIST training, and its TOML/checkpoint round-trip."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Self

import torch

from tinydiffusion.data.datasets import DEFAULT_DATASET, DatasetSpec, dataset_spec
from tinydiffusion.diffusion.gaussian_diffusion import LossType, ModelMeanType, ModelVarType

# Fields whose declared type is not what TOML (or a checkpoint's provenance
# dict, which stringifies Paths) hands back, so they need coercing on the way in.
_PATH_FIELDS = frozenset({"data_root", "out_dir", "ckpt_dir", "log_dir"})
_TUPLE_FIELDS = frozenset({"channel_mult", "attn_resolutions"})


@dataclass(slots=True)
class TrainConfig:
    """Hyperparameters for an MNIST training run.

    Defaults target a single consumer GPU and reach recognisable digits within
    a handful of epochs. Field names mirror the constructor arguments of
    :class:`~tinydiffusion.models.unet.UNet` and
    :class:`~tinydiffusion.diffusion.ddpm.DDPM` so the wiring in
    :func:`~tinydiffusion.training.train_mnist.build_model` stays a
    rename-free pass-through.

    ``predict``, ``variance`` and ``objective`` select the diffusion
    parameterisation. At their defaults they describe exactly what
    :class:`~tinydiffusion.diffusion.ddpm.DDPM` implements, and that is what
    gets built; any other combination is served by
    :class:`~tinydiffusion.diffusion.gaussian_diffusion.GaussianDiffusion`.

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

    ``lr_warmup`` ramps the learning rate linearly from zero over that many
    optimiser steps before holding it at ``lr``. Diffusion training is unstable
    in the first few hundred steps at full LR; 0 turns the ramp off.

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

    # diffusion
    num_timesteps: int = 1000
    schedule: Literal["cosine", "linear"] = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    predict: Literal["epsilon", "start_x", "previous_x"] = "epsilon"
    variance: Literal["fixed_small", "fixed_large", "learned", "learned_range"] = "fixed_small"
    objective: Literal["mse", "rescaled_mse", "kl", "rescaled_kl"] = "mse"

    # optimisation
    num_epochs: int = 30
    lr: float = 2e-4
    lr_warmup: int = 500  # optimiser steps to ramp the LR over; 0 disables it
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
    sample_every: int = 1
    num_samples: int = 16
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

        Raises:
            ValueError: if the dataset is unregistered, the schedule is
                unknown, a step count cannot index the training schedule, a
                count that must be non-negative is not, or the conditioning
                settings do not describe a trainable model.
        """
        # Raises on an unregistered name, and is what every downstream shape
        # is read from, so it is checked before anything else can use it.
        spec = self.dataset_spec()
        if self.image_size < 1:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.schedule not in ("cosine", "linear"):
            raise ValueError(f"unknown schedule {self.schedule!r}, expected 'cosine' or 'linear'")
        if not 1 <= self.sample_steps <= self.num_timesteps:
            raise ValueError(
                f"sample_steps must lie in [1, {self.num_timesteps}], got {self.sample_steps}"
            )
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        if not 1 <= self.val_steps <= self.num_timesteps:
            raise ValueError(
                f"val_steps must lie in [1, {self.num_timesteps}], got {self.val_steps}"
            )
        if self.val_batches < 0:
            raise ValueError(f"val_batches must not be negative, got {self.val_batches}")
        if self.lr_warmup < 0:
            raise ValueError(f"lr_warmup must not be negative, got {self.lr_warmup}")
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

    def diffusion_types(self) -> tuple[ModelMeanType, ModelVarType, LossType]:
        """Resolve the three parameterisation fields to their enums.

        Called from :meth:`__post_init__` so a bad combination fails while the
        config is being read rather than after the dataset has downloaded, and
        again by :func:`~tinydiffusion.training.train_mnist.build_model`, which
        is what keeps validation and construction from drifting apart.

        Returns:
            Tuple of ``(mean_type, var_type, loss_type)``.

        Raises:
            ValueError: if a field names no such option, or the combination
                cannot be trained.
        """
        try:
            mean_type = ModelMeanType(self.predict)
            var_type = ModelVarType(self.variance)
            loss_type = LossType(self.objective)
        except ValueError as exc:
            raise ValueError(f"bad diffusion parameterisation: {exc}") from exc

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
        return mean_type, var_type, loss_type

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
