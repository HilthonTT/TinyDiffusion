"""Deprecated alias for the training modules this package was split into.

The loop was named ``train_mnist`` back when MNIST was the only thing it could
train on. It now trains whatever :data:`~tinydiffusion.data.datasets.DATASETS`
holds, so the name is gone; importing it still works, and warns.

New code should import from the module that owns the name:

============================  ==================================================
was                           now
============================  ==================================================
``train_mnist.train_mnist``   :func:`tinydiffusion.training.train.train`
``train_mnist.build_model``   :func:`tinydiffusion.training.model.build_model`
``train_mnist.*checkpoint*``  :mod:`tinydiffusion.training.checkpoints`
``train_mnist.lr_factor``     :func:`tinydiffusion.training.lr.lr_factor`
============================  ==================================================
"""

import warnings

from tinydiffusion.training.checkpoints import (
    ARCHITECTURE_FIELDS,
    BEST_CHECKPOINT,
    INTERRUPTED_CHECKPOINT,
    LAST_CHECKPOINT,
    check_resume_compatible,
    load_checkpoint,
    read_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from tinydiffusion.training.config import TrainConfig
from tinydiffusion.training.lr import lr_factor
from tinydiffusion.training.model import build_model
from tinydiffusion.training.train import (
    QUARTILE_EVERY,
    epoch_seed,
    save_samples,
    train,
    validation_batches,
)

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
    "lr_factor",
    "read_checkpoint",
    "restore_checkpoint",
    "save_checkpoint",
    "save_samples",
    "train_mnist",
    "validation_batches",
]

train_mnist = train
"""Deprecated alias for :func:`tinydiffusion.training.train.train`."""

warnings.warn(
    "tinydiffusion.training.train_mnist is deprecated and will be removed in 0.3.0; "
    "the loop is in tinydiffusion.training.train, the model builder in "
    "tinydiffusion.training.model, and checkpoint I/O in "
    "tinydiffusion.training.checkpoints",
    DeprecationWarning,
    stacklevel=2,
)
