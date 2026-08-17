"""Training: the loop, and the pieces it is assembled from.

The loop itself is :func:`tinydiffusion.training.train.train`, and it stays in
its module rather than being re-exported here — binding the function on the
package would shadow the module of the same name. Around it sit
:mod:`~tinydiffusion.training.model` (build the process a config describes),
:mod:`~tinydiffusion.training.checkpoints` (save and resume it),
:mod:`~tinydiffusion.training.lr` (the LR schedule),
:mod:`~tinydiffusion.training.ema` and
:mod:`~tinydiffusion.training.validation`.

Only the model builder and checkpoint I/O are of interest outside training —
sampling, evaluation and the server all rebuild a checkpoint before loading it
— so they are re-exported here and reachable without importing the loop.
"""

from tinydiffusion.training.checkpoints import (
    BEST_CHECKPOINT,
    INTERRUPTED_CHECKPOINT,
    LAST_CHECKPOINT,
    load_checkpoint,
    read_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from tinydiffusion.training.config import TrainConfig, load_config
from tinydiffusion.training.ema import EMA
from tinydiffusion.training.model import build_model

__all__ = [
    "BEST_CHECKPOINT",
    "EMA",
    "INTERRUPTED_CHECKPOINT",
    "LAST_CHECKPOINT",
    "TrainConfig",
    "build_model",
    "load_checkpoint",
    "load_config",
    "read_checkpoint",
    "restore_checkpoint",
    "save_checkpoint",
]
