"""Training loop components: weight averaging, schedules, checkpointing.

The training entry point stays in :mod:`tinydiffusion.training.train_mnist`
rather than being re-exported here: binding the function on the package would
shadow the module of the same name.
"""

from tinydiffusion.training.config import TrainConfig, load_config
from tinydiffusion.training.ema import EMA

__all__ = ["EMA", "TrainConfig", "load_config"]
