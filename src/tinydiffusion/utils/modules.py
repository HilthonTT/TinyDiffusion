"""Helpers for temporarily changing the state of an ``nn.Module``."""

from collections.abc import Generator
from contextlib import contextmanager

import torch.nn as nn


@contextmanager
def eval_mode(module: nn.Module) -> Generator[nn.Module]:
    """Put `module` in eval mode for the duration of the block, then restore it.

    Sampling must not run dropout, but the caller's train/eval state has to
    survive: training loops sample mid-epoch and then keep training.

    Args:
        module: the module to switch into eval mode.

    Yields:
        The same module, now in eval mode.
    """
    was_training = module.training
    module.eval()
    try:
        yield module
    finally:
        module.train(was_training)
