"""Reproducibility helpers."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch RNGs.

    Args:
        seed: The seed applied to every RNG.
        deterministic: If true, force deterministic cuDNN/cuBLAS kernels. This
            makes runs bit-reproducible at a noticeable throughput cost.

    Returns:
        The seed that was applied, so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required for deterministic matmul on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    return seed
