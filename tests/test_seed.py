import random

import numpy as np
import torch

from tinydiffusion.utils import seed_everything


def test_returns_the_applied_seed():
    assert seed_everything(1234) == 1234


def test_torch_stream_is_reproducible():
    seed_everything(42)
    first = torch.randn(8)
    seed_everything(42)
    assert torch.equal(first, torch.randn(8))


def test_python_and_numpy_streams_are_reproducible():
    seed_everything(42)
    expected = (random.random(), float(np.random.rand()))
    seed_everything(42)
    assert (random.random(), float(np.random.rand())) == expected
