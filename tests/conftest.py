import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_torch():
    """Keep every test on a known RNG state."""
    torch.manual_seed(0)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
