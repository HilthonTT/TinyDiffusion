import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_torch():
    """Keep every test on a known RNG state, and on a known cuDNN autotuner.

    `train` turns the autotuner on for any run on CUDA and leaves it on, so
    without this a GPU test silently decides what the next test that reads the
    flag sees — which makes the suite pass or fail on collection order.
    """
    torch.manual_seed(0)
    benchmark = torch.backends.cudnn.benchmark
    yield
    torch.backends.cudnn.benchmark = benchmark


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def wake():
    """Perturb every zero-initialised weight in a module.

    zero_module leaves each ResBlock's second conv and the U-Net's output conv
    at zero, so a freshly built network predicts zeros whatever it is
    conditioned on. Any test asking whether conditioning *reaches the output*
    would otherwise pass for the wrong reason.
    """

    def perturb(module):
        for param in module.parameters():
            if not param.any():
                torch.nn.init.normal_(param, std=0.1)
        return module

    return perturb
