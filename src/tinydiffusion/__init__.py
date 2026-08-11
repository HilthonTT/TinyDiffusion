"""TinyDiffusion: a compact PyTorch implementation of a diffusion model."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tinydiffusion")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
