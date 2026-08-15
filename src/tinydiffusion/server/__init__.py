"""HTTP API over a trained checkpoint.

``create_app`` and ``serve`` need FastAPI, which ships in the optional
``server`` extra, so they are imported lazily: ``ServerConfig`` and
``SamplerService`` stay importable — and the CLI stays runnable — without it.
"""

from typing import TYPE_CHECKING, Any

from tinydiffusion.server.config import (
    DEFAULT_HOST,
    DEFAULT_MAX_IMAGES,
    DEFAULT_PORT,
    ServerConfig,
)
from tinydiffusion.server.service import SamplerService

if TYPE_CHECKING:
    from tinydiffusion.server.app import create_app, serve

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_MAX_IMAGES",
    "DEFAULT_PORT",
    "SamplerService",
    "ServerConfig",
    "create_app",
    "serve",
]

_LAZY = frozenset({"create_app", "serve"})


def __getattr__(name: str) -> Any:
    """Import the FastAPI-dependent names only when they are asked for.

    Args:
        name: attribute being looked up.

    Returns:
        The requested object.

    Raises:
        AttributeError: if `name` is not exported by this package.
    """
    if name in _LAZY:
        from tinydiffusion.server import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
