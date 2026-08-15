"""Configuration for the sampling server."""

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
"""Loopback by default. Serving a checkpoint is not an authenticated operation."""

DEFAULT_PORT = 8000

DEFAULT_MAX_IMAGES = 64
"""Ceiling on ``num_images`` per request, so one caller cannot pin the GPU."""


@dataclass(slots=True)
class ServerConfig:
    """How to run the sampling server.

    Attributes:
        checkpoint: the trained checkpoint to serve. Loaded once at startup,
            not per request.
        host: interface to bind. The default is loopback: the API has no
            authentication, and generating images is expensive enough that an
            open port is a denial-of-service invitation. Set ``0.0.0.0``
            deliberately, behind something that does authenticate.
        port: port to bind.
        device: device to sample on, or None for CUDA when available.
        use_ema: serve the EMA weights, which are what ``sample`` uses.
        max_images: largest ``num_images`` a single request may ask for.
        image_dir: where rendered PNGs are written, or None for a directory
            under the system temp folder.
        cors_origins: origins allowed to call the API from a browser. Empty
            disables CORS entirely, which is right unless a web page needs it.
    """

    checkpoint: Path
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    device: str | None = None
    use_ema: bool = True
    max_images: int = DEFAULT_MAX_IMAGES
    image_dir: Path | None = None
    cors_origins: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Reject settings that would only fail once a request arrives.

        Raises:
            ValueError: if the port or image ceiling is out of range.
        """
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must lie in [1, 65535], got {self.port}")
        if self.max_images < 1:
            raise ValueError(f"max_images must be positive, got {self.max_images}")
