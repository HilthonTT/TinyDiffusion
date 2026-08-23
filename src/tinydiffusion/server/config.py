"""Configuration for the sampling server."""

from dataclasses import dataclass, field
from pathlib import Path

from tinydiffusion.utils.precision import DEFAULT_PRECISION, PRECISIONS

DEFAULT_HOST = "127.0.0.1"
"""Loopback by default. Serving a checkpoint is not an authenticated operation."""

DEFAULT_PORT = 8000

DEFAULT_MAX_IMAGES = 64
"""Ceiling on ``num_images`` per request, so one caller cannot pin the GPU."""

DEFAULT_IMAGE_TTL = 3600.0
"""Seconds a rendered PNG is kept. Long enough for a caller to fetch it."""

DEFAULT_KEEP_IMAGES = 256
"""Rendered PNGs retained regardless of age, newest first."""


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
        precision: what to run the network in; see
            :mod:`tinydiffusion.utils.precision`. Resolved once at startup
            rather than per request, so a caller cannot ask for one — the
            speed/accuracy trade is the operator's to make, and a per-request
            choice would make two identical requests return different images.
        max_images: largest ``num_images`` a single request may ask for.
        image_dir: where rendered PNGs are written, or None for a directory
            under the system temp folder.
        cors_origins: origins allowed to call the API from a browser. Empty
            disables CORS entirely, which is right unless a web page needs it.
        image_ttl: seconds a rendered PNG survives before it is swept, or 0 to
            keep everything forever. Every request writes a file that nothing
            else deletes, so an unbounded directory is a slow disk leak rather
            than a hypothetical one.
        keep_images: PNGs retained regardless of age, newest first, or 0 for no
            cap. Bounds the directory when requests arrive faster than the TTL
            expires them.
    """

    checkpoint: Path
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    device: str | None = None
    use_ema: bool = True
    precision: str = DEFAULT_PRECISION
    max_images: int = DEFAULT_MAX_IMAGES
    image_dir: Path | None = None
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    image_ttl: float = DEFAULT_IMAGE_TTL
    keep_images: int = DEFAULT_KEEP_IMAGES

    def __post_init__(self) -> None:
        """Reject settings that would only fail once a request arrives.

        Raises:
            ValueError: if the port, the image ceiling, either retention
                setting, or the precision is out of range.
        """
        if self.precision not in PRECISIONS:
            raise ValueError(
                f"unknown precision {self.precision!r}, expected one of {', '.join(PRECISIONS)}"
            )
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must lie in [1, 65535], got {self.port}")
        if self.max_images < 1:
            raise ValueError(f"max_images must be positive, got {self.max_images}")
        if self.image_ttl < 0:
            raise ValueError(f"image_ttl must not be negative, got {self.image_ttl}")
        if self.keep_images < 0:
            raise ValueError(f"keep_images must not be negative, got {self.keep_images}")
