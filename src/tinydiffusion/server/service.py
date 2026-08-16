"""The checkpoint behind the API: loaded once, sampled per request."""

import re
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

import torch
from torchvision.utils import save_image

from tinydiffusion.data.datasets import denormalize
from tinydiffusion.diffusion.ddim import ddim_sample
from tinydiffusion.diffusion.guidance import conditioned
from tinydiffusion.sampling import grid_width, load_for_sampling, resolve_labels
from tinydiffusion.server.config import ServerConfig

_FILENAME = re.compile(r"\A[0-9a-f]{32}\.png\Z")
"""What :meth:`SamplerService.image_path` will accept. See its docstring."""


class SamplerService:
    """Holds a loaded checkpoint and renders grids from it.

    Loading is the expensive part — weights, EMA and the schedule buffers — so
    it happens once at startup rather than per request. Sampling itself is
    serialised behind a lock: the requests share one network on one device, and
    letting two chains interleave on a GPU buys nothing but memory pressure.

    Attributes:
        config: the server settings this was built from.
        image_dir: directory rendered PNGs are written to.
    """

    def __init__(self, config: ServerConfig) -> None:
        """Load the checkpoint named by `config`.

        Args:
            config: server settings.
        """
        self.config = config
        self._diffusion, ema, self._cfg = load_for_sampling(config.checkpoint, config.device)
        self._spec = self._cfg.dataset_spec()
        self._net = ema.module if config.use_ema else self._diffusion.net
        self._lock = threading.Lock()

        self.image_dir = (
            config.image_dir
            if config.image_dir is not None
            else Path(tempfile.gettempdir()) / "tinydiffusion-images"
        )
        self.image_dir.mkdir(parents=True, exist_ok=True)

    @property
    def num_classes(self) -> int | None:
        """The checkpoint's class count, or None if it is unconditional."""
        return self._cfg.num_classes

    @property
    def device(self) -> str:
        """The device the checkpoint was loaded onto."""
        return self._cfg.device

    @property
    def default_steps(self) -> int:
        """DDIM steps used when a request does not ask for a count."""
        return self._cfg.sample_steps

    @property
    def default_guidance(self) -> float:
        """Guidance scale used when a request does not ask for one."""
        return self._cfg.guidance

    def image_path(self, filename: str) -> Path:
        """Resolve a served filename to a path inside the image directory.

        Every name this service hands out is a uuid4 hex plus ``.png``, so
        anything else is not ours to serve. Matching against that shape is what
        keeps a crafted name — ``..%2F..%2Fetc%2Fpasswd``, which the URL layer
        decodes back into separators before the handler sees it — from reaching
        outside the directory.

        Args:
            filename: the name from the URL path.

        Returns:
            The path to read.

        Raises:
            ValueError: if `filename` is not a name this service issued.
        """
        if not _FILENAME.fullmatch(filename):
            raise ValueError(f"not an image this server issued: {filename!r}")
        return self.image_dir / filename

    def sample(
        self,
        *,
        num_images: int,
        labels: Sequence[int] | None = None,
        guidance: float | None = None,
        steps: int | None = None,
        eta: float = 0.0,
        seed: int | None = None,
    ) -> Path:
        """Generate a grid of images and write it as a PNG.

        Args:
            num_images: how many images to generate.
            labels: classes to generate, cycled over the grid. Conditional
                checkpoints only.
            guidance: classifier-free guidance scale, or None for the
                checkpoint's.
            steps: DDIM steps, or None for the checkpoint's.
            eta: 0.0 is deterministic DDIM; 1.0 reproduces ancestral DDPM.
            seed: seed for this request's own generator, or None to draw from
                the global RNG.

        Returns:
            The path of the written PNG.

        Raises:
            ValueError: if the request does not fit the checkpoint — too many
                images, labels for an unconditional model, a label naming no
                class, or an out-of-range step count or eta.
        """
        if not 1 <= num_images <= self.config.max_images:
            raise ValueError(
                f"num_images must lie in [1, {self.config.max_images}], got {num_images}"
            )
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta must lie in [0, 1], got {eta}")
        num_steps = steps if steps is not None else self._cfg.sample_steps
        if not 1 <= num_steps <= self._cfg.num_timesteps:
            raise ValueError(f"steps must lie in [1, {self._cfg.num_timesteps}], got {num_steps}")
        if guidance is not None and self._cfg.num_classes is None and guidance != 1.0:
            raise ValueError("this checkpoint is unconditional, so guidance does not apply")

        # resolve_labels validates the labels against the checkpoint, and is
        # what makes the API's conditioning behave exactly like the CLI's.
        y = resolve_labels(
            labels,
            num_images=num_images,
            num_classes=self._cfg.num_classes,
            device=self._cfg.device,
        )
        scale = self._cfg.guidance if guidance is None else guidance

        # A request-local generator rather than seed_everything: reseeding the
        # process from a request would make one caller's `seed` reach into
        # every other caller's sampling, and outlive the request that asked
        # for it. This keeps the reproducibility and drops the side effect.
        generator = (
            torch.Generator(device=self._cfg.device).manual_seed(seed) if seed is not None else None
        )

        with self._lock:
            images = ddim_sample(
                self._diffusion,
                num_images,
                (self._spec.channels, self._cfg.image_size, self._cfg.image_size),
                self._cfg.device,
                num_steps=num_steps,
                eta=eta,
                model=conditioned(self._net, y, num_classes=self._cfg.num_classes, scale=scale),
                generator=generator,
            )

        path = self.image_dir / f"{uuid.uuid4().hex}.png"
        save_image(
            denormalize(images),
            path,
            nrow=grid_width(num_images, self._cfg.num_classes, labels),
        )
        self.prune_images()
        return path

    def prune_images(self) -> int:
        """Delete rendered PNGs past their age or count limit.

        Called after each render, because nothing else ever removes these
        files: a long-lived server would otherwise grow its image directory
        without bound. Only names this service issued are considered, so a
        directory shared with anything else keeps its other contents.

        Returns:
            How many files were deleted.
        """
        ttl = self.config.image_ttl
        keep = self.config.keep_images
        if not ttl and not keep:
            return 0

        # (mtime, path), oldest first. A file that vanishes underneath us — a
        # concurrent sweep, a user with a broom — is already in the state the
        # sweep wants it in.
        entries: list[tuple[float, Path]] = []
        for candidate in self.image_dir.glob("*.png"):
            if not _FILENAME.fullmatch(candidate.name):
                continue
            try:
                entries.append((candidate.stat().st_mtime, candidate))
            except OSError:
                continue
        entries.sort()

        now = time.time()
        doomed = set()
        if ttl:
            doomed |= {path for mtime, path in entries if now - mtime > ttl}
        if keep and len(entries) > keep:
            doomed |= {path for _, path in entries[: len(entries) - keep]}

        removed = 0
        for path in doomed:
            try:
                path.unlink()
            except OSError:
                continue
            removed += 1
        return removed

    def status(self) -> dict[str, object]:
        """Describe what is loaded, for the status endpoint.

        Returns:
            A JSON-serialisable mapping.
        """
        memory: dict[str, object] = {}
        if torch.cuda.is_available() and torch.device(self._cfg.device).type == "cuda":
            memory = {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
                "name": torch.cuda.get_device_name(torch.device(self._cfg.device)),
            }
        return {
            "checkpoint": str(self.config.checkpoint),
            "dataset": self._spec.name,
            "device": self._cfg.device,
            "weights": "ema" if self.config.use_ema else "raw",
            "image_size": self._cfg.image_size,
            "num_classes": self._cfg.num_classes,
            "num_timesteps": self._cfg.num_timesteps,
            "default_steps": self.default_steps,
            "default_guidance": self.default_guidance,
            "max_images": self.config.max_images,
            "image_ttl": self.config.image_ttl,
            "keep_images": self.config.keep_images,
            "memory": memory,
        }
