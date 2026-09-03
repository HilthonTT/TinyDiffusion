"""The checkpoint behind the API: loaded once, sampled per request."""

import re
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.utils import save_image

from tinydiffusion.data.datasets import denormalize
from tinydiffusion.diffusion.guidance import conditioned
from tinydiffusion.diffusion.samplers import get_sampler
from tinydiffusion.sampling import grid_width, load_for_sampling, resolve_labels
from tinydiffusion.server.config import ServerConfig
from tinydiffusion.utils.precision import apply_precision, resolve_precision

_FILENAME = re.compile(r"\A[0-9a-f]{32}\.png\Z")
"""What :meth:`SamplerService.image_path` will accept. See its docstring."""


@dataclass(frozen=True, slots=True)
class SamplePlan:
    """A request that has been checked against the checkpoint, ready to render.

    Splitting the checking from the rendering is what lets the server answer a
    bad request without spending anything on it. :meth:`SamplerService.plan`
    is pure arithmetic over the checkpoint's own numbers — microseconds, safe
    to run on the event loop — while :meth:`SamplerService.render` is the
    minutes-long GPU call that has to be dispatched to a thread and admitted
    through the server's concurrency limit. A request that was never going to
    work is therefore rejected before it occupies either.

    Attributes:
        num_images: how many images to generate.
        labels: the classes the caller asked for, kept as given because the
            grid layout depends on whether they were named or defaulted.
        y: one label per image, or None for an unconditional checkpoint.
        num_steps: denoising steps, with the checkpoint's default resolved in.
        eta: 0.0 is deterministic DDIM; 1.0 reproduces ancestral DDPM.
        scale: classifier-free guidance scale, resolved.
        rescale: guidance rescale factor, resolved.
        generator: the request's own RNG, or None to draw from the global one.
    """

    num_images: int
    labels: tuple[int, ...] | None
    y: torch.Tensor | None
    num_steps: int
    eta: float
    scale: float
    rescale: float
    generator: torch.Generator | None


class SamplerService:
    """Holds a loaded checkpoint and renders grids from it.

    Loading is the expensive part — weights, EMA and the schedule buffers — so
    it happens once at startup rather than per request. Sampling itself is
    serialised behind a lock: the requests share one network on one device, and
    letting two chains interleave on a GPU buys nothing but memory pressure.

    Attributes:
        config: the server settings this was built from.
        precision: what the network actually runs in, after the requested
            setting was reduced to something this device supports.
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
        raw_net = ema.module if config.use_ema else self._diffusion.net
        self.precision = resolve_precision(config.precision, self._cfg.device)
        self._net = apply_precision(raw_net, self.precision, self._cfg.device)
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

    @property
    def default_guidance_rescale(self) -> float:
        """Guidance rescale factor used when a request does not ask for one."""
        return self._cfg.guidance_rescale

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
        guidance_rescale: float | None = None,
        steps: int | None = None,
        eta: float = 0.0,
        seed: int | None = None,
    ) -> Path:
        """Check a request and render it, in one call.

        What a caller that is doing nothing else wants. The server splits the
        two halves apart — see :meth:`plan` — because it has to answer a bad
        request without dispatching it to a thread.

        Args:
            num_images: how many images to generate.
            labels: classes to generate, cycled over the grid. Conditional
                checkpoints only.
            guidance: classifier-free guidance scale, or None for the
                checkpoint's.
            guidance_rescale: guidance rescale factor in [0, 1], or None for
                the checkpoint's. See
                :func:`~tinydiffusion.diffusion.guidance.rescale_guided`.
            steps: denoising steps, or None for the checkpoint's.
            eta: 0.0 is deterministic DDIM; 1.0 reproduces ancestral DDPM.
                Only a checkpoint sampled with ``ddim`` accepts a non-zero
                value; the deterministic solvers reject one.
            seed: seed for this request's own generator, or None to draw from
                the global RNG.

        Returns:
            The path of the written PNG.

        Raises:
            ValueError: if the request does not fit the checkpoint. See
                :meth:`plan`.
        """
        return self.render(
            self.plan(
                num_images=num_images,
                labels=labels,
                guidance=guidance,
                guidance_rescale=guidance_rescale,
                steps=steps,
                eta=eta,
                seed=seed,
            )
        )

    def plan(
        self,
        *,
        num_images: int,
        labels: Sequence[int] | None = None,
        guidance: float | None = None,
        guidance_rescale: float | None = None,
        steps: int | None = None,
        eta: float = 0.0,
        seed: int | None = None,
    ) -> SamplePlan:
        """Check a request against the checkpoint and resolve its defaults.

        Cheap enough to run anywhere: it compares numbers, and the largest
        thing it allocates is one label per image. Nothing here touches the
        network. See :meth:`render` for the half that does.

        Args:
            num_images: how many images to generate.
            labels: classes to generate, cycled over the grid. Conditional
                checkpoints only.
            guidance: classifier-free guidance scale, or None for the
                checkpoint's.
            guidance_rescale: guidance rescale factor in [0, 1], or None for
                the checkpoint's.
            steps: denoising steps, or None for the checkpoint's.
            eta: 0.0 is deterministic DDIM; 1.0 reproduces ancestral DDPM.
            seed: seed for this request's own generator, or None to draw from
                the global RNG.

        Returns:
            The checked request.

        Raises:
            ValueError: if the request does not fit the checkpoint — too many
                images, labels for an unconditional model, a label naming no
                class, or an out-of-range step count, eta or rescale factor.
        """
        if not 1 <= num_images <= self.config.max_images:
            raise ValueError(
                f"num_images must lie in [1, {self.config.max_images}], got {num_images}"
            )
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta must lie in [0, 1], got {eta}")
        if eta != 0.0 and self._cfg.sampler != "ddim":
            raise ValueError(
                f"this checkpoint samples with {self._cfg.sampler}, a deterministic "
                f"solver, so eta must be 0, got {eta}"
            )
        num_steps = steps if steps is not None else self._cfg.sample_steps
        if not 1 <= num_steps <= self._cfg.num_timesteps:
            raise ValueError(f"steps must lie in [1, {self._cfg.num_timesteps}], got {num_steps}")
        if guidance is not None and self._cfg.num_classes is None and guidance != 1.0:
            raise ValueError("this checkpoint is unconditional, so guidance does not apply")

        y = resolve_labels(
            labels,
            num_images=num_images,
            num_classes=self._cfg.num_classes,
            device=self._cfg.device,
        )

        generator = (
            torch.Generator(device=self._cfg.device).manual_seed(seed) if seed is not None else None
        )

        return SamplePlan(
            num_images=num_images,
            labels=None if labels is None else tuple(labels),
            y=y,
            num_steps=num_steps,
            eta=eta,
            scale=self._cfg.guidance if guidance is None else guidance,
            rescale=self._cfg.guidance_rescale if guidance_rescale is None else guidance_rescale,
            generator=generator,
        )

    def render(self, plan: SamplePlan) -> Path:
        """Draw a checked request and write it as a PNG.

        The expensive half: a full denoising chain per image, serialised behind
        a lock because the requests share one network on one device. Minutes,
        for a large request on a slow device — which is why the server runs
        this off the event loop, and why it bounds how many callers may be
        waiting on the lock at once.

        Args:
            plan: a request from :meth:`plan`.

        Returns:
            The path of the written PNG.
        """
        with self._lock:
            images = get_sampler(self._cfg.sampler)(
                self._diffusion,
                plan.num_images,
                (self._spec.channels, self._cfg.image_size, self._cfg.image_size),
                self._cfg.device,
                num_steps=plan.num_steps,
                eta=plan.eta,
                model=conditioned(
                    self._net,
                    plan.y,
                    num_classes=self._cfg.num_classes,
                    scale=plan.scale,
                    rescale=plan.rescale,
                ),
                generator=plan.generator,
                spacing=self._cfg.sample_spacing,
            )

        path = self.image_dir / f"{uuid.uuid4().hex}.png"
        save_image(
            denormalize(images),
            path,
            nrow=grid_width(plan.num_images, self._cfg.num_classes, plan.labels),
        )
        self.prune_images(keep_path=path)
        return path

    def prune_images(self, *, keep_path: Path | None = None) -> int:
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
        # Concurrent renders each prune after writing; with a small keep count
        # one of them would otherwise delete the file another is returning.
        doomed.discard(keep_path)

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
        device = torch.device(self._cfg.device)
        if torch.cuda.is_available() and device.type == "cuda":
            memory = {
                "allocated_gb": round(torch.cuda.memory_allocated(device) / 1024**3, 2),
                "reserved_gb": round(torch.cuda.memory_reserved(device) / 1024**3, 2),
                "name": torch.cuda.get_device_name(device),
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
            "sampler": self._cfg.sampler,
            "sample_spacing": self._cfg.sample_spacing,
            "precision": self.precision,
            "default_guidance": self.default_guidance,
            "default_guidance_rescale": self.default_guidance_rescale,
            "max_images": self.config.max_images,
            "max_inflight": self.config.max_inflight,
            "image_ttl": self.config.image_ttl,
            "keep_images": self.config.keep_images,
            "memory": memory,
        }
