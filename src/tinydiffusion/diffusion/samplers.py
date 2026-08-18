"""The samplers a checkpoint can be drawn from, by name.

A sampler is a function over a trained process, not a property of it, so which
one to use is a runtime choice: the same checkpoint can be drawn from with
either. This module is the one place that knows the set of them, so the CLI,
the config, the training loop's per-epoch grids and the FID command all agree
on the names without importing each other.
"""

from typing import Protocol

import torch
import torch.nn as nn

from tinydiffusion.diffusion.ddim import ddim_sample
from tinydiffusion.diffusion.dpm_solver import dpmpp_sample
from tinydiffusion.diffusion.gaussian_diffusion import Diffusion

__all__ = ["DEFAULT_SAMPLER", "SAMPLERS", "Sampler", "get_sampler", "sampler_names"]


class Sampler(Protocol):
    """The call signature every sampler shares.

    Keyword-for-keyword the signature of
    :func:`~tinydiffusion.diffusion.ddim.ddim_sample`, which is the reference
    implementation; a sampler that cannot honour an argument — `eta`, for the
    deterministic solvers — raises rather than ignoring it.
    """

    def __call__(
        self,
        diffusion: Diffusion,
        num_samples: int,
        size: tuple[int, ...],
        device: torch.device | str,
        num_steps: int = ...,
        eta: float = ...,
        model: nn.Module | None = ...,
        timesteps: torch.Tensor | None = ...,
        clip_denoised: bool = ...,
        noise: torch.Tensor | None = ...,
        generator: torch.Generator | None = ...,
    ) -> torch.Tensor:
        """Draw ``num_samples`` images. See :func:`ddim_sample` for the arguments."""
        ...


SAMPLERS: dict[str, Sampler] = {
    "ddim": ddim_sample,
    "dpmpp": dpmpp_sample,
}
"""Name to sampler. ``ddim`` is the safe default; ``dpmpp`` is the fast one."""

DEFAULT_SAMPLER = "ddim"
"""What a config that says nothing gets, and what every checkpoint so far used."""


def sampler_names() -> tuple[str, ...]:
    """The registered sampler names.

    Returns:
        The keys of :data:`SAMPLERS`, sorted, for error messages and CLI choices.
    """
    return tuple(sorted(SAMPLERS))


def get_sampler(name: str) -> Sampler:
    """Resolve a sampler name to the function that implements it.

    Args:
        name: a key of :data:`SAMPLERS`.

    Returns:
        The sampler.

    Raises:
        ValueError: if no sampler is registered under that name.
    """
    try:
        return SAMPLERS[name]
    except KeyError:
        raise ValueError(
            f"unknown sampler {name!r}, expected one of {', '.join(sampler_names())}"
        ) from None
