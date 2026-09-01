"""The x_T every sampler starts from, and the checks worth doing before it."""

import torch

__all__ = ["initial_latent"]


def initial_latent(
    num_samples: int,
    size: tuple[int, ...],
    device: torch.device | str,
    *,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return the starting latent, either the one given or a fresh draw.

    Args:
        num_samples: batch size to generate.
        size: shape of one sample, e.g. ``(1, 28, 28)``.
        device: device to generate on.
        noise: the starting x_T, of shape ``(num_samples, *size)``. None draws
            a fresh one. Passing it in is what makes a series of grids
            comparable: reusing one latent across epochs shows the same images
            sharpening, where a fresh draw each time shows a different sample
            of the model.
        generator: RNG for the draw. None uses the global RNG.

    Returns:
        Tensor of shape ``(num_samples, *size)`` on `device`.

    Raises:
        ValueError: if `noise` has the wrong shape, or `generator` lives on
            another device.
    """
    if generator is not None and generator.device.type != torch.device(device).type:
        raise ValueError(
            f"generator is on {generator.device.type}, but sampling runs on "
            f"{torch.device(device).type}"
        )
    if noise is not None:
        if noise.shape != (num_samples, *size):
            raise ValueError(
                f"noise must be shaped {(num_samples, *size)}, got {tuple(noise.shape)}"
            )
        return noise.to(device)
    return torch.randn(num_samples, *size, device=device, generator=generator)
