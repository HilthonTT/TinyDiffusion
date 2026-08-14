"""Class conditioning and classifier-free guidance (Ho & Salimans 2022).

Every process here — :class:`~tinydiffusion.diffusion.ddpm.DDPM`,
:class:`~tinydiffusion.diffusion.gaussian_diffusion.GaussianDiffusion`,
:func:`~tinydiffusion.diffusion.ddim.ddim_sample` — already takes the network
to evaluate as a ``model=`` argument and calls it as ``model(x_t, t)``. Rather
than thread a label through all of them, conditioning is packaged as wrappers
matching that same two-argument call: the label is bound at construction, and
the process stays unaware that it exists.

That indirection is what makes guidance a drop-in. :class:`Conditioned`
supplies a fixed label; :class:`ClassifierFreeGuidance` runs the conditional
and unconditional predictions in a single batched pass and extrapolates away
from the latter, which is a different *model* from the sampler's point of view
but not a different sampler.
"""

import torch
import torch.nn as nn

__all__ = [
    "ClassifierFreeGuidance",
    "Conditioned",
    "conditioned",
    "cycled_labels",
    "drop_labels",
]


class _LabelBound(nn.Module):
    """Shared plumbing for the wrappers: hold a network and a batch of labels.

    Args:
        net: conditional network taking ``(x, t, y)``.
        labels: ``(B,)`` integer labels, matching the batch it will be called
            with. Held as a plain attribute rather than a buffer: it is a
            per-call binding, not state the wrapper owns or should checkpoint.
    """

    def __init__(self, net: nn.Module, labels: torch.Tensor) -> None:
        super().__init__()
        self.net = net
        self.labels = labels
        # Adopt the wrapped network's train/eval mode. `eval_mode` restores
        # whatever mode it found on the module it was handed, so a wrapper
        # left at nn.Module's default of training=True would quietly put a
        # net that was in eval mode back into training.
        self.train(net.training)


class Conditioned(_LabelBound):
    """Bind class labels to a conditional network.

    Args:
        net: conditional network taking ``(x, t, y)``.
        labels: ``(B,)`` integer labels to condition on.
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the network on the bound labels.

        Args:
            x: ``(B, C, H, W)`` latents.
            t: ``(B,)`` integer timesteps.

        Returns:
            Whatever the network predicts, unchanged.
        """
        return self.net(x, t, self.labels)


class ClassifierFreeGuidance(_LabelBound):
    """Extrapolate away from the unconditional prediction.

    The guided output is ``uncond + scale * (cond - uncond)``: at ``scale=1``
    it is the conditional prediction, at ``scale=0`` the unconditional one, and
    above 1 it sharpens class identity at the cost of diversity. Both
    predictions come from one forward pass over a doubled batch, so a step
    costs twice an unguided one rather than two dispatches' worth of latency.

    Only the mean channels are guided. When the network also emits a learned
    variance the conditional variance is kept as-is: guidance is an
    extrapolation of the *mean* prediction, and pushing a log-variance the same
    way has no interpretation and readily leaves the schedule's bracket.

    Args:
        net: conditional network taking ``(x, t, y)``.
        labels: ``(B,)`` integer labels to condition on.
        scale: guidance scale.
        num_classes: the model's class count, whose value doubles as the index
            of the null token.
    """

    def __init__(
        self, net: nn.Module, labels: torch.Tensor, scale: float, num_classes: int
    ) -> None:
        super().__init__(net, labels)
        self.scale = scale
        self.null_class = num_classes

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate conditionally and unconditionally, and extrapolate.

        Args:
            x: ``(B, C, H, W)`` latents.
            t: ``(B,)`` integer timesteps.

        Returns:
            The guided prediction, shaped like the network's own output.
        """
        null = torch.full_like(self.labels, self.null_class)
        out = self.net(
            torch.cat([x, x]),
            torch.cat([t, t]),
            torch.cat([self.labels, null]),
        )
        cond, uncond = out.chunk(2, dim=0)

        channels = x.shape[1]
        if cond.shape[1] == 2 * channels:
            cond_mean, cond_var = cond.split(channels, dim=1)
            uncond_mean, _ = uncond.split(channels, dim=1)
            guided = uncond_mean + self.scale * (cond_mean - uncond_mean)
            return torch.cat([guided, cond_var], dim=1)

        return uncond + self.scale * (cond - uncond)


def conditioned(
    net: nn.Module,
    labels: torch.Tensor | None,
    *,
    num_classes: int | None = None,
    scale: float = 1.0,
) -> nn.Module:
    """Wrap `net` so a process calling ``model(x, t)`` gets the right prediction.

    Args:
        net: the network to wrap.
        labels: ``(B,)`` integer labels, or None for an unconditional model,
            in which case `net` is handed back untouched.
        num_classes: the model's class count. Required once `scale` differs
            from 1, since guidance needs the null token's index.
        scale: guidance scale. At exactly 1 the unconditional pass would be
            multiplied out of the result, so it is skipped and the batch stays
            single-width.

    Returns:
        A module taking ``(x, t)``.

    Raises:
        ValueError: if guidance is asked for without `num_classes`.
    """
    if labels is None:
        return net
    if scale == 1.0:
        return Conditioned(net, labels)
    if num_classes is None:
        raise ValueError("guidance needs num_classes to find the null token")
    return ClassifierFreeGuidance(net, labels, scale, num_classes)


def drop_labels(labels: torch.Tensor, num_classes: int, p: float) -> torch.Tensor:
    """Replace a random fraction of labels with the null token.

    This is the whole of what training has to do differently: the same network
    learns the conditional and unconditional predictions, the second from the
    examples whose label was dropped. A model trained with ``p=0`` has never
    seen the null token, so guidance at sample time extrapolates away from an
    untrained embedding and produces noise.

    Args:
        labels: ``(B,)`` integer labels.
        num_classes: the model's class count, whose value doubles as the index
            of the null token.
        p: probability of dropping each label, independently.

    Returns:
        A new ``(B,)`` tensor. The input is returned unchanged when ``p <= 0``.
    """
    if p <= 0.0:
        return labels
    drop = torch.rand(labels.shape, device=labels.device) < p
    return labels.masked_fill(drop, num_classes)


def cycled_labels(num_images: int, num_classes: int, device: torch.device | str) -> torch.Tensor:
    """Labels cycling ``0, 1, ..., num_classes - 1`` across a batch.

    The default for a sample grid: one image per class, wrapping round, so the
    grid shows the whole label space rather than whichever class was asked for.

    Args:
        num_images: how many labels to produce.
        num_classes: the model's class count.
        device: device to build the tensor on.

    Returns:
        A ``(num_images,)`` long tensor.
    """
    return torch.arange(num_images, device=device) % num_classes
