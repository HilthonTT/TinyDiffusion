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
    "rescale_guided",
]


def rescale_guided(guided: torch.Tensor, cond: torch.Tensor, rescale: float) -> torch.Tensor:
    """Pull an over-extrapolated guided prediction back towards the conditional scale.

    Guidance extrapolates along ``cond - uncond`` without regard for how far it
    travels, so the guided prediction's standard deviation grows roughly with
    the scale. The model was trained on targets of a fixed scale, and one that
    is too large drives the recovered x_0 to the edges of the range — flat,
    over-saturated images that the fixed ``clip_denoised`` clamp then hides
    rather than fixes. Lin et al. 2023 §3.4
    (https://arxiv.org/abs/2305.08891) rescale the guided prediction back to
    the conditional one's per-sample standard deviation, then blend, since
    going all the way back is itself too strong and flattens the detail
    guidance was asked for.

    Args:
        guided: the extrapolated prediction, ``(B, ...)``.
        cond: the conditional prediction, whose scale is the reference. Same
            shape as `guided`.
        rescale: blend factor phi in [0, 1]. 0 returns `guided` untouched; 1 is
            the fully rescaled prediction. 0.7 is the paper's recommendation.

    Returns:
        The blended prediction, shaped like `guided`.
    """
    if rescale <= 0.0:
        return guided
    # Over every dimension but the batch: the correction is per-sample, since
    # each image in the batch extrapolates its own distance.
    dims = list(range(1, guided.ndim))
    std_cond = cond.std(dim=dims, keepdim=True)
    # A guided prediction of exactly zero variance is degenerate rather than
    # impossible — a constant tensor early in a badly initialised run — and
    # dividing by it would put NaN into the chain with nothing to report it.
    std_guided = guided.std(dim=dims, keepdim=True).clamp(min=1e-12)
    return rescale * (guided * (std_cond / std_guided)) + (1.0 - rescale) * guided


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
        rescale: how much of the guided prediction's scale to correct back
            towards the conditional one; see :func:`rescale_guided`. 0 is plain
            guidance.
    """

    def __init__(
        self,
        net: nn.Module,
        labels: torch.Tensor,
        scale: float,
        num_classes: int,
        rescale: float = 0.0,
    ) -> None:
        super().__init__(net, labels)
        self.scale = scale
        self.null_class = num_classes
        self.rescale = rescale

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
            # Rescaled against the mean channels alone: the variance channels
            # are not being guided, so they are not part of the scale that
            # guidance inflated.
            guided = rescale_guided(guided, cond_mean, self.rescale)
            return torch.cat([guided, cond_var], dim=1)

        guided = uncond + self.scale * (cond - uncond)
        return rescale_guided(guided, cond, self.rescale)


def conditioned(
    net: nn.Module,
    labels: torch.Tensor | None,
    *,
    num_classes: int | None = None,
    scale: float = 1.0,
    rescale: float = 0.0,
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
        rescale: guidance rescale factor; see :func:`rescale_guided`. It has no
            effect at ``scale=1``, where the guided prediction *is* the
            conditional one and the correction is the identity, so that path
            still takes the cheap single-width branch.

    Returns:
        A module taking ``(x, t)``.

    Raises:
        ValueError: if guidance is asked for without `num_classes`, or
            `rescale` falls outside [0, 1].
    """
    if not 0.0 <= rescale <= 1.0:
        raise ValueError(f"guidance rescale must lie in [0, 1], got {rescale}")
    if labels is None:
        return net
    if scale == 1.0:
        return Conditioned(net, labels)
    if num_classes is None:
        raise ValueError("guidance needs num_classes to find the null token")
    return ClassifierFreeGuidance(net, labels, scale, num_classes, rescale)


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
