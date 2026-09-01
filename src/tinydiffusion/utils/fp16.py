"""Half-precision weights with a float32 master copy.

The strategy openai/guided-diffusion trains with, and an alternative to the
autocast path in :mod:`torch.amp`. Autocast leaves the weights in float32 and
casts each operand on the way into a kernel; this keeps the *weights*
themselves in float16 and holds one flattened float32 copy — the master params
— for the optimiser to step. The network's activations, its gradients and every
convolution it runs are float16 throughout, with no per-operation casts.

Three things make that safe, and all three are load-bearing:

* The optimiser updates the master copy, never the float16 weights. A single
  Adam step is routinely smaller than the gap between two adjacent float16
  values, so applying it to a float16 weight rounds straight back to where it
  started and the model stops learning.
* Gradients are scaled before the backward pass and unscaled before the step,
  because float16 flushes to zero below roughly 6e-8 and diffusion gradients
  live down there. That scaling is :class:`torch.amp.GradScaler`'s job, and it
  is shared with the autocast path rather than reimplemented here.
* Normalisation layers, the timestep and label embeddings, and the FiLM
  projections stay in float32 — see
  :func:`~tinydiffusion.models.blocks.group_norm` and
  :meth:`~tinydiffusion.models.unet.UNet.convert_to_fp16`. Only the
  convolutions are converted, which is where nearly all of the compute is and
  none of the numerical fragility.

The flattening is what makes the master copy cheap to maintain: the whole
network is one contiguous tensor, so moving gradients onto it and weights back
off it is a pair of kernel launches per step rather than one per parameter.

Adapted from openai/guided-diffusion.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

__all__ = [
    "convert_module_to_f16",
    "convert_module_to_f32",
    "make_master_params",
    "master_params_to_model_params",
    "master_params_to_state_dict",
    "model_grads_to_master_grads",
    "model_params_to_master_params",
    "unflatten_master_params",
    "zero_grad",
]


def convert_module_to_f16(module: nn.Module) -> None:
    """Convert a primitive module's weights to float16, in place.

    Written for :meth:`torch.nn.Module.apply`, so it is handed every module in
    a tree and ignores the ones it does not recognise. Convolutions only:
    normalisation layers and the small embedding MLPs are where half precision
    actually costs accuracy, and they hold a negligible share of the weights.

    Args:
        module: a module from the tree being converted. Anything that is not a
            convolution is left alone.
    """
    if isinstance(module, nn.Conv1d | nn.Conv2d | nn.Conv3d):
        module.weight.data = module.weight.data.half()
        if module.bias is not None:
            module.bias.data = module.bias.data.half()


def convert_module_to_f32(module: nn.Module) -> None:
    """Convert a primitive module's weights back to float32, in place.

    Undoes :func:`convert_module_to_f16`. The round trip is lossy — the low
    mantissa bits are gone — so this restores the dtype, not the values. What
    makes that safe during a run is that the master copy, not the module, is
    the authority on the weights.

    Args:
        module: a module from the tree being converted. Anything that is not a
            convolution is left alone.
    """
    if isinstance(module, nn.Conv1d | nn.Conv2d | nn.Conv3d):
        module.weight.data = module.weight.data.float()
        if module.bias is not None:
            module.bias.data = module.bias.data.float()


def make_master_params(model_params: Iterable[torch.Tensor]) -> list[nn.Parameter]:
    """Copy model parameters into one flat, full-precision parameter.

    Args:
        model_params: the network's parameters, in a fixed order. Everything
            here reads them in that same order, so it has to be the order
            :meth:`torch.nn.Module.parameters` yields rather than, say, a set.

    Returns:
        A single-element list holding the flattened float32 copy. A list
        because that is what an optimiser's ``params`` argument wants, and
        because it leaves room for the per-group split a larger model wants.

    Raises:
        ValueError: if there are no parameters to copy. Flattening an empty
            list gives an empty tensor that the optimiser then steps forever
            without complaint.
    """
    tensors = [param.detach().float() for param in model_params]
    if not tensors:
        raise ValueError("cannot build master parameters from a model with no parameters")

    master = nn.Parameter(_flatten_dense_tensors(tensors))  # type: ignore[no-untyped-call]
    master.requires_grad = True
    return [master]


def model_grads_to_master_grads(
    model_params: Iterable[torch.Tensor], master_params: list[nn.Parameter]
) -> None:
    """Copy the model's gradients onto the master parameters.

    Assigns rather than accumulates, so gradient accumulation has to happen on
    the model's own float16 gradients: call this once per optimiser step, after
    the last micro-batch's backward pass.

    The gradients are still scaled at this point. Unscaling them is
    :meth:`torch.amp.GradScaler.unscale_`'s job, and it has to happen on the
    far side of this copy, because the optimiser only ever sees the master.

    Args:
        model_params: the network's parameters, in the order
            :func:`make_master_params` read them.
        master_params: the list :func:`make_master_params` returned.
    """
    grads = [
        param.grad.detach().float()
        if param.grad is not None
        else torch.zeros_like(param, dtype=torch.float32)
        for param in model_params
    ]
    master_params[0].grad = _flatten_dense_tensors(grads)  # type: ignore[no-untyped-call]


@torch.no_grad()
def master_params_to_model_params(
    model_params: Iterable[torch.Tensor], master_params: list[nn.Parameter]
) -> None:
    """Copy the stepped master parameters back into the model.

    Args:
        model_params: the network's parameters, in the order
            :func:`make_master_params` read them. Written to in place, and
            rounded back down to whatever dtype each one holds.
        master_params: the list :func:`make_master_params` returned.
    """
    model_params = list(model_params)

    for param, master_param in zip(
        model_params, unflatten_master_params(model_params, master_params), strict=True
    ):
        param.detach().copy_(master_param)


@torch.no_grad()
def model_params_to_master_params(
    model_params: Iterable[torch.Tensor], master_params: list[nn.Parameter]
) -> None:
    """Copy the model's parameters into the master copy, the other way round.

    The inverse of :func:`master_params_to_model_params`, and only worth
    calling when something outside the training step has written to the
    network — restoring a checkpoint, above all. Doing it while the weights are
    still float32 is what lets a resumed run continue from the checkpoint's own
    values rather than from the checkpoint rounded to float16.

    Args:
        model_params: the network's parameters, in the order
            :func:`make_master_params` read them.
        master_params: the list :func:`make_master_params` returned, overwritten.
    """
    model_params = list(model_params)

    for master_param, param in zip(
        unflatten_master_params(model_params, master_params), model_params, strict=True
    ):
        master_param.copy_(param)


def unflatten_master_params(
    model_params: Iterable[torch.Tensor], master_params: list[nn.Parameter]
) -> tuple[torch.Tensor, ...]:
    """Split the flat master parameter back into per-parameter views.

    Args:
        model_params: the network's parameters, read for their shapes.
        master_params: the list :func:`make_master_params` returned.

    Returns:
        One float32 view per model parameter, in the same order. Views into the
        flat tensor rather than copies, so writing to one writes through — and
        saving one saves the whole flat tensor behind it. See
        :func:`master_params_to_state_dict` for the copy that avoids that.
    """
    return _unflatten_dense_tensors(  # type: ignore[no-untyped-call]
        master_params[0].detach(), tuple(model_params)
    )


def master_params_to_state_dict(
    model: nn.Module, master_params: list[nn.Parameter]
) -> dict[str, torch.Tensor]:
    """Build an ordinary float32 state dict from the master copy.

    What a run in this mode should checkpoint. ``model.state_dict()`` hands
    back the float16 weights, which are a rounded copy of these — so a resume
    would restart from weights slightly worse than the ones the run actually
    had, and every downstream reader would get a checkpoint whose dtype depends
    on how the run happened to be trained.

    Args:
        model: the network the master copy was built from.
        master_params: the list :func:`make_master_params` returned.

    Returns:
        The model's state dict with every parameter replaced by its float32
        master value. Buffers are passed through untouched.
    """
    state = model.state_dict()
    params = list(model.parameters())
    for (name, _), value in zip(
        model.named_parameters(), unflatten_master_params(params, master_params), strict=True
    ):
        state[name] = value.clone()
    return state


def zero_grad(model_params: Iterable[torch.Tensor]) -> None:
    """Zero the model's gradients without releasing them.

    Deliberately not ``set_to_none=True``: :func:`model_grads_to_master_grads`
    reads every gradient positionally, and dropping the tensors only makes it
    allocate the zeros it would otherwise have found already there.

    Args:
        model_params: the network's parameters.
    """
    for param in model_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
