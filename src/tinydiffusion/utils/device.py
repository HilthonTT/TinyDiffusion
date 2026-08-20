"""Device selection, with a CPU fallback when no GPU is visible."""

import torch


def resolve_device(requested: str | None = None, *, verbose: bool = True) -> str:
    """Pick a device, degrading to the CPU rather than failing.

    A CUDA request on a machine without a visible GPU would otherwise die deep
    in the first ``.to(device)`` call, which is a confusing place to learn that
    the installed torch is a CPU-only build.

    Args:
        requested: device string such as ``"cuda"``, ``"cuda:1"`` or ``"cpu"``.
            None picks CUDA when it is available.
        verbose: print a line when a CUDA request is downgraded.

    Returns:
        A device string that is safe to allocate on.

    Raises:
        ValueError: if `requested` does not name a device torch understands.
    """
    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # torch raises RuntimeError for a typo like "gpu", which the CLI treats
        # as a crash rather than the user error it is.
        device = torch.device(requested)
    except RuntimeError as exc:
        raise ValueError(f"unknown device {requested!r}: {exc}") from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        if verbose:
            print(f"no CUDA device visible, falling back to CPU (requested {requested!r})")
        return "cpu"
    return requested


def describe_device(device: str) -> str:
    """Render a device string with its GPU model, when it names one.

    Args:
        device: a resolved device string.

    Returns:
        Something like ``"cuda (NVIDIA GeForce RTX 5060 Laptop GPU)"``, or just
        the device string when there is nothing to add.
    """
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return device
    return f"{device} ({torch.cuda.get_device_name(resolved)})"


def bf16_supported() -> bool:
    """Whether this GPU runs bfloat16 in hardware rather than emulating it.

    ``torch.cuda.is_bf16_supported`` answers a broader question than the one a
    training run is asking: it counts the emulation path, so it returns True on
    pre-Ampere cards that have no bfloat16 units at all. Emulation is not a
    fallback worth taking — measured on a Turing card at the
    ``configs/mnist.toml`` settings, an emulated bf16 step runs about 1210ms
    against fp16's 258ms and float32's 326ms, so it is nearly five times slower
    than either of the dtypes it would be chosen over.

    Returns:
        True when the device has bfloat16 hardware. False off CUDA, and False
        where torch would only emulate it.
    """
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:  # pragma: no cover - torch before the keyword existed
        return bool(torch.cuda.is_bf16_supported())


def enable_tf32() -> None:
    """Allow TF32 matmuls and convolutions on Ampere-and-later GPUs.

    Roughly free throughput for a model like this one: the reduced mantissa
    costs nothing visible in a diffusion sample, and float32 accumulation is
    unchanged. A no-op on hardware without TF32 units.
    """
    # `fp32_precision` supersedes the older `allow_tf32` flags; prefer it when
    # the installed torch has it so this does not start warning later.
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        # Present at runtime, absent from the shipped stubs.
        torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore[attr-defined]
    else:  # pragma: no cover - older torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
