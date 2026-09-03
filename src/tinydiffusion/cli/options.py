"""Turning command line text into the values the commands run on.

Two kinds of conversion live here. The ``type=`` callables read one argument
each, and raise :class:`argparse.ArgumentTypeError` so that a bad value is
reported as the flag it came from rather than a traceback.
:func:`config_from_args` is the larger one: it folds a config file, a resumed
checkpoint, the named flags and ``--set`` into the single
:class:`~tinydiffusion.training.config.TrainConfig` a run is given.
"""

import argparse
import dataclasses
import tomllib
from typing import Any

from tinydiffusion.training.checkpoints import config_from_checkpoint
from tinydiffusion.training.config import TrainConfig, load_config
from tinydiffusion.utils.precision import DEFAULT_PRECISION, PRECISIONS

__all__ = [
    "add_precision_argument",
    "class_labels",
    "config_from_args",
    "config_override",
    "sweep_axis",
    "toml_value",
]


def class_labels(value: str) -> list[int]:
    """Parse a comma-separated list of class labels.

    Args:
        value: the raw ``--labels`` argument, e.g. ``"0,1,2"``.

    Returns:
        The labels, in the order given.

    Raises:
        argparse.ArgumentTypeError: if the list is empty or holds a non-integer.
    """
    try:
        labels = [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"labels must be whole numbers: {exc}") from exc
    if not labels:
        raise argparse.ArgumentTypeError("no labels given")
    return labels


def config_override(value: str) -> tuple[str, Any]:
    """Parse one ``--set field=value`` pair into a config field and its value.

    The value is read as a TOML value, so it types itself exactly as the same
    text would in a config file: ``lr=1e-4`` is a float, ``amp=false`` a bool,
    ``channel_mult=[1,2,2]`` a list. Anything TOML cannot parse is taken as a
    bare string, which is what makes ``dataset=cifar10`` and
    ``out_dir=runs/sweep`` work without shell-hostile quoting —
    :meth:`~tinydiffusion.training.config.TrainConfig.from_mapping` coerces
    the string to whatever the field actually holds.

    Args:
        value: the raw argument, e.g. ``"batch_size=64"``.

    Returns:
        The field name and its parsed value.

    Raises:
        argparse.ArgumentTypeError: if there is no ``=``, or the name is empty.
    """
    name, sep, raw = value.partition("=")
    name = name.strip()
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected field=value, got {value!r}")
    return name, toml_value(raw)


def toml_value(raw: str) -> Any:
    """Read one config value the way a config file would read it.

    Args:
        raw: the text on the right of an ``=``.

    Returns:
        The TOML value it denotes, or `raw` itself where TOML cannot parse it —
        the common case for paths and registry names, which
        :meth:`~tinydiffusion.training.config.TrainConfig.from_mapping` then
        coerces to whatever the field holds.
    """
    try:
        return tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        return raw


def sweep_axis(value: str) -> tuple[str, list[Any]]:
    """Parse one ``--axis field=a,b,c`` into a config field and its values.

    Each value is read exactly as ``--set`` reads one, so a sweep's literals
    and a config file's are the same literals.

    Args:
        value: the raw argument, e.g. ``"lr=1e-4,2e-4"``.

    Returns:
        The field name and the values to sweep it over, in the order given.

    Raises:
        argparse.ArgumentTypeError: if there is no ``=``, the name is empty, or
            no values follow it.
    """
    name, sep, raw = value.partition("=")
    name = name.strip()
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected field=value[,value...], got {value!r}")
    values = _axis_values(raw)
    if not values:
        raise argparse.ArgumentTypeError(f"axis {name!r} has no values: {value!r}")
    return name, values


def _axis_values(raw: str) -> list[Any]:
    """Split the right-hand side of an axis into its values.

    The whole list is first read as one TOML array, which is what lets a
    value carry a comma of its own: ``[1,2],[1,2,4]`` is two arrays, not five
    fragments. Only when that fails — bare words like ``cosine,linear``, which
    TOML will not read unquoted — does it fall back to splitting on commas
    and reading each piece as ``--set`` would.

    Args:
        raw: the text after the ``=``.

    Returns:
        The values, in the order given.
    """
    try:
        parsed = tomllib.loads(f"value = [{raw}]")["value"]
    except tomllib.TOMLDecodeError:
        return [toml_value(part.strip()) for part in raw.split(",") if part.strip()]
    return list(parsed)


def add_precision_argument(parser: argparse.ArgumentParser) -> None:
    """Give a sampling subcommand its ``--precision`` flag.

    Four subcommands draw samples and all four take the same setting, so the
    help text lives here rather than four times over.

    Args:
        parser: the subcommand parser to add the flag to.
    """
    parser.add_argument(
        "--precision",
        choices=PRECISIONS,
        default=DEFAULT_PRECISION,
        help="What to run the network in. 'fp32' is the default and the only one "
        "whose result does not depend on the GPU it ran on. 'tf32' keeps float32 "
        "storage and uses reduced-mantissa matmuls on Ampere and later; 'fp16' and "
        "'bf16' roughly halve the time a step takes on any card with tensor cores. "
        "They move a score slightly, so hold this fixed across the checkpoints "
        "being compared. Anything but 'fp32' falls back to it off CUDA.",
    )


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Assemble the config a run should use, from a file and the flags over it.

    Shared by ``train`` and ``tui``, which take the same settings and have to
    resolve them the same way: a config file, or the checkpoint being resumed,
    or the defaults — then the named flags the user actually passed, then
    ``--set``.

    Args:
        args: the parsed arguments. Flags a subcommand does not define are
            simply absent, and count as unset.

    Returns:
        The resolved configuration.

    Raises:
        ValueError: if a config file, a checkpoint or an override is unusable.
    """
    if args.config is not None:
        cfg = load_config(args.config)
    elif args.resume is not None:
        cfg = config_from_checkpoint(args.resume)
    else:
        cfg = TrainConfig()
    overrides: dict[str, Any] = {
        name: value
        for name in (
            "dataset",
            "seed",
            "device",
            "num_epochs",
            "log_dir",
            "tensorboard",
            "wandb",
            "wandb_project",
            "log_console",
            "deterministic",
        )
        if (value := getattr(args, name, None)) is not None
    }
    overrides.update(dict(getattr(args, "overrides", None) or ()))
    return TrainConfig.from_mapping({**dataclasses.asdict(cfg), **overrides})
