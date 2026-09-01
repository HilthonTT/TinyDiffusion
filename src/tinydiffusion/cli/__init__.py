"""Command line entry point for TinyDiffusion.

``tinydiffusion <command>`` resolves to :func:`main`, which is the whole of the
public surface here. Underneath it, :mod:`~tinydiffusion.cli.parser` declares
the commands and their flags, :mod:`~tinydiffusion.cli.options` turns the text
they collect into values, and :mod:`~tinydiffusion.cli.commands` runs them.

The parsing helpers stay re-exported because they are the seam the tests reach
for: what a flag means is checkable without running the command behind it.
"""

from tinydiffusion.cli.commands import main
from tinydiffusion.cli.options import (
    add_precision_argument,
    class_labels,
    config_from_args,
    config_override,
    sweep_axis,
    toml_value,
)
from tinydiffusion.cli.parser import build_parser

__all__ = [
    "add_precision_argument",
    "build_parser",
    "class_labels",
    "config_from_args",
    "config_override",
    "main",
    "sweep_axis",
    "toml_value",
]
