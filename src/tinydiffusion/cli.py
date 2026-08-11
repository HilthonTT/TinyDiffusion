"""Command line entry point for TinyDiffusion."""

import argparse
from collections.abc import Sequence

from tinydiffusion import __version__


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="tinydiffusion",
        description="Train and sample from a tiny diffusion model.",
    )
    parser.add_argument("--version", action="version", version=f"tinydiffusion {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a diffusion model.")
    train.add_argument("--config", required=True, help="Path to a training config file.")
    train.add_argument("--seed", type=int, default=0, help="Random seed.")

    sample = subparsers.add_parser("sample", help="Sample images from a checkpoint.")
    sample.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    sample.add_argument("--num-images", type=int, default=8, help="How many images to generate.")
    sample.add_argument("--seed", type=int, default=0, help="Random seed.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    args = build_parser().parse_args(argv)
    print(f"'{args.command}' is not implemented yet.")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
