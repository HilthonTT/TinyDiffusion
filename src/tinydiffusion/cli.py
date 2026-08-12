"""Command line entry point for TinyDiffusion."""

import argparse
import dataclasses
from collections.abc import Sequence
from pathlib import Path

from tinydiffusion import __version__
from tinydiffusion.evaluation import DEFAULT_EVAL_STEPS, evaluate_checkpoint
from tinydiffusion.sampling import sample_from_checkpoint
from tinydiffusion.training.config import TrainConfig, load_config
from tinydiffusion.training.train_mnist import train_mnist


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="tinydiffusion",
        description="Train and sample from a tiny diffusion model.",
    )
    parser.add_argument("--version", action="version", version=f"tinydiffusion {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a diffusion model.")
    train.add_argument(
        "--config", type=Path, help="Path to a training config file. Omit to use the defaults."
    )
    train.add_argument("--resume", type=Path, help="Checkpoint to continue training from.")
    train.add_argument("--seed", type=int, help="Random seed, overriding the config.")
    train.add_argument("--device", help="Device to train on, e.g. 'cuda' or 'cpu'.")
    train.add_argument("--epochs", type=int, dest="num_epochs", help="Epochs, overriding config.")

    evaluate = subparsers.add_parser("eval", help="Score a checkpoint on held-out data.")
    evaluate.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint.")
    evaluate.add_argument(
        "--split", choices=("test", "train"), default="test", help="Split to score."
    )
    evaluate.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_EVAL_STEPS,
        dest="num_steps",
        help="How many timesteps to score at.",
    )
    evaluate.add_argument("--batch-size", type=int, help="Override the checkpoint's batch size.")
    evaluate.add_argument("--data-root", type=Path, help="Override the dataset directory.")
    evaluate.add_argument(
        "--no-ema", action="store_false", dest="use_ema", help="Score the raw weights, not the EMA."
    )
    evaluate.add_argument("--seed", type=int, default=0, help="Random seed.")
    evaluate.add_argument("--device", help="Device to score on, e.g. 'cuda' or 'cpu'.")

    sample = subparsers.add_parser("sample", help="Sample images from a checkpoint.")
    sample.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint.")
    sample.add_argument("--num-images", type=int, default=8, help="How many images to generate.")
    sample.add_argument("--steps", type=int, help="DDIM steps. Defaults to the checkpoint's.")
    sample.add_argument("--eta", type=float, default=0.0, help="0 is DDIM, 1 is ancestral DDPM.")
    sample.add_argument("--out", type=Path, default=Path("contents/samples.png"), help="Output.")
    sample.add_argument("--seed", type=int, default=0, help="Random seed.")
    sample.add_argument("--device", help="Device to sample on, e.g. 'cuda' or 'cpu'.")

    return parser


def _train(args: argparse.Namespace) -> int:
    """Run the training subcommand."""
    cfg = load_config(args.config) if args.config else TrainConfig()
    # Only flags the user actually passed override the file.
    overrides = {
        name: getattr(args, name)
        for name in ("seed", "device", "num_epochs")
        if getattr(args, name) is not None
    }
    cfg = dataclasses.replace(cfg, **overrides)

    train_mnist(cfg, resume=args.resume)
    print(f"checkpoints in {cfg.ckpt_dir}, samples in {cfg.out_dir}")
    return 0


def _eval(args: argparse.Namespace) -> int:
    """Run the evaluation subcommand."""
    result = evaluate_checkpoint(
        args.checkpoint,
        split=args.split,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        data_root=args.data_root,
        use_ema=args.use_ema,
        seed=args.seed,
        device=args.device,
    )
    print(result.format())
    return 0


def _sample(args: argparse.Namespace) -> int:
    """Run the sampling subcommand."""
    out = sample_from_checkpoint(
        args.checkpoint,
        args.out,
        num_images=args.num_images,
        num_steps=args.steps,
        eta=args.eta,
        seed=args.seed,
        device=args.device,
    )
    print(f"wrote {args.num_images} images to {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    args = build_parser().parse_args(argv)
    handlers = {"train": _train, "eval": _eval, "sample": _sample}
    handler = handlers[args.command]
    try:
        return handler(args)
    except (OSError, ValueError, KeyError) as exc:
        # Bad paths and bad configs are user errors, not something to hand back
        # as a traceback.
        print(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
