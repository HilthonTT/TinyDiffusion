"""Command line entry point for TinyDiffusion."""

import argparse
import dataclasses
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tinydiffusion import __version__
from tinydiffusion.data.datasets import dataset_names
from tinydiffusion.diffusion.ddim import spacing_names
from tinydiffusion.diffusion.samplers import sampler_names
from tinydiffusion.evaluation import DEFAULT_EVAL_STEPS, evaluate_checkpoint
from tinydiffusion.interpolation import interpolate_from_checkpoint
from tinydiffusion.metrics.evaluate import DEFAULT_FID_IMAGES, fid_for_checkpoint
from tinydiffusion.metrics.kid import DEFAULT_KID_SUBSET_SIZE, DEFAULT_KID_SUBSETS
from tinydiffusion.metrics.precision_recall import DEFAULT_NEIGHBOURS
from tinydiffusion.plotting import plot_runs
from tinydiffusion.sampling import sample_from_checkpoint
from tinydiffusion.server.config import (
    DEFAULT_HOST,
    DEFAULT_IMAGE_TTL,
    DEFAULT_KEEP_IMAGES,
    DEFAULT_MAX_IMAGES,
    DEFAULT_PORT,
    ServerConfig,
)
from tinydiffusion.training.checkpoints import config_from_checkpoint
from tinydiffusion.training.config import TrainConfig, load_config
from tinydiffusion.training.train import train as train_run
from tinydiffusion.utils.precision import DEFAULT_PRECISION, PRECISIONS


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
    try:
        # A one-key document is the cheapest way to borrow TOML's own literals;
        # anything it rejects is a bare string, which is the common case for
        # paths and registry names.
        parsed = tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        return name, raw
    return name, parsed


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


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="tinydiffusion",
        description="Train and sample from a tiny diffusion model.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"tinydiffusion {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train a diffusion model.")
    train.add_argument(
        "--config",
        type=Path,
        help="Path to a training config file. Omit to use the defaults, or the "
        "settings stored in --resume's checkpoint.",
    )
    train.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint to continue training from. Its own config is used unless "
        "--config says otherwise.",
    )
    train.add_argument(
        "--dataset",
        choices=dataset_names(),
        help="Dataset to train on, overriding the config. A conditional run's "
        "num_classes has to match it.",
    )
    train.add_argument(
        "--set",
        type=config_override,
        action="append",
        dest="overrides",
        metavar="FIELD=VALUE",
        help="Override any config field, repeatable: --set lr=1e-4 --set batch_size=64 "
        "--set sample_spacing=quadratic. Values are read as TOML, so quoting rules "
        "match the config file; a bare word is a string. Applied last, so it wins "
        "over the other flags.",
    )
    train.add_argument("--seed", type=int, help="Random seed, overriding the config.")
    train.add_argument("--device", help="Device to train on, e.g. 'cuda' or 'cpu'.")
    train.add_argument("--epochs", type=int, dest="num_epochs", help="Epochs, overriding config.")
    train.add_argument("--log-dir", type=Path, help="Directory for metrics.jsonl and TB events.")
    # store_true with default None so an unpassed flag leaves the config alone.
    train.add_argument(
        "--tensorboard",
        action="store_true",
        default=None,
        help="Also write TensorBoard events. Needs the 'tracking' extra.",
    )
    train.add_argument(
        "--deterministic",
        action="store_true",
        default=None,
        help="Force deterministic kernels and disable the cuDNN autotuner. "
        "Reproducible to the bit, at a noticeable throughput cost.",
    )
    train.add_argument(
        "--quiet",
        action="store_false",
        default=None,
        dest="log_console",
        help="Do not print the per-epoch metrics table.",
    )

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

    fid = subparsers.add_parser("fid", help="Score a checkpoint's samples against real data.")
    fid.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint.")
    fid.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_FID_IMAGES,
        help="Samples to draw, and real images to compare against. "
        "Below a few thousand the score is mostly its own bias.",
    )
    fid.add_argument(
        "--split", choices=("train", "test"), default="train", help="Real split to compare against."
    )
    fid.add_argument("--batch-size", type=int, help="Override the checkpoint's batch size.")
    fid.add_argument("--data-root", type=Path, help="Override the dataset directory.")
    fid.add_argument("--steps", type=int, help="Denoising steps. Defaults to the checkpoint's.")
    fid.add_argument(
        "--sampler",
        choices=sampler_names(),
        help="Sampler to draw with, overriding the checkpoint's. It moves the "
        "score, so hold it fixed across the checkpoints being compared.",
    )
    fid.add_argument(
        "--spacing",
        choices=spacing_names(),
        help="Timestep spacing, overriding the checkpoint's. Like --sampler it "
        "moves the score, so hold it fixed across the checkpoints being compared.",
    )
    fid.add_argument("--eta", type=float, default=0.0, help="0 is DDIM, 1 is ancestral DDPM.")
    fid.add_argument(
        "--guidance",
        type=float,
        help="Classifier-free guidance scale. Defaults to the checkpoint's; "
        "worth sweeping, since FID usually bottoms out above 1.",
    )
    fid.add_argument(
        "--guidance-rescale",
        type=float,
        help="How much of the scale guidance inflates to correct back, in [0, 1]. "
        "Defaults to the checkpoint's; sweep it jointly with --guidance.",
    )
    fid.add_argument(
        "--no-ema",
        action="store_false",
        dest="use_ema",
        help="Sample the raw weights, not the EMA.",
    )
    fid.add_argument(
        "--no-cache",
        action="store_false",
        dest="cache",
        help="Recompute the real images' features instead of reusing the cached "
        "ones. They do not depend on the checkpoint, so a sweep normally wants "
        "the cache; this is for forcing a rebuild.",
    )
    fid.add_argument(
        "--kid",
        action="store_true",
        help="Also report the Kernel Inception Distance. Unlike FID it is unbiased, "
        "so it stays comparable between scores taken over different image counts, "
        "which is what makes --num-images in the low thousands worth reading. It "
        "comes with a spread, so two checkpoints can be told apart from noise.",
    )
    fid.add_argument(
        "--kid-subsets",
        type=int,
        default=DEFAULT_KID_SUBSETS,
        metavar="N",
        help="Subsets to average the KID over.",
    )
    fid.add_argument(
        "--kid-subset-size",
        type=int,
        default=DEFAULT_KID_SUBSET_SIZE,
        metavar="N",
        help="Images per KID subset, per side, capped at the smaller set. The "
        "reported spread is a spread over subsets of this size, so hold it fixed "
        "across the checkpoints being compared.",
    )
    fid.add_argument(
        "--precision-recall",
        action="store_true",
        help="Also estimate manifold precision and recall: what fraction of the "
        "samples look real, and what fraction of the real data the samples cover. "
        "They split a bad score into its two causes, which no single number can. "
        "Cost is quadratic in --num-images.",
    )
    fid.add_argument(
        "--neighbours",
        type=int,
        default=DEFAULT_NEIGHBOURS,
        metavar="K",
        help="Neighbours defining each precision/recall manifold ball.",
    )
    fid.add_argument("--seed", type=int, default=0, help="Random seed.")
    fid.add_argument("--device", help="Device to score on, e.g. 'cuda' or 'cpu'.")
    add_precision_argument(fid)

    serve = subparsers.add_parser("serve", help="Serve a checkpoint over HTTP.")
    serve.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint.")
    serve.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Interface to bind. Loopback by default; the API is unauthenticated, "
        "so only widen it behind something that is not.",
    )
    serve.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind.")
    serve.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES,
        help="Largest num_images a single request may ask for.",
    )
    serve.add_argument(
        "--image-dir", type=Path, help="Where to write PNGs. Defaults to a temp dir."
    )
    serve.add_argument(
        "--image-ttl",
        type=float,
        default=DEFAULT_IMAGE_TTL,
        metavar="SECONDS",
        help="How long a rendered PNG is kept before it is swept. 0 keeps them forever.",
    )
    serve.add_argument(
        "--keep-images",
        type=int,
        default=DEFAULT_KEEP_IMAGES,
        metavar="N",
        help="PNGs retained regardless of age, newest first. 0 for no cap.",
    )
    serve.add_argument(
        "--cors-origin",
        action="append",
        dest="cors_origins",
        metavar="ORIGIN",
        help="Origin allowed to call the API from a browser. Repeatable; omit to leave CORS off.",
    )
    serve.add_argument(
        "--no-ema",
        action="store_false",
        dest="use_ema",
        help="Serve the raw weights, not the EMA.",
    )
    serve.add_argument("--device", help="Device to sample on, e.g. 'cuda' or 'cpu'.")
    add_precision_argument(serve)

    sample = subparsers.add_parser("sample", help="Sample images from a checkpoint.")
    sample.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint.")
    sample.add_argument("--num-images", type=int, default=8, help="How many images to generate.")
    sample.add_argument("--steps", type=int, help="Denoising steps. Defaults to the checkpoint's.")
    sample.add_argument(
        "--sampler",
        choices=sampler_names(),
        help="Sampler to draw with, overriding the checkpoint's. 'dpmpp' is "
        "DPM-Solver++(2M), which needs roughly a third of the steps 'ddim' does.",
    )
    sample.add_argument(
        "--spacing",
        choices=spacing_names(),
        help="Which subsequence of the training schedule to visit, overriding the "
        "checkpoint's. 'quadratic' packs the steps towards t=0 and is worth trying "
        "whenever --steps is low; it costs no extra network evaluations.",
    )
    sample.add_argument("--eta", type=float, default=0.0, help="0 is DDIM, 1 is ancestral DDPM.")
    sample.add_argument(
        "--labels",
        type=class_labels,
        help="Comma-separated classes to generate, cycled over the grid, e.g. '3' or '0,1,2'. "
        "Conditional checkpoints only; the default is one image per class.",
    )
    sample.add_argument(
        "--guidance",
        type=float,
        help="Classifier-free guidance scale. 1.0 is the plain conditional prediction; "
        "higher sharpens class identity. Defaults to the checkpoint's.",
    )
    sample.add_argument(
        "--guidance-rescale",
        type=float,
        help="How much of the scale guidance inflates to correct back, in [0, 1]. "
        "0.7 is the published value, and worth setting above --guidance 3, where "
        "plain guidance starts washing images out. Defaults to the checkpoint's.",
    )
    sample.add_argument("--out", type=Path, default=Path("contents/samples.png"), help="Output.")
    sample.add_argument("--seed", type=int, default=0, help="Random seed.")
    sample.add_argument("--device", help="Device to sample on, e.g. 'cuda' or 'cpu'.")
    add_precision_argument(sample)

    interpolate = subparsers.add_parser(
        "interpolate", help="Walk between two latents and sample every point."
    )
    interpolate.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint.")
    interpolate.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Points along the walk, counting both ends.",
    )
    interpolate.add_argument(
        "--denoise-steps",
        type=int,
        dest="num_steps",
        help="Denoising steps per image. Defaults to the checkpoint's.",
    )
    interpolate.add_argument(
        "--sampler",
        choices=sampler_names(),
        help="Sampler to draw with, overriding the checkpoint's.",
    )
    interpolate.add_argument(
        "--spacing",
        choices=spacing_names(),
        help="Timestep spacing, overriding the checkpoint's.",
    )
    interpolate.add_argument(
        "--labels",
        type=class_labels,
        help="Class to hold fixed across the walk, e.g. '7'. Conditional checkpoints "
        "only; the default is class 0. More than one is cycled, which moves two "
        "things at once and is rarely what you want.",
    )
    interpolate.add_argument(
        "--guidance",
        type=float,
        help="Classifier-free guidance scale. Defaults to the checkpoint's.",
    )
    interpolate.add_argument(
        "--guidance-rescale",
        type=float,
        help="How much of the scale guidance inflates to correct back, in [0, 1]. "
        "Defaults to the checkpoint's.",
    )
    interpolate.add_argument(
        "--seed-start", type=int, default=0, help="Seed for the latent the walk starts at."
    )
    interpolate.add_argument(
        "--seed-end", type=int, default=1, help="Seed for the latent it ends at."
    )
    interpolate.add_argument(
        "--out", type=Path, default=Path("contents/interpolation.png"), help="Output."
    )
    interpolate.add_argument("--device", help="Device to sample on, e.g. 'cuda' or 'cpu'.")
    add_precision_argument(interpolate)

    plot = subparsers.add_parser("plot", help="Draw a run's metrics as a figure.")
    plot.add_argument(
        "runs",
        type=Path,
        nargs="+",
        metavar="RUN",
        help="Run log directories, or metrics.jsonl files. More than one draws "
        "them on shared axes, which is how a sweep is compared.",
    )
    plot.add_argument(
        "--out",
        type=Path,
        default=Path("contents/metrics.png"),
        help="Image to write. The extension picks the format, so .svg works too.",
    )
    plot.add_argument("--dpi", type=int, default=120, help="Resolution for raster formats.")

    return parser


def _train(args: argparse.Namespace) -> int:
    """Run the training subcommand."""
    if args.config is not None:
        cfg = load_config(args.config)
    elif args.resume is not None:
        # A checkpoint carries the config it was trained with, so a bare
        # --resume continues that run. Falling back to the defaults instead
        # would refuse every checkpoint not trained on them, by way of a
        # mismatch report about settings the user never asked to change.
        cfg = config_from_checkpoint(args.resume)
    else:
        cfg = TrainConfig()
    # Only flags the user actually passed override the file.
    overrides: dict[str, Any] = {
        name: getattr(args, name)
        for name in (
            "dataset",
            "seed",
            "device",
            "num_epochs",
            "log_dir",
            "tensorboard",
            "log_console",
            "deterministic",
        )
        if getattr(args, name) is not None
    }
    # Applied last, so `--set` wins wherever it and a named flag spell the same
    # field. It is the escape hatch: whatever the file and the flags worked out
    # between them, this is the value.
    overrides.update(dict(args.overrides or ()))
    # Rebuilt through from_mapping rather than dataclasses.replace: it is what
    # knows a --set of a path or a tuple field arrives as a string or a list,
    # and it is what reports an unknown field name as such rather than as a
    # TypeError about an unexpected keyword argument.
    cfg = TrainConfig.from_mapping({**dataclasses.asdict(cfg), **overrides})

    train_run(cfg, resume=args.resume)
    print(f"checkpoints in {cfg.ckpt_dir}, samples in {cfg.out_dir}, metrics in {cfg.log_dir}")
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


def _fid(args: argparse.Namespace) -> int:
    """Run the scoring subcommand: FID always, KID and precision/recall on request."""
    result = fid_for_checkpoint(
        args.checkpoint,
        num_images=args.num_images,
        split=args.split,
        batch_size=args.batch_size,
        data_root=args.data_root,
        num_steps=args.steps,
        eta=args.eta,
        sampler=args.sampler,
        spacing=args.spacing,
        guidance=args.guidance,
        guidance_rescale=args.guidance_rescale,
        use_ema=args.use_ema,
        cache=args.cache,
        seed=args.seed,
        device=args.device,
        kid=args.kid,
        kid_subsets=args.kid_subsets,
        kid_subset_size=args.kid_subset_size,
        precision_recall=args.precision_recall,
        neighbours=args.neighbours,
        sample_precision=args.precision,
    )
    print(result.format())
    return 0


def _interpolate(args: argparse.Namespace) -> int:
    """Run the interpolate subcommand."""
    out = interpolate_from_checkpoint(
        args.checkpoint,
        args.out,
        steps=args.steps,
        num_steps=args.num_steps,
        sampler=args.sampler,
        spacing=args.spacing,
        labels=args.labels,
        guidance=args.guidance,
        guidance_rescale=args.guidance_rescale,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        device=args.device,
        precision=args.precision,
    )
    print(f"wrote {out}")
    return 0


def _plot(args: argparse.Namespace) -> int:
    """Run the plot subcommand."""
    out = plot_runs(args.runs, args.out, dpi=args.dpi)
    print(f"wrote {out}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    """Run the serve subcommand."""
    config = ServerConfig(
        checkpoint=args.checkpoint,
        host=args.host,
        port=args.port,
        device=args.device,
        use_ema=args.use_ema,
        max_images=args.max_images,
        image_dir=args.image_dir,
        cors_origins=tuple(args.cors_origins or ()),
        image_ttl=args.image_ttl,
        keep_images=args.keep_images,
        precision=args.precision,
    )
    if not config.checkpoint.is_file():
        # uvicorn would otherwise bind the port and only fail during startup,
        # which reads as a server crash rather than a bad path.
        raise ValueError(f"no such checkpoint: {config.checkpoint}")

    # Imported here so `serve` is the only command that needs the extra.
    from tinydiffusion.server.app import serve as run_server

    print(f"serving {config.checkpoint} on http://{config.host}:{config.port}")
    run_server(config)
    return 0


def _sample(args: argparse.Namespace) -> int:
    """Run the sampling subcommand."""
    out = sample_from_checkpoint(
        args.checkpoint,
        args.out,
        num_images=args.num_images,
        num_steps=args.steps,
        eta=args.eta,
        sampler=args.sampler,
        spacing=args.spacing,
        labels=args.labels,
        guidance=args.guidance,
        guidance_rescale=args.guidance_rescale,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
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
    handlers = {
        "train": _train,
        "eval": _eval,
        "fid": _fid,
        "interpolate": _interpolate,
        "plot": _plot,
        "sample": _sample,
        "serve": _serve,
    }
    handler = handlers[args.command]
    try:
        return handler(args)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        # Bad paths, bad configs and a missing optional extra are user errors,
        # not something to hand back as a traceback.
        print(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
