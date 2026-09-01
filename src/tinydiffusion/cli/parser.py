"""The argument parser, one subcommand at a time.

``tinydiffusion`` has nine subcommands and most of them take a dozen flags, so
a single builder would be several hundred lines with no seam in it. Each
command defines its own here, and :func:`build_parser` is the assembly —
which is also what keeps a flag two commands share, like ``--precision``,
visibly shared rather than written out twice.
"""

import argparse
from pathlib import Path

from tinydiffusion import __version__
from tinydiffusion.cli.options import (
    add_precision_argument,
    class_labels,
    config_override,
    sweep_axis,
)
from tinydiffusion.data.datasets import dataset_names
from tinydiffusion.diffusion.ddim import spacing_names
from tinydiffusion.diffusion.samplers import sampler_names
from tinydiffusion.evaluation import DEFAULT_BPD_IMAGES, DEFAULT_EVAL_STEPS
from tinydiffusion.metrics.evaluate import DEFAULT_FID_IMAGES
from tinydiffusion.metrics.inception_score import DEFAULT_IS_SPLITS
from tinydiffusion.metrics.kid import DEFAULT_KID_SUBSET_SIZE, DEFAULT_KID_SUBSETS
from tinydiffusion.metrics.precision_recall import DEFAULT_NEIGHBOURS
from tinydiffusion.server.config import (
    DEFAULT_HOST,
    DEFAULT_IMAGE_TTL,
    DEFAULT_KEEP_IMAGES,
    DEFAULT_MAX_IMAGES,
    DEFAULT_MAX_INFLIGHT,
    DEFAULT_PORT,
)

__all__ = ["build_parser"]

type Subparsers = argparse._SubParsersAction[argparse.ArgumentParser]


def _add_train(subparsers: Subparsers) -> None:
    """Define the ``train`` subcommand: Train a diffusion model.

    Args:
        subparsers: the top-level subparser action to register on.
    """
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
    train.add_argument(
        "--tensorboard",
        action="store_true",
        default=None,
        help="Also write TensorBoard events. Needs the 'tracking' extra.",
    )
    train.add_argument(
        "--wandb",
        action="store_true",
        default=None,
        help="Also stream metrics to Weights & Biases. Needs the 'tracking' extra "
        "and an authenticated wandb; WANDB_MODE=offline records locally to sync later.",
    )
    train.add_argument(
        "--wandb-project",
        help="W&B project to log into. Ignored without --wandb.",
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


def _add_eval(subparsers: Subparsers) -> None:
    """Define the ``eval`` subcommand: Score a checkpoint on held-out data.

    Args:
        subparsers: the top-level subparser action to register on.
    """
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
    evaluate.add_argument(
        "--bpd",
        action="store_true",
        help="Also evaluate the full variational bound, in bits per dimension. "
        "Unlike the loss it is comparable against published likelihoods and "
        "across parameterisations, and unlike the loss it costs a network "
        "evaluation per timestep per image, so it covers --bpd-images rather "
        "than the split. Needs a checkpoint trained with a non-default "
        "predict, variance or objective; plain DDPM defines no bound.",
    )
    evaluate.add_argument(
        "--bpd-images",
        type=int,
        default=DEFAULT_BPD_IMAGES,
        metavar="N",
        help="Images to estimate the bound over. Rounded up to a whole batch.",
    )
    evaluate.add_argument("--seed", type=int, default=0, help="Random seed.")
    evaluate.add_argument("--device", help="Device to score on, e.g. 'cuda' or 'cpu'.")


def _add_fid(subparsers: Subparsers) -> None:
    """Define the ``fid`` subcommand: Score a checkpoint's samples against real data.

    Args:
        subparsers: the top-level subparser action to register on.
    """
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
        "--sfid",
        action="store_true",
        help="Also report the spatial FID: the same distance taken in an "
        "intermediate, unpooled Inception feature map. FID's features are "
        "spatially averaged, so it cannot see an image whose parts are each "
        "plausible and jointly arranged wrong; this is the reading that can. "
        "It rides along on the same Inception pass and costs almost nothing.",
    )
    fid.add_argument(
        "--inception-score",
        action="store_true",
        help="Also report the Inception Score. It reads only the generated "
        "samples, so it says nothing about whether they resemble your data — "
        "worth little on MNIST, and free to compute.",
    )
    fid.add_argument(
        "--is-splits",
        type=int,
        default=DEFAULT_IS_SPLITS,
        metavar="N",
        help="Chunks to average the Inception Score over. The score depends on "
        "the chunk size, so hold this fixed across the checkpoints being compared.",
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


def _add_serve(subparsers: Subparsers) -> None:
    """Define the ``serve`` subcommand: Serve a checkpoint over HTTP.

    Args:
        subparsers: the top-level subparser action to register on.
    """
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
        "--max-inflight",
        type=int,
        default=DEFAULT_MAX_INFLIGHT,
        metavar="N",
        help="Sampling requests accepted at once, being drawn or waiting. "
        "Further ones are refused with a 503 rather than queued.",
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


def _add_sample(subparsers: Subparsers) -> None:
    """Define the ``sample`` subcommand: Sample images from a checkpoint.

    Args:
        subparsers: the top-level subparser action to register on.
    """
    sample = subparsers.add_parser("sample", help="Sample images from a checkpoint.")
    sample.add_argument("--checkpoint", type=Path, required=True, help="Trained checkpoint.")
    sample.add_argument("--num-images", type=int, default=8, help="How many images to generate.")
    sample.add_argument(
        "--batch-size",
        type=int,
        help="Images to draw at a time. Peak memory follows this rather than --num-images, "
        "so it is what makes a large --num-images fit. Each image keeps the latent and the "
        "label it would have had unsplit, so at the default --eta 0 the pictures are the "
        "same. Defaults to drawing them all in one go.",
    )
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
    sample.add_argument(
        "--save-individual",
        action="store_true",
        help="Also write each image on its own beside the grid, named after it — "
        "'samples.png' gives 'samples_0000.png' and so on.",
    )
    sample.add_argument("--seed", type=int, default=0, help="Random seed.")
    sample.add_argument("--device", help="Device to sample on, e.g. 'cuda' or 'cpu'.")
    add_precision_argument(sample)


def _add_interpolate(subparsers: Subparsers) -> None:
    """Define the ``interpolate`` subcommand: Walk between two latents.

    Args:
        subparsers: the top-level subparser action to register on.
    """
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


def _add_tui(subparsers: Subparsers) -> None:
    """Define the ``tui`` subcommand: Train in a terminal dashboard.

    Args:
        subparsers: the top-level subparser action to register on.
    """
    tui = subparsers.add_parser(
        "tui",
        help="Train in a terminal dashboard: live loss, progress and samples.",
    )
    tui.add_argument(
        "--config",
        type=Path,
        help="Path to a training config file. Omit to use the defaults, or the "
        "settings stored in --resume's checkpoint.",
    )
    tui.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint to continue training from. Its own config is used unless "
        "--config says otherwise.",
    )
    tui.add_argument(
        "--dataset",
        choices=dataset_names(),
        help="Dataset to train on, overriding the config.",
    )
    tui.add_argument(
        "--set",
        type=config_override,
        action="append",
        dest="overrides",
        metavar="FIELD=VALUE",
        help="Override any config field, repeatable. Read exactly as `train --set` is.",
    )
    tui.add_argument("--seed", type=int, help="Random seed, overriding the config.")
    tui.add_argument("--device", help="Device to train on, e.g. 'cuda' or 'cpu'.")
    tui.add_argument("--epochs", type=int, dest="num_epochs", help="Epochs, overriding config.")
    tui.add_argument("--log-dir", type=Path, help="Directory for metrics.jsonl and TB events.")
    tui.add_argument(
        "--start",
        action="store_true",
        help="Begin training as soon as the dashboard opens, rather than waiting for 's'.",
    )


def _add_sweep(subparsers: Subparsers) -> None:
    """Define the ``sweep`` subcommand: Train one config over a grid of hyperparameters.

    Args:
        subparsers: the top-level subparser action to register on.
    """
    sweep = subparsers.add_parser(
        "sweep",
        help="Train one config over a grid of hyperparameters, one directory per point.",
    )
    sweep.add_argument(
        "--config",
        type=Path,
        help="Config every point starts from. Omit for the defaults.",
    )
    sweep.add_argument(
        "--axis",
        type=sweep_axis,
        action="append",
        dest="axes",
        required=True,
        metavar="FIELD=A,B,C",
        help="A field to vary and the values to vary it over, repeatable: "
        "--axis lr=1e-4,2e-4 --axis sample_spacing=uniform,quadratic. Every "
        "combination is run, so that is four points and four training runs. "
        "Values are read exactly as --set reads one.",
    )
    sweep.add_argument(
        "--set",
        type=config_override,
        action="append",
        dest="overrides",
        metavar="FIELD=VALUE",
        help="Override a field for every point, repeatable. This is where the "
        "settings a sweep holds fixed go, so they stay out of the directory names.",
    )
    sweep.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/sweep"),
        help="Directory the points are created under. Each gets its own "
        "metrics.jsonl, checkpoints and samples, so `plot <root>/*` compares them.",
    )
    sweep.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the grid and what each point would be, without training anything.",
    )
    sweep.add_argument(
        "--skip-existing",
        action="store_true",
        help="Leave a point alone if its directory already holds metrics, which "
        "is how an interrupted sweep is resumed without redoing what finished.",
    )
    sweep.add_argument("--seed", type=int, help="Random seed, overriding the config.")
    sweep.add_argument("--device", help="Device to train on, e.g. 'cuda' or 'cpu'.")
    sweep.add_argument("--epochs", type=int, dest="num_epochs", help="Epochs, overriding config.")


def _add_plot(subparsers: Subparsers) -> None:
    """Define the ``plot`` subcommand: Draw a run's metrics as a figure.

    Args:
        subparsers: the top-level subparser action to register on.
    """
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


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser.

    Returns:
        A parser covering every subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="tinydiffusion",
        description="Train and sample from a tiny diffusion model.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"tinydiffusion {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)
    for add in (
        _add_train,
        _add_eval,
        _add_fid,
        _add_serve,
        _add_sample,
        _add_interpolate,
        _add_tui,
        _add_sweep,
        _add_plot,
    ):
        add(subparsers)
    return parser
