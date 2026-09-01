"""What each subcommand does, and the dispatch that picks one.

Every handler takes the parsed namespace and returns a process exit code, so
:func:`main` is a lookup and a call. The error handling sits there rather than
in the handlers: a missing checkpoint, an unreadable config and a Ctrl+C all
reach the terminal as one line, not a traceback.
"""

import argparse
import dataclasses
import sys
from collections.abc import Sequence

from tinydiffusion.cli.options import config_from_args
from tinydiffusion.cli.parser import build_parser
from tinydiffusion.evaluation import evaluate_checkpoint
from tinydiffusion.interpolation import interpolate_from_checkpoint
from tinydiffusion.metrics.evaluate import fid_for_checkpoint
from tinydiffusion.plotting import plot_runs
from tinydiffusion.sampling import sample_from_checkpoint
from tinydiffusion.server.config import ServerConfig
from tinydiffusion.sweep import run_sweep, sweep_points, sweep_summary
from tinydiffusion.training.train import train as train_run

__all__ = ["main"]


def _train(args: argparse.Namespace) -> int:
    """Run the training subcommand."""
    cfg = config_from_args(args)
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
        bpd=args.bpd,
        bpd_images=args.bpd_images,
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
        sfid=args.sfid,
        inception_score=args.inception_score,
        is_splits=args.is_splits,
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


def _tui(args: argparse.Namespace) -> int:
    """Run the dashboard subcommand."""
    from tinydiffusion.tui import run_tui

    cfg = config_from_args(args)
    cfg = dataclasses.replace(cfg, log_console=False)
    run_tui(cfg, resume=args.resume, autostart=args.start)
    return 0


def _sweep(args: argparse.Namespace) -> int:
    """Run the sweep subcommand."""
    args.resume = None
    base = config_from_args(args)
    points = sweep_points(base, args.axes, args.out_root)

    print(f"{len(points)} points under {args.out_root}")
    for point in points:
        print(f"  {point.name}")
    if args.dry_run:
        return 0

    runs = list(run_sweep(points, train=train_run, skip_existing=args.skip_existing))
    print()
    print(sweep_summary(runs))
    return 0 if all(run.ok for run in runs) else 1


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
        max_inflight=args.max_inflight,
        image_dir=args.image_dir,
        cors_origins=tuple(args.cors_origins or ()),
        image_ttl=args.image_ttl,
        keep_images=args.keep_images,
        precision=args.precision,
    )
    if not config.checkpoint.is_file():
        raise ValueError(f"no such checkpoint: {config.checkpoint}")

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
        batch_size=args.batch_size,
        num_steps=args.steps,
        eta=args.eta,
        sampler=args.sampler,
        spacing=args.spacing,
        labels=args.labels,
        guidance=args.guidance,
        guidance_rescale=args.guidance_rescale,
        save_individual=args.save_individual,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
    )
    print(f"wrote {args.num_images} images to {out}")
    if args.save_individual:
        print(f"and one file each, {out.with_name(f'{out.stem}_0000{out.suffix}')} onwards")
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
        "sweep": _sweep,
        "serve": _serve,
        "tui": _tui,
    }
    handler = handlers[args.command]
    try:
        return handler(args)
    except (OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
