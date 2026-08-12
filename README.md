# TinyDiffusion

[![CI](https://github.com/HilthonTT/TinyDiffusion/actions/workflows/ci.yml/badge.svg)](https://github.com/HilthonTT/TinyDiffusion/actions/workflows/ci.yml)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A PyTorch implementation of a diffusion model for image generation: a DDPM
U-Net trained on MNIST, sampled with DDIM.

> **Status:** early development. Training and sampling work end to end on
> MNIST; other datasets are not wired up yet.

## Installation

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras --dev
```

This pulls the CPU build of PyTorch. For CUDA, set `UV_TORCH_BACKEND` (e.g.
`cu124`) before syncing.

## Usage

`./run.sh` picks the project's interpreter and forwards to the CLI:

```bash
./run.sh train  --config configs/mnist.toml
./run.sh sample --checkpoint checkpoints/last.pt --num-images 8
```

Equivalently, without the wrapper:

```bash
uv run tinydiffusion train  --config configs/mnist.toml --seed 0
uv run tinydiffusion sample --checkpoint checkpoints/last.pt --num-images 8
```

Note that `python src/tinydiffusion/cli.py` does **not** work: in a `src/`
layout the package is importable only from an environment it is installed
into, and running a file by path puts that file's own directory on `sys.path`
rather than `src/`. Use `run.sh`, `uv run`, or `python -m tinydiffusion.cli`.

### Checking the pipeline quickly

`configs/mnist.toml` is sized for a GPU — 30 epochs at 64 channels over 60k
images takes hours per epoch on a CPU. `configs/smoke.toml` is the same
pipeline shrunk to finish one epoch in minutes, which is what to run when you
want to know the wiring works rather than to get good digits:

```bash
./run.sh train  --config configs/smoke.toml
./run.sh sample --checkpoint runs/smoke/checkpoints/last.pt --out runs/smoke/gen.png
```

Training writes `sample_XXXX.png` grids to `out_dir` (generated digits above a
strip of real ones, for direct comparison) and a resumable `last.pt` to
`ckpt_dir`. Pick up an interrupted run with `--resume`:

```bash
./run.sh train --config configs/mnist.toml --resume checkpoints/last.pt
```

### Configuration

Configs are TOML. Tables are cosmetic grouping only — every key is flattened
into the flat `TrainConfig` namespace, so a key must name a real field and may
appear in exactly one table. Unknown or repeated keys are errors rather than
silent no-ops.

```toml
[model]
base_channels = 64
channel_mult = [1, 2, 2]
num_res_blocks = 2
attn_resolutions = [16]

[diffusion]
num_timesteps = 1000
schedule = "cosine"     # or "linear", which uses beta_start/beta_end
```

See `configs/mnist.toml` for every field with its default, or
`TrainConfig` in `src/tinydiffusion/training/config.py` for the source of
truth. `--config` is optional; omit it to run the defaults. `--seed`,
`--device`, and `--epochs` override the file when passed.

Checkpoints embed the config they were trained with, so `sample` reconstructs
the architecture from the `.pt` alone — no need for the TOML that produced it.
Sampling always uses the EMA weights, which is what the training grids are
drawn from.

## Project layout

```
.
├── .github/
│   ├── workflows/          # CI (lint, types, test matrix, build) and release
│   ├── ISSUE_TEMPLATE/
│   ├── dependabot.yml
│   └── CODEOWNERS
├── configs/                # Training configs, versioned alongside the code
│   ├── mnist.toml          # Full run, sized for a GPU
│   └── smoke.toml          # Tiny run for checking the pipeline
├── src/tinydiffusion/
│   ├── data/               # Datasets, transforms, dataloaders
│   ├── diffusion/          # Noise schedules, forward/reverse process, samplers
│   ├── models/             # U-Net backbone, embeddings, blocks
│   ├── training/           # Config, EMA, checkpoint I/O, the MNIST loop
│   ├── utils/              # Seeding, module state helpers
│   ├── sampling.py         # Generate images from a checkpoint
│   └── cli.py              # `tinydiffusion` entry point
├── tests/                  # Mirrors the src/ tree
├── run.sh                  # Runs the CLI with the project's interpreter
├── pyproject.toml          # Deps + ruff/mypy/pytest/coverage config
└── CONTRIBUTING.md
```

A `src/` layout means tests import the installed package, not the working
directory — so a broken packaging config fails in CI instead of silently working
locally.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow. Quick version:

```bash
uv run pre-commit install
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

## License

[MIT](LICENSE)
