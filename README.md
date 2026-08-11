# TinyDiffusion

[![CI](https://github.com/HilthonTT/TinyDiffusion/actions/workflows/ci.yml/badge.svg)](https://github.com/HilthonTT/TinyDiffusion/actions/workflows/ci.yml)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A PyTorch implementation of a diffusion model for image generation.

> **Status:** early development. The training and sampling pipelines are not
> implemented yet — the CLI currently exits with a "not implemented" message.

## Installation

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras --dev
```

This pulls the CPU build of PyTorch. For CUDA, set `UV_TORCH_BACKEND` (e.g.
`cu124`) before syncing.

## Usage

```bash
uv run tinydiffusion train  --config configs/mnist.toml --seed 0
uv run tinydiffusion sample --checkpoint checkpoints/last.pt --num-images 8
```

## Project layout

```
.
├── .github/
│   ├── workflows/          # CI (lint, types, test matrix, build) and release
│   ├── ISSUE_TEMPLATE/
│   ├── dependabot.yml
│   └── CODEOWNERS
├── configs/                # Training configs, versioned alongside the code
├── src/tinydiffusion/
│   ├── data/               # Datasets, transforms, dataloaders
│   ├── diffusion/          # Noise schedules, forward/reverse process, samplers
│   ├── models/             # U-Net backbone, embeddings, blocks
│   ├── utils/              # Seeding, logging, checkpoint I/O
│   └── cli.py              # `tinydiffusion` entry point
├── tests/                  # Mirrors the src/ tree
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
