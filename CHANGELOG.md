# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffolding: `src/` layout, uv-managed environment, ruff, mypy, pytest.
- GitHub Actions CI (lint, type-check, test matrix, build) and release workflow.
- Dependabot for `uv` and `github-actions`.
- `tinydiffusion` CLI skeleton with `train` and `sample` subcommands.
- Metric tracking (`tinydiffusion.utils.tracking`): a per-epoch console table,
  `metrics.jsonl`, and optional TensorBoard events via the `tracking` extra.
  Training logs loss, per-timestep-quartile loss, gradient norm, learning rate,
  EMA decay, AMP scale, skipped-step rate and throughput.
- `log_dir`, `log_console`, `log_jsonl` and `tensorboard` config fields, with
  matching `--log-dir`, `--tensorboard` and `--quiet` flags on `train`.
- Class-conditional training and classifier-free guidance
  (Ho & Salimans 2022). `num_classes` gives the U-Net a label embedding with a
  reserved null token; `class_dropout` trains that token by replacing a
  fraction of labels with it; `guidance` extrapolates away from it at sample
  time. `tinydiffusion.diffusion.guidance` packages both as wrappers over the
  network, so no process or sampler needed changing. Both shipped configs
  train on MNIST's ten digits.
- `--labels` and `--guidance` on `sample`, for generating a chosen digit and
  choosing how hard to insist on it. Conditional sample grids are laid out one
  class per column, and the per-epoch training grids now generate on the real
  strip's own labels, pairing each generated digit with a real one of the same
  class.
- `DDPM.loss_terms`, which returns the per-image loss and its timesteps
  alongside the scalar, and `EMA.current_decay`.
- `-V` as a short alias for `--version`.

### Changed

- `DDPM.forward` and `DDPM.loss_terms` take a `model` argument, matching
  `GaussianDiffusion` and letting either process be trained through a
  conditioning wrapper.
- Conditional and unconditional checkpoints are not interchangeable: the label
  embedding is part of the state dict, so `--resume` cannot carry a run across
  a change to `num_classes`.
- The package version is read from `src/tinydiffusion/version.py`, which
  `pyproject.toml` now builds the distribution version from — so a source
  checkout and an installed wheel report the same string.

[Unreleased]: https://github.com/HilthonTT/TinyDiffusion/compare/main...HEAD
