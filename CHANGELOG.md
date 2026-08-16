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
- FID (`tinydiffusion.metrics`) and a `fid` subcommand. `FeatureStats`
  accumulates the mean and covariance of Inception-v3 activations in a single
  pass, so the score covers more images than fit in memory; `compute_fid`
  evaluates the distance through a symmetric matrix square root rather than the
  eigenvalues of the raw covariance product, which keeps it real-valued and
  stable on the singular covariances that small sample counts produce. The
  report flags when either side has too few images for the score to be
  comparable.
- An HTTP sampling server (`tinydiffusion.server`) and a `serve` subcommand,
  behind the optional `server` extra. `POST /api/sample` runs the DDIM chain on
  a checkpoint loaded once at startup and returns a URL for the grid;
  `GET /api/status` reports what is loaded and what the request defaults are.
  Requests are serialised behind a lock and run off the event loop, the bind
  defaults to loopback since nothing authenticates, and served filenames are
  matched against the uuid names the server itself issues so a percent-encoded
  traversal cannot escape the image directory.
- `noise` on `ddim_sample` and `save_samples`, letting a caller supply the
  starting x_T. Training uses it to hold the per-epoch sample grids on one set
  of latents, so the sequence of PNGs shows the same images sharpening rather
  than an unrelated draw each epoch.
- `DDPM.loss_terms`, which returns the per-image loss and its timesteps
  alongside the scalar, and `EMA.current_decay`.
- `-V` as a short alias for `--version`.
- Per-epoch held-out validation (`tinydiffusion.training.validation`), scoring
  the EMA weights on a fixed slice of the test split at a pinned timestep grid
  with pinned noise, so `val/loss` moves only with the weights. Configured by
  `val_every`, `val_steps` and `val_batches`, and logged as `val/loss` and
  `val/best_loss`.
- `best.pt`, the lowest-scoring epoch so far, kept alongside `last.pt` and
  controlled by `keep_best`. `keep_last` additionally retains a rolling window
  of numbered `epoch_NNNN.pt` snapshots. Neither existed before, so a run that
  peaked mid-training had already overwritten its best weights.
- `train_mnist.check_resume_compatible`, which refuses a `--resume` whose
  checkpoint was trained with a different architecture, schedule or
  parameterisation, naming the field that changed. Previously this surfaced as
  a raw `load_state_dict` size-mismatch dump, or — for a differing schedule —
  not at all.
- `ddim_sample(generator=...)`, for a reproducible sample without touching the
  global RNG.
- `--image-ttl` and `--keep-images` on `serve`, with matching `ServerConfig`
  fields, bounding the rendered-PNG directory by age and by count.

### Changed

- A `Ctrl+C` save now writes `interrupted.pt` rather than overwriting
  `last.pt`. An interrupt lands mid-epoch, so its weights are worse than the
  ones the previous epoch finished on, and it is recorded under that previous
  epoch's number — writing it to `last.pt` replaced a good checkpoint with a
  worse one carrying the same label.
- The server's `seed` request field seeds a request-local generator instead of
  calling `seed_everything`. A client's seed no longer reseeds the process, so
  it cannot reach into concurrent requests or outlive its own.
- The server sweeps rendered PNGs it has issued once they pass `image_ttl` or
  fall outside `keep_images`. Nothing deleted them before, so a long-lived
  server grew its image directory without bound.
- `evaluation.eval_timesteps` moved to `tinydiffusion.training.validation`,
  shared with the in-loop validation; it is still importable from
  `tinydiffusion.evaluation`.
- `sampling._grid_width` is now public as `sampling.grid_width`, since the
  server needs the same grid layout the CLI produces.
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
