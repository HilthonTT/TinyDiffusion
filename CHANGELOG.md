# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Pure float16 training (`full_fp16 = true`), the alternative to autocast: the
  U-Net's convolutions hold float16 weights and the optimiser steps a
  flattened float32 master copy of them, so the convolutions run with no
  per-operation casts. Norms, embeddings, FiLM projections and the output head
  stay in float32, and the gradient scaler is always on — without float32
  weights to fall back on there is no unscaled path. Checkpoints are written
  from the master copy, so a run in this mode produces ordinary float32
  checkpoints; only AdamW's moments cannot cross the setting, and a `--resume`
  that does says so and starts them fresh. Needs CUDA and `amp_dtype = "fp16"`.
- Gradient checkpointing (`grad_checkpoint = true`). Each ResBlock and
  attention layer drops its intermediate activations and recomputes them in
  the backward pass, which is what lets a wider model or a larger batch fit on
  a card that otherwise refuses it — roughly a third more compute per step for
  a large cut in activation memory. It changes no weights and no result, so a
  checkpoint resumes across the setting either way, and it is inert under
  `torch.no_grad`, so no sampler pays for it.
- Classifier-free guidance rescaling (`guidance_rescale`, `--guidance-rescale`,
  Lin et al. 2023 §3.4). Guidance extrapolates along `cond - uncond` without
  regard for distance, so the prediction's standard deviation grows with the
  scale; past about 3 it outgrows anything the model was trained on and the
  images come back flat and over-saturated, which the `clip_denoised` clamp
  hides rather than fixes. The correction puts the guided prediction back on
  the conditional one's per-sample standard deviation and blends, 0.7 being the
  published factor. Available on `sample`, `fid`, the sample server and the
  per-epoch training grids; 0 is the default and leaves guidance exactly as it
  was. Setting it above 0 at `guidance = 1.0` is rejected when the config is
  read, since there is no extrapolation there to correct.
- Velocity prediction (`predict = "v"`, Salimans & Ho 2022) and the
  zero-terminal-SNR schedule rescaling it pairs with (`zero_snr = true`, Lin et
  al. 2024). A schedule that stops short of zero signal leaves `x_T` holding a
  trace of the image's mean brightness — which training sees and sampling,
  starting from pure noise, never does — and the velocity target is what stays
  informative once that trace is removed. `zero_snr` with epsilon prediction is
  rejected when the config is read, since at zero terminal SNR there is no
  epsilon that says anything about `x_0`.
- Min-SNR loss weighting (`loss_weighting = "min_snr"`, `min_snr_gamma`, Hang
  et al. 2023), which clamps each timestep's weight at gamma so the nearly
  solved low-noise steps stop dominating the gradient. The weight is expressed
  in whatever space the network predicts in; the logged per-quartile losses
  stay unweighted, so they remain comparable across buckets and runs.
- Importance-sampled timesteps (`timestep_sampler = "loss_second_moment"`,
  Nichol & Dhariwal 2021) in `tinydiffusion.diffusion.timesteps`: a ten-deep
  per-timestep loss history, a proposal proportional to its RMS, and matching
  `1/(T*p)` weights so the estimator stays unbiased. Mainly for the variational
  objectives, whose per-timestep terms differ by orders of magnitude.
- DPM-Solver++(2M) sampling (`tinydiffusion.diffusion.dpm_solver`, Lu et al.
  2022) and a sampler registry (`tinydiffusion.diffusion.samplers`). The solver
  integrates the linear part of the probability-flow ODE in closed form and
  reuses the previous step's network evaluation for the second-order term, so a
  step costs what a DDIM step costs and 15-20 of them land where 50 DDIM steps
  do. Selected by the `sampler` config field and by `--sampler` on `sample` and
  `fid`; it is deterministic, so a non-zero `eta` is refused rather than
  ignored.
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
- `deterministic` config field and a matching `--deterministic` flag on
  `train`, forcing deterministic CUDA kernels and leaving the cuDNN autotuner
  off for the run.
- `generator` on the dataloader, and `train_mnist.epoch_seed`, which derives
  one epoch's shuffle seed from the run seed and the epoch index.
- A dataset registry (`tinydiffusion.data.datasets`) and a `dataset` config
  field, with a matching `--dataset` on `train`. `DatasetSpec` carries the
  channel count, native size, label space and whether a horizontal flip
  preserves the label; MNIST, Fashion-MNIST and CIFAR-10 ship registered, and
  adding another is an entry in `DATASETS` rather than an edit to five modules.
  `configs/cifar10.toml` is a worked three-channel example.
- Training-split augmentation, applied only where a spec marks a flip
  label-preserving and never to a scored split.
- `grad_accum`, running that many micro-batches per optimiser step for an
  effective batch of `batch_size * grad_accum`. Each group is averaged over the
  batches it holds, so a ragged trailing group is not a fractional update.
- `lr_schedule = "cosine"`, decaying the learning rate to zero over the run's
  optimiser steps after the warmup ramp. The two compose without a
  discontinuity where they meet. `constant`, the previous behaviour, stays the
  default.
- `betas` and `weight_decay`, and `train_mnist.lr_factor` alongside them.
- `amp_dtype`, choosing fp16 (scaled, as before) or bf16 (unscaled, no skipped
  steps, Ampere or newer, with a reported fallback to fp16 where unsupported).
- `compile`, wrapping the network in `torch.compile` for the training step
  only. The checkpoint, the EMA and the samplers keep the eager module they
  share parameters with, so a compiled run's checkpoints carry no
  `_orig_mod.` prefix.
- `channels_last`, applied to the network before the EMA is taken and to each
  batch as it lands. Worth about 11% on an RTX 5060 at the shipped MNIST
  settings, where bf16 alone is worth nothing measurable.
- A `compile` run on CUDA without Triton installed now says so at startup and
  trains eagerly, rather than failing on the first batch inside dynamo. The
  PyTorch Windows wheels do not ship Triton.
- `docs/INSTALL.md`: install, GPU verification and troubleshooting.

### Fixed

- `amp_dtype = "bf16"` no longer runs emulated on a card without bfloat16
  units. The startup check asked `torch.cuda.is_bf16_supported()`, which
  counts the emulation path and so answers True on pre-Ampere hardware, and
  the fallback it guards never fired. Measured on a Turing card at the
  `configs/mnist.toml` settings, that left a bf16 run at ~1210ms/step against
  fp16's ~258ms and float32's ~326ms — nearly five times slower than either
  dtype it would have been chosen over, with the startup line still reporting
  `amp bf16`. The check now asks for native support only.
- The AMP gradient scaler is no longer restored from a checkpoint written by a
  run that had it disabled. `GradScaler` refuses the empty state dict such a
  run stores, so an fp16 run could not resume a bf16, CPU or `amp = false`
  checkpoint even though its weights fit; the scale now simply starts fresh.
- The startup line reports the precision a run actually used. A pre-Ampere card
  asking for `bf16` is quietly given fp16, and the line said `bf16` anyway.
- `eval` scored every batch against the same noise. The seed was reapplied
  before each batch to keep the score reproducible, but with the same value
  every time, so a run covering 10,000 images averaged over a single noise
  draw per slot and carried whatever bias that one draw had. The seed is now
  offset by the batch index: still reproducible, but the noise varies across
  the split as the number implies.
- A resumed run no longer restarts the random stream. The loader's shuffle
  order was already a function of the epoch index, but the diffusion noise,
  the timesteps, dropout and label dropout all come from the global generator,
  which a fresh process seeded from `seed` alone — so epoch 5 of a `--resume`
  saw different noise than epoch 5 of a run trained straight through, even
  under `deterministic`. Checkpoints now carry the RNG state, and the training
  loop restores it. Checkpoints written before this still resume, and say that
  their stream restarts.

- `run.ps1` reported a Python that could not start as one missing the package,
  which sent you off reinstalling something that was already installed. It
  probes in two stages now, as `run.sh` already did. Both wrappers additionally
  recognise the case behind it: a `.venv` whose base interpreter has been moved
  or uninstalled — `did not find executable at ...` — which `uv sync` does not
  repair, because it reuses the existing venv rather than rebuilding it. They
  name the venv and print the two commands that do fix it.
- A resumed run replayed the batch order of a fresh one. The loader was seeded
  once at startup, so the order depended on how many epochs had run in that
  process: resuming at epoch 5 handed it epoch 0's ordering, and every later
  epoch followed suit. The order is now a function of `(seed, epoch)`, so an
  epoch draws the same batches whether it was reached by resuming or by
  training straight through.
- `train` re-enabled the cuDNN autotuner immediately after `seed_everything`
  had disabled it, so a deterministic run was not one — the autotuner is free
  to pick a different kernel on an identical input. It is now left off when
  `deterministic` is set, which is also the first time that setting is
  reachable from a config or the command line.
- The per-epoch sample grids stopped being a flipbook across a `--resume`. The
  real strip was lifted off whichever batch the loop saw first, and the batch
  order is a function of the epoch, so a resumed run compared against different
  images than the epochs before it — and, being conditional, generated on their
  labels too, which moved the class of every image in the grid. The strip is
  now read from the front of the unshuffled, unaugmented split
  (`training.train.reference_batch`), which depends on the dataset alone, as
  the fixed starting noise already did.
- `TrainConfig` accepted several settings that only failed later, or not at all.
  `ema_decay` outside [0, 1] was the quiet one: the average extrapolates away
  from the weights it follows, so the loss keeps falling while every sample,
  every `best.pt` comparison and every shipped checkpoint comes from weights
  that are diverging. `batch_size`, `num_workers`, `base_channels`,
  `channel_mult`, `num_res_blocks`, `dropout`, `num_timesteps`, `num_epochs`,
  `lr`, `grad_clip` and `ema_warmup` are now range-checked alongside the
  settings that already were, while the run is being read.
- Held-out validation drew one noise tensor per batch and reused it at every
  scored timestep, so `val/loss` estimated the objective under a single
  realisation and a draw far from the mean biased every timestep the same way.
  Each `(batch, timestep)` now draws its own noise from the same seeded
  generator, so the score stays replayable but is no longer tied to one draw.
  Values shift slightly against runs scored before this change.

### Changed

- `train --resume` without `--config` now continues the checkpoint's own
  config (`training.checkpoints.config_from_checkpoint`) instead of the
  built-in defaults, which refused every checkpoint not trained on them by way
  of a mismatch report about settings the user never asked to change. Passing
  `--config` still wins, and the individual flags still override either.
- `tinydiffusion.training.train_mnist` is split up, and the MNIST in its name is
  gone now that the loop trains whatever `DATASETS` holds:
  `train_mnist.train_mnist` is `training.train.train`, `build_model` is
  `training.model.build_model`, the checkpoint functions and their constants are
  `training.checkpoints`, and `lr_factor` is `training.lr`. Sampling, evaluation,
  FID and the server rebuild a checkpoint before loading it, and previously
  imported a training loop to do so; they now reach the builder and the
  checkpoint I/O without it. Importing the old module still works and warns; it
  goes in 0.3.0.
- `tinydiffusion.data.mnist` is now `tinydiffusion.data.datasets`, and its
  MNIST-specific names are general: `MNIST_CHANNELS` is `DatasetSpec.channels`,
  `mnist_transform`/`mnist_dataset`/`mnist_dataloader` are
  `image_transform`/`image_dataset`/`image_dataloader` and take a spec.
- `dataset` is part of `ARCHITECTURE_FIELDS`, so `--resume` refuses a
  checkpoint trained on a different one rather than failing on a channel-count
  mismatch deep inside `load_state_dict`.
- `num_classes` has to match the dataset's label space. It was previously
  unchecked, so a count below the real one trained until a batch carried a
  label past the end of the embedding table.
- `/api/status` reports the checkpoint's `dataset`.
- The optimiser is AdamW rather than Adam. At the default `weight_decay = 0.0`
  the two are the same algorithm, so nothing about an existing run changes, and
  an Adam checkpoint resumes into it unaltered.
- `fid_for_checkpoint` moves a caller-supplied extractor to the device the run
  resolved to, when it is an `nn.Module`. One built on the CPU and handed to a
  run that resolved to CUDA previously failed inside a matmul, complaining
  about `mat2` rather than about a device.
- `EMA.update` folds the average through `torch._foreach_lerp_` — the same
  arithmetic in a handful of fused kernels rather than one per parameter
  tensor — and refuses a model whose parameter count does not match.
- A `--resume` checkpoint is read and checked before the dataset is built, so a
  mismatched config fails without first waiting on a download.
- `uv sync` installs a CUDA build of PyTorch on Windows, where PyPI's wheel is
  CPU-only. `pyproject.toml` points the Windows `torch` and `torchvision`
  wheels at PyTorch's `cu132` index through an explicit `[tool.uv.sources]`
  entry, and `uv.lock` carries both builds behind a platform marker. The CUDA
  build previously had to be installed by hand, outside the lockfile, where the
  next `uv sync` — or the implicit one a bare `uv run` performs — silently
  replaced it with the CPU wheel. CI opts out with `--no-sources`.
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
