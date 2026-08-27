# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Two more samplers, `heun` and `plms`, joining `ddim` and `dpmpp` in the
  registry and so available to `--sampler`, the `sampler` config field, the
  per-epoch grids, `fid`, `interpolate` and the server alike. Both are
  deterministic ODE solvers that beat DDIM's first-order step, and they differ
  in what they spend to do it, which is the reason to have both.
  `heun` takes the DDIM step provisionally, evaluates the network again at
  where it landed, and re-takes the step along the average of the two
  directions — second order, at two network calls a step rather than one. What
  that buys over `dpmpp`, which reaches second order for free by reusing the
  previous step's evaluation, is being correct from the *first* step: a
  multistep method has no history until its second, and a very short chain is
  mostly first steps. `num_steps` therefore counts steps and not evaluations,
  and the docs say so — the comparison against the other three only means
  anything at a fixed evaluation budget.
  `plms` (Liu et al. 2022) goes the other way and remembers: an
  Adams-Bashforth formula fits a cubic through the last four noise estimates,
  which are already paid for, and extrapolates. Fourth order at one call a
  step, the cheapest order on offer, with the order ramping 1, 2, 3, 4 as the
  buffer fills — so it wants 20 steps or more before the ramp has paid for
  itself. The estimate stored is the measured one rather than the extrapolated
  one, or the error would compound.
  Only `ddim` remains stochastic; the other three refuse a non-zero `--eta`
  rather than ignoring it, since the probability-flow ODE has no noise term to
  scale.

- `eval --bpd`, evaluating the full variational bound in bits per dimension.
  The held-out loss is whatever the run was trained on, which makes it useless
  for comparing a v-prediction model against an epsilon one or either against a
  published number; every diffusion model defines the same bound, and
  `GaussianDiffusion.calc_bpd_loop` has implemented it all along with nothing
  reaching it. The prior term is reported beside the total rather than folded
  into it: it depends on the schedule alone and no amount of training moves it,
  so a reader comparing two checkpoints can see how much of the difference was
  ever theirs to move.
  Scored unclipped, since clamping the implied `x_0` tightens the number
  without it still being an upper bound — which is the only thing that makes it
  comparable with anyone else's. It walks every timestep of the training
  schedule, so it costs `num_timesteps` network evaluations per image against
  the couple of dozen the loss spends; `--bpd-images` bounds that and defaults
  to 128. A checkpoint on the default parameterisation is served by the plain
  DDPM implementation, which defines no bound, and the command says which
  settings would give it one rather than failing on a missing method.

- `fid --sfid`, the spatial FID (Nash et al. 2021). FID's features are pooled —
  averaged over the image — which makes them a summary of what an image
  contains and blind to where: a model that draws perfect strokes and arranges
  them into something that is not a digit scores well on a metric that cannot
  see the arrangement. sFID is the same distance in an unpooled reading of the
  same network, the first seven channels of an intermediate feature map kept
  spatially, for 2023 dimensions against FID's 2048.
  It rides along on the Inception pass FID is already making rather than
  running a second one, which matters on the generated side where there is no
  second pass to run: the samples are produced lazily and are gone by the time
  the first accumulator has seen them. Its reference half is a different
  feature space, so it caches separately under the same key plus `_spatial`;
  the payload's own feature width is what a load checks, so a suffix collision
  could only waste a read, never produce a wrong score.

- `fid --inception-score`, with `--is-splits`. The one metric here that never
  looks at a real image: it asks the classifier whether each sample is
  confidently some ImageNet class and whether the samples between them cover
  many, and reports the exponentiated KL between the two. That is both its
  appeal — no reference set — and its limit, and the documentation is blunt
  about the limit: on MNIST it is close to meaningless, since handwritten
  digits are not an ImageNet class. It is here because it is free on a pass
  already running Inception over every sample.
  Getting it meant keeping Inception's 1000-way classifier rather than
  discarding it. `InceptionFeatures` now moves `fc` aside instead of replacing
  it, and `analyse` returns all three readings — pooled, spatial and class
  probabilities — from one forward pass, with the spatial map captured by a
  hook on the block that produces it rather than by re-implementing
  torchvision's forward.

- `sweep`, training one config over a grid of hyperparameters. `--set` already
  made a sweep a shell loop, and the moment one of the directory overrides is
  forgotten every point writes over the last one's record of itself — which is
  the comparison the sweep was for. This is that loop with `log_dir`,
  `ckpt_dir` and `out_dir` set per point, each named after the values that
  distinguish it, so `plot runs/sweep/*` draws the whole grid on shared axes
  with a legend that reads itself.
  `--axis field=a,b,c` is repeatable and every combination is run; values are
  read exactly as `--set` reads one, and `--set` still works alongside for the
  settings a sweep holds fixed, which keeps them out of the directory names.
  A combination that is not a valid config is rejected while the grid is being
  expanded rather than after the points before it have already run, and an axis
  over one of the three directory fields is refused outright. `--dry-run`
  prints the grid without training anything, `--skip-existing` leaves a point
  whose directory already holds metrics alone and reads its numbers, and a
  point that fails is reported rather than ending the sweep — five good runs
  are not worth losing to one bad combination — with the whole command exiting
  non-zero if any did. `Ctrl+C` still ends the sweep rather than one point. It
  finishes by printing what each point reached, ranked by best `val/loss`.

- A Weights & Biases logging backend, `--wandb` and `--wandb-project`, joining
  the console, JSONL and TensorBoard sinks. The one that leaves the machine,
  which is the reason to want it: a run on a remote box is watchable from a
  laptop, and several runs land on shared axes without anyone copying
  `metrics.jsonl` around. The run is named after `log_dir` so the dashboard
  lines up with the checkpoints on disk, and the training config is sent once
  so the sweep view can group and filter by hyperparameter; nothing else goes.
  A send that fails warns and continues rather than raising — the numbers are
  already on disk, and losing an epoch of training to a network blip would make
  the backend cost more than it is worth. Needs the `tracking` extra, which now
  carries `wandb` alongside `tensorboard`, and rank 0 alone logs under
  `torchrun`.

- `dataset = "folder"`, training on a directory of your own images rather than
  a packaged download. Two layouts work and which one you have is inferred:
  loose images in `data_root` are unlabelled data, and one subdirectory per
  class is conditional data, numbered by sorted name. Everything is resized and
  centre-cropped to `image_size` and converted to `folder_channels`, so the
  images need not be square, uniformly sized, or all the same mode — a
  directory of photographs rarely is, and the U-Net's input width is fixed by
  the config rather than by whichever file was read first. `configs/folder.toml`
  is a starting point for both layouts.
  The facts a packaged dataset ships with are declared rather than detected:
  `folder_channels` (1 for greyscale, 3 for RGB), `folder_hflip` and
  `folder_holdout`, plus the existing `num_classes`. The config therefore
  resolves without reading the directory, which is what keeps a checkpoint
  loadable and samplable on a machine that never had the images; the
  declaration is checked against the disk when the folder is first opened, so a
  `num_classes` that disagrees with the layout is a message naming both counts
  rather than an index error at the embedding table. `folder_channels` joins
  `dataset` in the fields `--resume` refuses to see change, since it is the
  width of every tensor in the state dict.
  A folder carries no train/test split and `val_every`, `eval` and `fid` all
  need one. `train/` and `test/` (or `val/`) subdirectories are used verbatim
  where they exist; otherwise `folder_holdout` of the images are held back,
  chosen by hashing each image's path rather than by cutting a sorted list —
  adding one photo then moves that photo alone between the splits, where an
  index-based cut would reshuffle everything after it and quietly promote
  already-scored images into the training set. The hash is over the POSIX
  relative path, so the split is the same on Windows and Linux. A `train/` with
  no held-out directory beside it is an error rather than a silent fallback to
  the hash split, which would score the held-out loss on training images and
  never say so.
  Non-images, hidden directories and subdirectories holding no images are all
  ignored — which is what stops the `fid_cache/` that `fid` writes into
  `data_root` from becoming a class — and a directory mixing loose images with
  class subdirectories is rejected rather than guessed at.

- `sample --batch-size`, splitting a draw into chunks. A sampler runs one
  reverse chain over the whole batch it is handed, so the memory a draw needs
  followed `--num-images` directly and a few hundred images at once was an
  out-of-memory error rather than a slow run — on the very command whose job is
  to produce images in bulk. `fid` has batched its generation all along; this
  gives `sample` the same thing.
  The split is a memory knob and not a sampling one: every image's starting
  latent is drawn before the first chunk and handed out in order, and the label
  vector is sliced with the chunk, so image `i` gets the latent and the class it
  would have had unsplit. Two caveats, both documented: a positive `--eta` makes
  the sampler itself stochastic and its per-step noise does follow the split,
  and on CUDA the convolutions pick their algorithm by batch shape, so two
  splits agree to a pixel of rounding rather than byte-for-byte. Unset, the
  whole draw is one chunk and the images are exactly what they were.

- `sample --save-individual`, writing each image beside the grid and named after
  it — `samples.png` gives `samples_0000.png` onwards. A grid is for looking at;
  anything downstream of a generated set wants the files.

- Multi-GPU training, over `torch.distributed`. One process per GPU, launched
  by `torchrun --nproc_per_node=N -m tinydiffusion train ...`: every rank holds
  a complete copy of the network and draws a disjoint shard of each epoch, and
  `DistributedDataParallel` averages the gradients during the backward pass, so
  an epoch is still one pass over the dataset rather than N passes over all of
  it. Nothing in the config turns this on — the launcher sets `RANK`,
  `WORLD_SIZE` and `LOCAL_RANK`, and their absence is what an ordinary
  single-process run looks like, so the path every existing test takes is
  untouched. `--nproc_per_node=1` is deliberately not a group.
  The new `tinydiffusion.training.distributed` holds the whole of it: the
  process-group lifecycle, the rank facts, and the four collectives the loop
  needs. On one process a `Distributed` is rank 0 of 1, every `is_main` guard
  is true and every collective returns its argument untouched, which is what
  keeps the training loop from branching on whether it is distributed at all.
  Rank 0 alone writes `metrics.jsonl`, the sample grids, `last.pt` and
  `best.pt`, and alone draws the progress bar and prints the startup lines. The
  metrics it records are the group's rather than its own shard's: the loss is
  all-reduced before it is buffered, the timestep quartiles are totalled once
  per epoch, and `time/images_per_second` covers every rank, so it is
  comparable against a single-GPU run. The startup line reports the effective
  batch — `batch_size * grad_accum * world_size` — because that is what an
  optimiser step now averages over, and it is a real change to the run rather
  than only to its throughput.
  The checkpoints are ordinary ones. DDP shares its parameters with the eager
  network exactly as `compile` does, so the EMA, the checkpoint and every
  sampler go on using the unwrapped module and no key carries a `module.`
  prefix — a four-GPU run's `best.pt` resumes on one. `compile` composes with
  it, compiling the wrapper rather than the inner module so Dynamo can still
  overlap the all-reduces with the backward pass.
  Ctrl+C still asks. The launcher signals each process separately, so the ranks
  can see it a batch or two apart; the group agrees on the flag at a fixed
  batch cadence, rank 0 asks the question, and the answer is broadcast — the
  alternative being the first rank to leave the loop stranding the others at
  the next all-reduce for the whole hour-long timeout. An accumulated group
  reduces once rather than `grad_accum` times, via `no_sync` on every
  micro-batch but the one that applies.
  Verified by `tests/test_distributed.py`, which runs a real two-process group
  on the CPU over gloo: the ranks shard the data disjointly, end on
  bit-identical weights, and write exactly one set of files. The NCCL path and
  any scaling figure are unmeasured — the machine behind this has one GPU — and
  USAGE.md says so where it documents the feature.

- `python -m tinydiffusion`, running the same CLI as the console script. It
  exists because a launcher that starts the processes itself needs a module to
  hand them rather than a script on the PATH, which is what the `torchrun`
  invocation above uses.

- `RESULTS.md`: what the shipped configs actually score, with the exact
  commands, the checkpoint, the hardware and the run length behind each number.

- `tinydiffusion tui`, a terminal dashboard that trains the model and shows it
  happening: the resolved plan, epoch and batch progress with an ETA,
  throughput, sparklines of train and validation loss, the loss split by
  timestep quartile as four bars on one scale, and each epoch's sample grid
  drawn in the terminal — two pixels to a character cell as an upper half
  block, which is enough resolution at 32px to watch a 7 become a 7. Keys are
  `s` start, `x` stop, `l` log, `d` theme, `q` quit. Built on Textual, behind a
  new `tui` extra; without it the command reports the extra in one line the way
  `plot` and `serve` do.
  Stopping with `x` is the Ctrl+C path without the prompt — which has nobody to
  answer it while a display owns the terminal — so it stops at a batch
  boundary, where the model, optimiser and EMA agree, and writes
  `interrupted.pt` for `--resume`. Training runs on a worker thread and the
  display on the event loop; batch updates are throttled, so a slow terminal
  cannot stall the run.
- `tinydiffusion.training.observer`, the seam the dashboard hangs on:
  `TrainObserver`, `TrainPlan` and `BatchProgress`. Passing an observer to
  `train` redirects the lines it would have printed, replaces the tqdm bar, and
  lets the watcher stop the run; passing nothing leaves all three exactly as
  they were. Epoch metrics deliberately do not come through it — they already
  have a fan-out in `LoggerBackend`, and `RunLogger.for_run` now takes `extra`
  backends so a watcher registers as one rather than being given a second route
  to the same numbers. `save_samples` returns the path it wrote, so a watcher
  need not reconstruct the filename.

- `--precision` on every command that samples — `sample`, `interpolate`, `fid`
  and `serve` — taking `fp32`, `tf32`, `fp16` or `bf16`. Sampling is where a
  diffusion model's arithmetic is (a `fid` over 10,000 images at 50 steps with
  guidance is a million network evaluations) and all of it ran in float32
  before this. Measured on a Turing card at the `configs/mnist.toml` geometry,
  `fp16` draws 16.4 img/s against float32's 10.9, a 1.51x speedup, and moves a
  pixel by 0.09 of a 255-level on average. The default stays `fp32` on all
  four: it is bit-for-bit what those commands did before, and precision moves a
  score the way `--sampler` and `--steps` do, so `fid` records it and names it
  in the report whenever it is not float32. Everything falls back with a
  printed line rather than silently — to `fp32` off CUDA, and `bf16` to `fp16`
  on a card that only emulates bfloat16, which is the fallback a training run
  already made.
  Half precision carries the memory format with it: the network runs in NHWC,
  because tensor cores read that layout and cuDNN handed NCHW transposes per
  convolution instead. On that card the forward pass alone is 1.12x faster in
  NCHW `fp16` against 2.36x in NHWC, and NHWC float32 is *slower* than NCHW
  float32 — so it is coupled to the dtype rather than offered as its own
  switch. Guidance keeps extrapolating and rescaling in float32, since the
  wrapper sits under the conditioning rather than over it, and `fid`'s feature
  extractor stays in float32 whatever this is set to: it is the instrument the
  score is measured with, and the cached reference features were computed with
  it.

- A third timestep spacing, `karras` (Karras et al., 2022). `uniform` and
  `quadratic` both space the sampling steps by index; this one rewrites the
  forward process as `x_0 + sigma * eps` and spaces them evenly along the EDM
  ramp in `sigma^(1/7)` instead — evenly in what the denoiser sees rather than
  in a proxy for it. Measured on the shipped MNIST model at 1,000 images and
  equal network evaluations, it roughly halves KID against `uniform` (0.00963
  against 0.01360 at 12 NFEs, 0.00471 against 0.00720 at ~20) and loses to
  `quadratic` (0.00452 and 0.00273), every gap being several times its own
  spread. It does not honour `--steps` on a cosine schedule: that schedule ends
  at a sigma around 20,000 against the 80 the ramp was designed for, so part of
  the ramp falls inside a handful of timesteps and rounding collapses them — 20
  requested steps come back as 12, and 40 as 22. Ask for double what you want,
  or use a `linear` schedule, where 20 are 20.
- `interpolate`: a spherical walk between the latents two seeds draw, sampled
  at every point and written as a strip. A grid says what a model can draw; a
  walk says whether the space between two of its samples is populated, or
  whether it snaps from one mode to another with nothing in between. The path
  is slerp rather than a straight line because a latent lives on a thin shell
  at radius `sqrt(d)`, and the midpoint of a chord between two of them sits
  well inside it, on latents the model never saw. `--labels` holds the class
  fixed so the strip has exactly one thing moving along it.
- The `TimestepSpacing` protocol now takes the schedule's `alphabar` as a
  keyword. The index-based spacings accept and ignore it; a spacing defined on
  noise level has no way to work without it.
- KID (Kernel Inception Distance) on `fid`, behind `--kid`. FID fits a
  2048-dimensional Gaussian to each feature set, and a covariance estimated
  from fewer samples than it has dimensions is singular — an upward bias whose
  size depends on the sample count, so a FID over 1,000 images is not a noisier
  estimate of the FID over 50,000 but a different number. KID has no Gaussian
  in it: an unbiased kernel distance whose expected value does not move with
  the count, which is what makes a score affordable between checkpoints worth
  reading at all. It reports a spread across subsets alongside the mean, so two
  checkpoints closer together than the noise can be seen not to have been told
  apart. `--kid-subsets` and `--kid-subset-size` tune it.
- Manifold precision and recall on `fid`, behind `--precision-recall`
  (Kynkaanniemi et al., 2019). A model drawing beautiful images of three digits
  and one drawing all ten badly can score the same FID, and they call for
  opposite fixes. Precision is the fraction of generated images inside the real
  data's manifold, recall the fraction of real images inside the generated
  one — so a bad score splits into "not realistic" and "does not cover".
  Guidance moves them in opposite directions, which is the clearest reading
  either number gives. `--neighbours` sets the k the manifolds are built from.
- `tinydiffusion.metrics.features.FeatureBank`, the accumulator those two need:
  the feature vectors kept rather than folded into moments as they go by. Its
  moments are formed on demand, so a score that only wants a FID never pays for
  them and neither does a bank restored from cache. Memory is linear in the
  image count, about 8 KB an image, which is the trade both metrics require and
  the reason they are opt-in.
- The reference-feature cache learned a second kind of entry, holding a bank
  rather than moments, written only when a run asked for the metrics that need
  it. Same key with a `_features` suffix, so the two sit beside each other; a
  moments entry cannot stand in for a feature one.
- `plot`: a run's `metrics.jsonl` as a figure — losses, the four timestep
  quartiles, learning rate, gradient norm and throughput, each panel dropped if
  the run logged nothing for it. Several runs on shared axes compares a sweep.
  Needs the new `plots` extra (matplotlib). The per-epoch numbers have been
  written since logging landed and nothing read them back; a table per epoch is
  the wrong shape for a question about a trend.
- `read_metrics`, which reads `metrics.jsonl` back as one record per step. Every
  record now also carries a `session` number, counting how many times the file
  has been opened for appending, and the reader keeps the newest session for
  each step. A resumed run appends a second copy of every epoch it replays, so
  the raw file has more lines than the run has epochs and plotting it straight
  through shows the loss doubling back on itself; both copies stay on disk,
  since the first run did measure them, and the reader is where the run as it
  now stands comes from.
- Selectable sampling timestep spacing (`sample_spacing`, `--spacing`). The
  quadratic subsequence had been implemented and tested since the DDIM sampler
  landed but no config or flag could reach it, so every sampler ran uniform.
  `quadratic` packs the steps towards `t = 0`, where a short chain has least
  room to correct itself, for the same number of network evaluations — the
  DDIM paper's finding on CIFAR-10 at low step counts. Registered the same way
  samplers are, so both `ddim` and `dpmpp` take it, and available on `sample`,
  `fid`, the sample server and the per-epoch training grids. `uniform` remains
  the default, since the two are hard to tell apart above about 50 steps.
- A disk cache for the reference half of a FID score
  (`tinydiffusion.metrics.cache`). The real images' features depend on the
  dataset, split, resolution and image count and on nothing else — not the
  checkpoint, not any sampling setting — so sweeping `--guidance` over five
  values was pushing 50,000 real images through Inception-v3 to compute one
  number five times. They are now computed once and kept under
  `<data_root>/fid_cache`, keyed on every input that moves them, which makes a
  stale read a miss rather than a wrong score; an unreadable or truncated entry
  is treated as absent. Roughly halves every score after the first, and
  `--no-cache` opts out. Entries are about 33 MB each, since the covariance is
  2048 x 2048 in the float64 the accumulation needs.
- `--set field=value` on `train`, repeatable, reaching every config field
  without editing a file — which is what makes a sweep a shell loop rather than
  a directory of near-identical TOMLs. Values are parsed as TOML, so they type
  themselves exactly as the same text would in a config file, and anything TOML
  cannot read is taken as a bare string so paths and registry names need no
  quoting. Unknown field names are refused and the result is validated as a
  file's would be. Applied after the named flags, so it wins where both spell
  the same field.
- An end-to-end smoke job in CI: `configs/smoke.toml` trained for one epoch,
  then sampled with both samplers and both spacings, scored, and resumed. The
  unit tests mock the dataloader and never run the CLI, so nothing else in CI
  would notice a config field that stopped reaching the trainer or a subcommand
  wired to the wrong argument. The generated grid is uploaded as an artifact.
- `FeatureStats.state_dict` / `from_state_dict`, the round trip the FID cache
  is built on. The raw moments are what is stored, so a restored accumulator
  can still be extended with `merge` or `update`.
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
- `docs/ARCHITECTURE.md`: how the model and the codebase are put together —
  the shape of a run, a module map, the forward process, the backbone, the
  parameterisation choices, conditioning, sampling, EMA, DDP, and the
  metrics, plus an order to read the source in.

### Changed

- USAGE.md is now an index over nine pages under `docs/usage/`, one per topic:
  installing, running the CLI, training and checkpoints, metrics and logging,
  sampling, evaluation, serving, configuration, and troubleshooting. At 1,778
  lines it had become a document nobody read linearly and in-page search was
  the only way to navigate it — and a table of contents twenty entries long is
  the symptom, not the fix. `USAGE.md` keeps its path and every link to it
  still resolves; what was a heading anchor is now a page. Section anchors are
  unchanged within their new pages, so only the file part of a cross-reference
  moved, and README.md, RESULTS.md and `docs/ARCHITECTURE.md` were repointed
  accordingly.

- A checkpoint loaded for sampling no longer builds its U-Net with gradient
  checkpointing, whatever the run that produced it used. Every consumer of
  `load_for_sampling` — `sample`, `eval`, `fid`, `interpolate` and the server —
  runs under `no_grad`, where the blocks fall through to the plain call
  already, so no result changes and nothing gets faster; it just stops a model
  that can never take a backward pass from carrying wrappers built for one. The
  config handed back still reports the setting the checkpoint was trained with.

- `lr_warmup` documents that it counts *applied* optimiser steps. Under
  `amp_dtype = "fp16"` a step whose gradients overflowed is skipped and does
  not advance the ramp, which is the intent — the ramp exists for exactly those
  steps — but it means the field is not reliably the first `lr_warmup` batches
  of a run, and the docstring said "optimiser steps" without saying which kind.

- The README is half its former length and no longer duplicates USAGE.md. It
  had grown into a second usage guide — a paragraph per subcommand, each of
  them already documented at greater length one file over, which meant two
  places to keep in step and neither being the obvious one to read. What is
  left is what a landing page owes a reader: what the project is, a quickstart
  that runs, a table of the eight subcommands linking into the section of
  USAGE.md that documents each, and the map of the other documents. The
  "How it works" section moved to the new `docs/ARCHITECTURE.md` rather than
  being cut, and gained a diagram of a run, a module-by-module table of where
  everything lives, and a suggested order to read the source in.

- The training dashboard (`tui`) has been rebuilt. The two sparklines are now
  one chart: `train/loss` and `val/loss` drawn as braille lines on a single
  labelled axis. Braille puts a 2x4 grid of dots in every character cell, so an
  eight-row chart is thirty-two pixels tall — enough to see a validation curve
  stop following the training curve down, which two sparklines each normalised
  to their own range structurally cannot show. The arithmetic lives in the new
  `tinydiffusion.tui.chart`, which imports neither Textual nor Rich for the
  same reason `tui.preview` does not: it returns characters and series indices
  rather than markup, so it is testable on an install with neither.
- The screen itself is new: a status line that says at a glance whether the run
  is ready, training, stopping, finished or failed; five headline tiles
  (smoothed loss, validation loss, throughput, ETA, elapsed); panels whose
  headings are drawn into their borders rather than costing a row apiece; and a
  layout that follows the terminal, dropping the tiles below about 100 columns
  and the sidebar below about 72 rather than clipping either. The widgets moved
  out of `tui.app` into `tinydiffusion.tui.widgets`, which renders and decides
  nothing, leaving the app to own what the numbers mean.
- Thirty-odd themes where there were two. Ten are written for this dashboard —
  `tinydiffusion` and `tinydiffusion-light`, `latent`, `ember`, `mint`,
  `oceanic`, `synthwave`, `noir`, `paper`, `arctic` — and every theme Textual
  ships sits in the same cycle beside them. `t` opens a picker that applies
  each theme as the highlight moves, since a name alone says nothing about how
  a chart will look in it; `d` and `D` walk the list a keypress at a time. The
  colours reach the charts, the bars and the tiles rather than only the
  borders. The choice is remembered in `~/.config/tinydiffusion/tui.json` —
  moved by `TINYDIFFUSION_CONFIG_DIR` or `XDG_CONFIG_HOME` — and an unreadable
  one means the default rather than an error in front of a training run.
- New keys, and a `?` screen listing all of them: `r` restarts (stop, then
  start again once the worker has left), `f` is a focus mode that gives the
  charts the whole screen, `c` clears the log, and `ctrl+s` writes an SVG of
  the dashboard into the run's log directory. Every action is also in the
  command palette under `ctrl+p`, so none of them depends on knowing its key.
- The quartile bars name what they cover — `t 0-25%` through `t 75-100%`
  rather than `q0` through `q3` — and take a colour each from the theme.

### Fixed

- `timestep_sampler = "loss_second_moment"` no longer synchronises with the
  device on every batch. Its history update copied the batch's timesteps and
  losses to the host and folded them in with a Python loop, which is a blocking
  read per step — undoing, for any run that turned the setting on, exactly what
  the loop's `DRAIN_EVERY` buffering exists to buy. The history now lives on the
  training device and is updated with a handful of kernels: the ring's write
  slots are computed by sorting, so duplicate timesteps within a batch land on
  successive slots the way the sequential loop put them there, and the proposal
  picks between its adaptive and uniform branches with `torch.where` rather than
  a Python `if` on a device tensor. Verified two ways — against the previous
  implementation over 300 randomised cases including batches that overflow the
  ring, and under `torch.cuda.set_sync_debug_mode("error")`, which the old path
  trips and the new one does not. `warm` still reads back, and is documented as
  the one accessor that does; nothing on the hot path calls it.

- The same sampler now builds one proposal per *group* rather than one per rank.
  Each rank saw only its own shard, so an N-way run warmed N private histories
  on 1/N of the evidence each — not wrong, since the importance weights keep
  every rank's estimator unbiased, but N times the variance the group paid for,
  and with a small enough shard a timestep could go unseen and the proposal
  never warm at all. `all_gather_cat` collects the whole group's timesteps and
  losses first. It is opt-in from the training loop, so the single-process path
  is untouched, and the collective runs unconditionally once wired, so a rank
  cannot strand the others by skipping it.

- `tests/test_distributed.py` no longer hangs for five minutes on machines where
  gloo picks an unusable interface. Left to itself gloo resolves the hostname
  and binds the first address that comes back, which on a developer machine is
  quite often a VPN or hypervisor adapter that both ranks bind and neither can
  reach the other over; the symptom is not an error but the whole group sitting
  in `init_process_group` until the 300-second timeout fires. A session fixture
  now probes for a working configuration with a cheap two-rank all-reduce —
  gloo's own default first, since that is what CI uses, then the platform's
  loopback interface — and skips with a message naming `GLOO_SOCKET_IFNAME` if
  none works, rather than hanging. An operator who has already set that variable
  is left alone. Separately, the rendezvous port is now found rather than
  hard-coded at 29517, which was flaky by construction against a previous rank's
  socket in TIME_WAIT or a second copy of the suite.

- Documentation that predated the CUDA pinning in `pyproject.toml`. USAGE.md
  still told Windows readers that `uv sync` gives them a CPU-only PyTorch and
  walked them through an out-of-lockfile `uv pip install --torch-backend=auto`
  that the next sync would silently undo — the failure mode the `tool.uv.index`
  entry exists to remove. Its install and GPU sections now describe what
  actually happens, the corrupt-install repair uses `uv sync
  --reinstall-package` so it keeps the pinned index, and the startup banner it
  quotes is the one the loop prints today. Also: its table of contents linked
  to an anchor that no longer existed, and CONTRIBUTING.md's release steps
  pointed at a `version` field in `pyproject.toml` that has been dynamic since
  the number moved to `src/tinydiffusion/version.py`.
- The test suite no longer fails on an install without the `server` extra.
  `tests/test_server.py` imported FastAPI at module scope and broke collection
  outright, and one CLI test patched a module that does the same — so the
  `base-install` job added alongside the `plots` guard would have failed on
  them. Both are behind `importorskip` now, and the job's absent-extras check
  covers `textual` too.
- `plot` without the `plots` extra now reports the extra that supplies it
  instead of a traceback. `plot_runs` caught the `ImportError` matplotlib
  raises and re-raised it as a `RuntimeError`, which walked straight past the
  handler in `main` that exists to turn a missing optional dependency into one
  line — the same handler `serve` relies on for the `server` extra, which
  raises `ImportError` and has always reported cleanly. It raises `ImportError`
  now, so the two extras behave alike.
- The test suite no longer fails on an install without the `plots` extra. The
  twelve tests that draw a figure take a `pyplot` fixture that skips when
  matplotlib is absent; the ones that do not draw — the path helpers, and the
  test covering the message a missing matplotlib produces — still run there,
  since a base install is the install they are about. Every CI job synced
  `--all-extras`, so nothing exercised the configuration the optional extras
  exist to make possible; a `base-install` job now runs the suite without them,
  and fails if any extra has crept into the base dependencies.
- A metric that reached NaN or an infinity no longer makes `metrics.jsonl`
  unparsable. Python's `json` writes the bare tokens `NaN` and `Infinity`,
  which are an extension rather than JSON, and `jq` and `pandas.read_json`
  reject the whole file over one of them — losing the log of a diverged run,
  which is the run whose log is worth reading. Stored as `null` instead, so the
  epoch is still there and the gap in it stays visible.
- A metric named `step` or `time` no longer overwrites the step index or
  timestamp in its own record. The reserved keys are now written last.
- A backend that fails to close no longer replaces the exception that ended a
  training run. `RunLogger.__exit__` raised the close failure out of a block
  that was already unwinding, demoting whatever training actually failed on to
  a `__context__` few people look at; with an exception in flight the close
  failure is now a warning. It still raises when the block exited cleanly, when
  there is nothing to shadow.
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

- USAGE.md records what the dataloader actually costs, because the shape of the
  code invites the opposite conclusion: every epoch decodes the split from PIL
  and resizes it again, and caching the decoded tensors is the obvious
  optimisation. Measured on an RTX 2070 over MNIST at `batch_size = 128`, the
  PIL pipeline delivers 18,026 img/s with the default four workers and 4,652
  with none, against the 471 img/s `configs/mnist.toml` can consume and the
  2,629 the deliberately tiny `configs/smoke.toml` can — seven- to
  thirty-eight-fold headroom, prefetched so it overlaps the compute rather than
  adding to it. An epoch of `configs/mnist.toml` is 271.8 ms per batch, about
  128 s of pure compute, which is essentially the whole of it. Caching would
  have bought no wall-clock for 59 MB, a startup decode and a question about
  where the augmentation's flip is drawn, so the pipeline is unchanged and the
  measurement is written down instead.

- The samplers run under `torch.inference_mode` rather than `torch.no_grad`.
  Neither builds a graph; the stronger one also skips the version-counter and
  view-tracking bookkeeping that autograd keeps on every tensor it might later
  be asked about, which a sampler is never going to be. No behavioural
  difference, and no cost: samples leave the chain as ordinary tensors.

- The training loop no longer reads the loss and gradient norm back from the
  device on every batch. Both were fetched with `.item()` purely to log them,
  and each fetch blocks the CPU until the GPU catches up, which stops the loop
  queueing the next batch's work — the cost is the pipeline bubble, not the
  copy. They are now buffered on the device and fetched a run of batches at a
  time (`DRAIN_EVERY`), which is the same reasoning `QUARTILE_EVERY` already
  applied to the per-quartile losses. Every logged number is unchanged, down to
  the smoothed loss, and there is a test that asserts exactly that; the only
  visible difference is that the progress bar's loss updates every eighth batch
  rather than every batch. The gradient scaler's own scale check remains a sync
  on fp16, which is inherent to it.
- `fid` reports the sampler and spacing it drew with alongside the step count,
  both of which move the score.
- The sample server's status endpoint reports its `sampler` and
  `sample_spacing`.

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
