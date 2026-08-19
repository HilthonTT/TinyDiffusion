# Usage

Everything needed to install, run, and troubleshoot TinyDiffusion. For what
the project *is*, see [README.md](README.md); for contributing, see
[CONTRIBUTING.md](CONTRIBUTING.md).

- [Install](#install)
- [Using a GPU](#using-a-gpu)
- [What gets downloaded, and where](#what-gets-downloaded-and-where)
- [Running the CLI](#running-the-cli)
- [Training](#training)
- [Metrics and logging](#metrics-and-logging)
- [Sampling](#sampling)
- [Evaluating a checkpoint](#evaluating-a-checkpoint)
- [Measuring sample quality with FID](#measuring-sample-quality-with-fid)
- [Serving a checkpoint over HTTP](#serving-a-checkpoint-over-http)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [uv command reference](#uv-command-reference)

## Install

> A fuller walkthrough, with GPU setup and troubleshooting, lives in
> [docs/INSTALL.md](docs/INSTALL.md).


Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras --dev
```

That creates `.venv/` in the repo and installs the project into it in editable
mode. On Windows this gives you a **CPU-only** PyTorch; see below to get CUDA.

## Using a GPU

Training picks the GPU automatically when one is visible and falls back to the
CPU when it is not, so there is no flag to set — but the wheel has to support
your card. Check what you have:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

- A CPU build prints `2.13.0+cpu False []` — an empty arch list is the tell.
- A working CUDA build prints e.g. `2.13.0+cu132 True ['sm_75', ..., 'sm_120']`.

The arch list must contain your card's compute capability. Blackwell
(RTX 50-series) is `sm_120` and needs CUDA 12.8 or newer; an older wheel such
as `cu124` installs happily and then never sees the GPU.

### Getting a CUDA build on Windows

The lockfile resolves PyTorch from PyPI, whose **Windows** wheel is CPU-only —
CUDA builds live on PyTorch's own index. (The Linux wheel on PyPI already
bundles CUDA, so this section is Windows-only.) `--torch-backend=auto` reads
your driver and picks a matching CUDA version:

```powershell
uv pip install --reinstall --torch-backend=auto torch torchvision
```

> **This install sits outside the lockfile.** `uv sync` — and `uv run`, which
> syncs first — restores the locked CPU build and silently undoes it. Use
> `.\run.ps1`, `./run.sh`, or `uv run --no-sync` day to day, and re-run the
> command above after any deliberate `uv sync`.
>
> Pinning CUDA in `pyproject.toml` instead would drag multi-gigabyte wheels
> onto the Windows CI runner, which has no GPU to use them.

Once installed, the training banner names the device it is actually using:

```
2.36M parameters | device cuda (NVIDIA GeForce RTX 5060 Laptop GPU) | amp True
```

On CUDA the loop also turns on cuDNN autotuning, TF32 matmuls, and fp16
autocast (`amp = true` in the config, ignored on CPU). Measured on an RTX 5060
Laptop, one full epoch of `configs/smoke.toml` — training, sampling, and
checkpointing — takes **28 s on the GPU against 206 s on the CPU**. The gap
widens sharply for `configs/mnist.toml`, which is ~40x more compute per step.

## What gets downloaded, and where

| What | Size | Lands in |
| --- | --- | --- |
| Python deps (`uv sync`) | ~0.7 GB | `.venv/` |
| CUDA torch + torchvision | 1.8 GB download, 2.9 GB installed | `.venv/` |
| Cached wheels | mirrors the above | uv's cache — **not** the project |
| MNIST, fetched on first run | 63 MB | `data/` (`data_root`) |
| Inception-v3, fetched on first `fid` | 104 MB | torch hub cache (`TORCH_HOME`) |
| Checkpoints | ~3 MB per 0.2M params | `ckpt_dir` |
| Sample grids | ~25 KB each, one per epoch | `out_dir` |

MNIST downloads itself the first time you train and is reused afterwards.
`data/`, `checkpoints/`, `runs/` and `*.pt` are gitignored; `contents/` — the
default `out_dir` — is not, which is why the shipped configs write under
`runs/` instead.

### uv's cache and your disk

uv caches every wheel it downloads outside the project:

```powershell
uv cache dir      # e.g. C:\Users\<you>\AppData\Local\uv\cache
uv cache size     # bytes it is holding (experimental in uv 0.12)
```

If that cache and `.venv/` are on **different drives**, uv cannot hardlink and
copies instead — so a CUDA torch install costs ~3 GB in the cache *and* ~3 GB
in `.venv`, and the copy step takes minutes. uv warns when this happens:

```
warning: Failed to hardlink files; falling back to full copy.
```

Options, cheapest first:

| Command | Effect |
| --- | --- |
| `uv cache prune` | Removes dangling entries and cached environments. Safe: keeps wheels still in use, so nothing has to be re-downloaded. Run it periodically. |
| `uv cache clean torch torchvision` | Drops just those packages from the cache — the bulk of the space here. |
| `uv cache clean` | Wipes the cache entirely. Frees the most, but the next install re-downloads (1.8 GB for CUDA torch). |
| `uv cache prune --ci` | Trims the cache for CI persistence — keeps built wheels, drops what is cheap to re-fetch. Not for local use. |

Add `--force` to either command if uv refuses because it thinks entries are in
use, and never run them while an install is in flight.

The real fix is to move the cache onto the same volume as `.venv/`, so uv
hardlinks instead of copying and the duplicate gigabytes and slow copy both go
away:

```powershell
[Environment]::SetEnvironmentVariable('UV_CACHE_DIR','D:\uv-cache','User')
```

It takes effect in a new shell, and costs one re-download. `UV_CACHE_DIR` can
also be set per-command, or passed as `--cache-dir`.

## Running the CLI

The wrappers locate an interpreter that actually has the package installed and
forward everything else to the CLI. Use `run.ps1` from PowerShell, `run.sh`
from Git Bash, WSL, Linux, or macOS:

```powershell
.\run.ps1 train  --config configs\mnist.toml
.\run.ps1 sample --checkpoint checkpoints\last.pt --num-images 8
```

```bash
./run.sh train  --config configs/mnist.toml
./run.sh sample --checkpoint checkpoints/last.pt --num-images 8
```

Set `PYTHON` to force a specific interpreter: `PYTHON=/usr/bin/python3 ./run.sh …`.
With no arguments a wrapper prints the CLI help. `--version` (or `-V`) prints
the installed version and exits:

```bash
./run.sh --version      # tinydiffusion 0.1.0
```

The number comes from `src/tinydiffusion/version.py`, which is also what
`pyproject.toml` builds the distribution version from — so a checkout run from
source reports the same string as an installed wheel.

Equivalent invocations, if you prefer not to use the wrappers:

```bash
uv run --no-sync tinydiffusion train --config configs/mnist.toml
.venv/bin/python -m tinydiffusion.cli train --config configs/mnist.toml     # Unix
.\.venv\Scripts\python.exe -m tinydiffusion.cli train --config configs\mnist.toml
```

Two invocations that do **not** work:

- `python src/tinydiffusion/cli.py` — in a `src/` layout the package is
  importable only from an environment it is installed into, and running a file
  by path puts *that file's* directory on `sys.path` rather than `src/`.
  Result: `ModuleNotFoundError: No module named 'tinydiffusion'`.
- `bash .\run.sh` — bash reads the backslash as an escape and looks for a file
  named `.run.sh`. Use `./run.sh` or `bash run.sh`.

## Training

Start with the smoke config. It is the same pipeline shrunk to finish an epoch
in well under a minute on a GPU — the point is to prove the wiring works, not
to get good digits:

```bash
./run.sh train --config configs/smoke.toml
```

Then the real run:

```bash
./run.sh train --config configs/mnist.toml
```

Each epoch writes a `sample_XXXX.png` grid to `out_dir` — generated digits
above a strip of real ones, so contrast and stroke weight are directly
comparable — and a resumable `last.pt` to `ckpt_dir`. A conditional run
generates on the real strip's own labels, so the comparison is per class: a
generated 4 sits directly above a real 4. Both the strip and the starting
noise are fixed for the whole run, including across a `--resume`, so the PNGs
read as one set of digits sharpening rather than a fresh draw each epoch. The
checkpoint holds the
model, EMA shadow weights, optimiser moments, AMP scaler state, and the config,
so a resumed run continues rather than restarts:

```bash
./run.sh train --config configs/mnist.toml --resume checkpoints/last.pt
```

Flags override the config file when passed: `--seed`, `--device`, `--epochs`,
and the logging flags in [Metrics and logging](#metrics-and-logging).
`--config` itself is optional — omit it to run the built-in defaults, or the
settings stored in the checkpoint when [resuming](#resuming).

```bash
./run.sh train --config configs/mnist.toml --device cpu --epochs 1 --seed 7
```

### Stopping a run early

`Ctrl+C` does not kill the run outright. At the next batch boundary training
pauses and asks whether to stop, and if so whether to checkpoint first:

```text
stop training? [y/N] y
save a checkpoint so training can resume later? [Y/n] y
saved checkpoints/interrupted.pt (3 epochs complete, plus a partial epoch)
resume with: tinydiffusion train --resume checkpoints/interrupted.pt
```

Answering `n` to the first question resumes training where it left off. A
second `Ctrl+C` while the questions are on screen quits immediately, and a run
without a terminal attached — a CI job, a `nohup`ed script — saves and exits
without asking.

The save goes to `interrupted.pt`, never over `last.pt`. An interrupt lands
mid-epoch, so its weights are a few batches *worse* than the ones the previous
epoch finished on — and because a mid-epoch checkpoint records the last
**completed** epoch (so resuming replays the interrupted one in full), writing
it to `last.pt` would replace a good checkpoint with a worse one carrying the
same epoch number. Keeping them apart means `last.pt` is always the newest
complete epoch and `interrupted.pt` is always the furthest the run actually
got.

## Checkpoints

A run writes up to three kinds of file into `ckpt_dir`:

| File | Written | Holds |
| --- | --- | --- |
| `last.pt` | after every completed epoch | the newest complete epoch |
| `best.pt` | when held-out loss improves | the best epoch so far, by `val/loss` |
| `interrupted.pt` | on a saved `Ctrl+C` | the furthest the run got |
| `epoch_NNNN.pt` | when `keep_last > 0` | the last `keep_last` epochs |

Each is a full training state — weights, EMA shadow weights, optimiser moments,
AMP scale, and the config it was trained with — so any of them can be passed to
`--resume`, `sample`, `eval`, `fid` or `serve`.

`best.pt` is usually the one to sample from. Diffusion runs do not improve
monotonically, and the last epoch is not reliably the best one; without a
recorded score there is no way to tell which epoch was, and `last.pt` has
already overwritten the evidence. `keep_last` is the cruder insurance: set it
to keep a rolling window of numbered epoch snapshots.

```toml
[bookkeeping]
keep_best = true   # maintain best.pt (the default)
keep_last = 3      # also keep epoch_0028.pt, epoch_0029.pt, epoch_0030.pt
```

### Resuming

`--resume` on its own continues the run the checkpoint came from: the settings
it was trained with travel inside it, so the TOML file it started from is not
needed a second time.

```bash
./run.sh train --resume checkpoints/last.pt
```

Pass `--config` as well to resume into different settings, and the usual flags
still override whichever of the two was used:

```bash
./run.sh train --resume checkpoints/last.pt --epochs 60
```

`--resume` loads weights into the model the config describes, so the two have
to agree. They are checked before anything is loaded, and a mismatch names the
setting that changed:

```text
checkpoints/last.pt was trained with a different model, so it cannot resume
into this config:
  base_channels: checkpoint 64, config 128
match the config to the checkpoint, or start a fresh run without --resume
```

Everything the weights depend on is compared — the six architecture fields, the
schedule, and the three parameterisation fields. Settings that do not change
the weights, like `batch_size`, `lr` or `num_epochs`, are yours to change
between resumes.

A checkpoint written before the config travelled with it has nothing to rebuild
from, so a bare `--resume` on one asks for the config file instead.

## Metrics and logging

Every epoch, training prints a table of what it measured and appends the same
numbers to `log_dir/metrics.jsonl`:

```text
------------------------------------
| step 0                           |
------------------------------------
| time/epoch_seconds     |  27.9143 |
| time/images_per_second | 2148.31  |
| train/amp_scale        | 65536.00 |
| train/ema_decay        |   0.9950 |
| train/grad_norm        |   0.4127 |
| train/loss             |   0.0421 |
| train/loss_q0          |   0.1183 |
| train/loss_q1          |   0.0312 |
| train/loss_q2          |   0.0208 |
| train/loss_q3          |   0.0179 |
| train/lr               |  2.0e-04 |
| train/skipped_step     |   0.0000 |
| val/best_loss          |   0.0388 |
| val/loss               |   0.0388 |
------------------------------------
```

Per-batch values (`train/loss`, `train/grad_norm`, the quartiles) are averaged
over the epoch; states (`train/lr`, `train/amp_scale`, the timings) are recorded
as they stood at the end of it.

`val/loss` is the held-out score described under
[Validation and `best.pt`](#validation-and-bestpt) — the one number here that is
comparable between epochs, and between runs sharing a parameterisation.
`val/best_loss` is the lowest it has reached, which is the epoch `best.pt`
holds.

The quartile losses are the reason to look at this rather than just the progress
bar. `loss_q0` covers the lowest quarter of the diffusion schedule and `loss_q3`
the highest, so they say *where* the model is struggling: high-`t` error means
the denoiser cannot recover structure from near-pure noise, low-`t` error means
it cannot clean up the last little bit. The mean of the two moves for neither
reason. They are sampled from one batch in eight — slicing by timestep costs a
device sync, and every batch draws its timesteps independently, so the epoch
mean is the same either way. `train/skipped_step` is the fraction of batches AMP
threw away for inf/NaN gradients — a number that stays above zero is a reason to
lower `lr`.

`metrics.jsonl` is one JSON object per epoch, so a finished run is comparable
against the next one without re-reading a log:

```bash
python -c "import json;[print(json.loads(l)['train/loss']) for l in open('runs/mnist/metrics.jsonl')]"
```

TensorBoard is optional and off by default. It needs the `tracking` extra
(`uv sync --all-extras`, or `pip install 'tinydiffusion[tracking]'`) and writes
to `log_dir/tb`:

```bash
./run.sh train --config configs/mnist.toml --tensorboard
tensorboard --logdir runs/mnist/tb
```

| Flag | Config field | Meaning |
| --- | --- | --- |
| `--log-dir` | `log_dir` | Where `metrics.jsonl` and `tb/` are written |
| `--tensorboard` | `tensorboard` | Also write TensorBoard events |
| `--quiet` | `log_console` | Suppress the per-epoch table |

An unpassed flag leaves the config file's value alone, so `--tensorboard` can
be turned on for one run without editing the TOML.

## Sampling

```bash
./run.sh sample --checkpoint checkpoints/last.pt --num-images 16 --out contents/grid.png
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to sample from |
| `--num-images` | 8 | How many images to generate |
| `--steps` | the checkpoint's `sample_steps` | Denoising steps; fewer is faster, coarser |
| `--sampler` | the checkpoint's `sampler` | `ddim` or `dpmpp`; see [Choosing a sampler](#choosing-a-sampler) |
| `--eta` | 0.0 | 0 is deterministic DDIM, 1 is ancestral DDPM. `dpmpp` accepts only 0 |
| `--labels` | one image per class | Classes to generate, e.g. `7` or `0,1,2` |
| `--guidance` | the checkpoint's `guidance` | Classifier-free guidance scale |
| `--guidance-rescale` | the checkpoint's `guidance_rescale` | Corrects the scale guidance inflates; 0.7 above `--guidance 3` |
| `--out` | `contents/samples.png` | Where to write the grid |
| `--seed` | 0 | Seed applied before sampling |
| `--device` | auto | `cuda`, `cpu`, `cuda:1`, … |

Checkpoints embed the config they were trained with, so this reconstructs the
architecture from the `.pt` alone — the TOML that produced it is not needed.
Sampling always uses the EMA weights, which is what the training grids are
drawn from.

### Choosing a sampler

Two samplers, both usable with any checkpoint — which one to draw with is a
runtime choice, not something baked into the weights:

| `--sampler` | What it is | Steps it wants |
| --- | --- | --- |
| `ddim` | DDIM (Song et al. 2020), a first-order step along the probability-flow ODE | 50 |
| `dpmpp` | DPM-Solver++(2M) (Lu et al. 2022), second-order multistep | 15-20 |

`dpmpp` integrates the linear part of the same ODE in closed form and
approximates only the `x_0` prediction, reusing the previous step's network
evaluation rather than paying for a second one. A step therefore costs exactly
what a DDIM step costs, and ten to twenty of them land about where fifty DDIM
steps do:

```bash
./run.sh sample --checkpoint checkpoints/last.pt --sampler dpmpp --steps 20
```

It is deterministic, so `--eta` above 0 is refused rather than ignored — the
solver has no noise term to scale. `--eta 1` remains available under `ddim`,
where it reproduces ancestral DDPM sampling.

`sampler` is also a config field, so a run's per-epoch grids are drawn the same
way, and a checkpoint remembers what it was trained to be sampled with.
Sampling settings move FID, so hold `--sampler` and `--steps` fixed across the
checkpoints being compared.

### Asking for a particular digit

`configs/mnist.toml` trains class-conditionally, so the checkpoint knows which
digit is which:

```bash
./run.sh sample --checkpoint checkpoints/last.pt --labels 7 --num-images 8
./run.sh sample --checkpoint checkpoints/last.pt --labels 0,1,2 --guidance 4
```

`--labels` takes a comma-separated list, repeated in order until the grid is
full: `7` fills it with sevens, `0,1,2` cycles the three. Leaving it out gives
one image per class, laid out one class per column.

`--guidance` is the classifier-free guidance scale. At 1.0 you get the plain
conditional prediction. Above that, each step extrapolates away from the
model's unconditional prediction — digits get cleaner and more emphatically
class-typical, variety drops, and every step costs a second forward pass. 2–4
is the useful range on MNIST; past about 6 strokes start to blow out.

`--guidance-rescale` is what pushes that ceiling back. Extrapolation makes the
prediction *larger*, not just better aimed: its standard deviation grows with
the scale, and since the model was trained on targets of a fixed scale, the
recovered image ends up driven to the extremes of the range — the blow-out
above. Lin et al. 2023 §3.4 rescale the guided prediction back onto the
conditional one's standard deviation, then blend, since going all the way is
itself too strong:

```bash
./run.sh sample --checkpoint checkpoints/last.pt --guidance 5 --guidance-rescale 0.7
```

0.7 is the published blend and a good starting point. 0 is plain guidance and
the default, since the correction only matters once the scale is high enough to
need it. It is worth most on a `predict = "v"` model trained with `zero_snr`,
where the terminal step carries no signal to anchor the scale against.

All three flags are errors against an unconditional checkpoint, which has no
class space to name.

## Evaluating a checkpoint

Sampling shows you what the model draws; `eval` puts a number on it, by scoring
noise-prediction loss over the 10k held-out MNIST test split:

```bash
./run.sh eval --checkpoint checkpoints/last.pt
```

```
checkpoints/last.pt | test split | 10000 images | ema weights
loss 0.09984

     t     loss
     0   0.39324
    40   0.06916
    80   0.04649
   119   0.03946
   159   0.03071
   199   0.02001
```

The training loss is drawn at a *random* timestep per image, so it is far too
noisy to compare two checkpoints with. This pins the timesteps to a fixed grid
and reseeds before every batch, so the only thing that varies between two runs
is the weights — run it twice on the same checkpoint and you get the same
number to the last digit.

A conditional checkpoint is scored on the true labels, and never with
guidance: the objective it was trained on is the conditional prediction, so
scoring anything else would measure something the run never optimised.

Read the per-timestep column as well as the headline. Loss is always highest at
`t = 0`, where `x_t` is nearly clean and there is almost no noise left to
identify, and falls as `t` grows. A checkpoint that improves only at high `t`
has learned the easy end of the schedule and little else.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to score |
| `--split` | `test` | `test` (10k held out) or `train` (60k) |
| `--timesteps` | 10 | How many timesteps to score at |
| `--batch-size` | the checkpoint's | Larger is faster |
| `--data-root` | the checkpoint's | Dataset directory |
| `--no-ema` | off | Score the raw weights instead of the EMA |
| `--seed` | 0 | Fixes the noise; change it to resample |
| `--device` | auto | `cuda`, `cpu`, … |

Scoring both splits is how you see overfitting — a test loss that stalls or
rises while the train loss keeps falling:

```bash
./run.sh eval --checkpoint checkpoints/last.pt --split test
./run.sh eval --checkpoint checkpoints/last.pt --split train
```

One caveat: this is a proxy. Lower held-out loss means the network predicts
noise better, which correlates with sample quality but does not measure it
directly. For that, use `fid` below — and keep looking at the grids.

## Measuring sample quality with FID

`eval` scores the training objective; `fid` scores the thing you actually care
about. It draws samples, pushes them and an equal number of real images through
a pretrained Inception-v3, and measures the Frechet distance between the two
clouds of activations. Lower is better.

```bash
./run.sh fid --checkpoint checkpoints/last.pt
```

```
checkpoints/last.pt | train split | ema weights
fid 18.472

10000 generated vs 10000 real images
50 ddim steps | guidance 2
```

The Inception weights (~100 MB) download on first use into the usual torch hub
cache; see [What gets downloaded, and where](#what-gets-downloaded-and-where).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to score |
| `--num-images` | 10000 | Samples drawn, and real images compared against |
| `--split` | `train` | Real split to compare against |
| `--batch-size` | the checkpoint's | Larger is faster |
| `--data-root` | the checkpoint's | Dataset directory |
| `--steps` | the checkpoint's | Denoising steps per sample |
| `--sampler` | the checkpoint's | `ddim` or `dpmpp`; hold it fixed across compared checkpoints |
| `--eta` | 0.0 | 0 is DDIM, 1 is ancestral DDPM |
| `--guidance` | the checkpoint's | Classifier-free guidance scale |
| `--guidance-rescale` | the checkpoint's | Guidance rescale factor; sweep it jointly with `--guidance` |
| `--no-ema` | off | Sample the raw weights instead of the EMA |
| `--seed` | 0 | Fixes the samples; change it to redraw |
| `--device` | auto | `cuda`, `cpu`, … |

Read the number as a comparison, never as an absolute:

- **It only compares like with like.** `--num-images`, `--split`, `--steps` and
  the extractor each move the score on their own, so hold them fixed across the
  checkpoints you are comparing.
- **Small sample counts inflate it.** The Inception feature space is 2048-dimensional,
  so a covariance estimated from fewer than ~2048 images per side is singular and
  the score is biased upwards by an amount that depends on the count. Below that
  the report says so. Use 10k when the number needs to mean anything, and a few
  hundred only for a quick smoke test.
- **It is not the published FID.** Those numbers come from the original
  TensorFlow Inception graph; torchvision's port differs enough to shift the
  absolute value. The ordering it induces over checkpoints is what carries over.
- **A conditional model is sampled with a uniform class mix**, while real MNIST
  is not — the first 10k training images run from 8.6% (5s) to 11.3% (1s). That
  prior mismatch adds a small constant to the score. It is identical for every
  checkpoint scored the same way, so comparisons are unaffected; it is one more
  reason not to read the absolute number.

It is also slow — every score runs the full DDIM chain `--num-images` times —
which is why it is a command you run at the end of a run rather than a metric
logged per epoch.

Guidance is worth sweeping rather than leaving at the checkpoint's default. It
trades diversity for fidelity, and FID punishes both, so the minimum usually
sits somewhere above 1:

```bash
for g in 1 1.5 2 3 5; do
  ./run.sh fid --checkpoint checkpoints/last.pt --num-images 2000 --guidance $g
done
```

## Serving a checkpoint over HTTP

`serve` puts a checkpoint behind a small JSON API, so something other than a
shell can ask it for digits. It needs the `server` extra:

```bash
uv sync --extra server            # or: pip install 'tinydiffusion[server]'
./run.sh serve --checkpoint checkpoints/last.pt
```

```
serving checkpoints/last.pt on http://127.0.0.1:8000
INFO:     Application startup complete.
```

The checkpoint is loaded once at startup, not per request. Interactive API docs
are at `/docs`, and the schema at `/openapi.json`.

**`POST /api/sample`** generates a grid and returns where to fetch it. Every
field is optional except that the defaults come from the checkpoint:

```bash
curl -X POST localhost:8000/api/sample -H 'content-type: application/json' \
  -d '{"num_images": 8, "labels": [3], "guidance": 2.0, "steps": 50, "seed": 0}'
```

```json
{"url": "/images/d3d0e07831c3442197753ea2d7f367f9.png",
 "filename": "d3d0e07831c3442197753ea2d7f367f9.png",
 "num_images": 8}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `num_images` | 8 | Images in the grid, up to `--max-images` |
| `labels` | one per class | Classes to generate. Conditional checkpoints only |
| `guidance` | the checkpoint's | Classifier-free guidance scale |
| `guidance_rescale` | the checkpoint's | Guidance rescale factor, in [0, 1] |
| `steps` | the checkpoint's | DDIM steps |
| `eta` | 0.0 | 0 is DDIM, 1 is ancestral DDPM |
| `seed` | null | Fixes the sample; the same seed returns the same image |

`seed` is request-local: it seeds a generator used for that one sample and
nothing else. It does not reseed the server, so one caller's seed cannot reach
into another caller's images or outlive the request that asked for it.

**`GET /images/{filename}`** serves the PNG. **`GET /api/status`** reports what
is loaded — device, image size, class count, and the defaults above — which is
also how a client learns whether it may send `labels`.

A request that does not fit the checkpoint comes back as a 400 with the reason
(`labels` against an unconditional model, a class that does not exist, more
images than the ceiling); a malformed one is a 422 from the schema.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Checkpoint to serve |
| `--host` | `127.0.0.1` | Interface to bind |
| `--port` | 8000 | Port to bind |
| `--max-images` | 64 | Ceiling on `num_images` per request |
| `--image-dir` | a temp dir | Where PNGs are written |
| `--image-ttl` | 3600 | Seconds a PNG is kept before it is swept. 0 keeps them forever |
| `--keep-images` | 256 | PNGs retained regardless of age, newest first. 0 for no cap |
| `--cors-origin` | none | Origin allowed to call the API from a browser. Repeatable |
| `--no-ema` | off | Serve the raw weights instead of the EMA |
| `--device` | auto | `cuda`, `cpu`, … |

Two things to know before exposing it:

- **There is no authentication**, which is why the default bind is loopback
  rather than `0.0.0.0`. Generating an image is seconds of GPU time on request,
  so an open port is a denial-of-service invitation. Widen it only behind
  something that does authenticate.
- **Requests are serialised.** One checkpoint on one device, one chain at a
  time; concurrent callers queue rather than fighting over VRAM. Throughput
  comes from `num_images` in a single request, not from parallel requests.
- **Rendered PNGs are swept.** Every request writes a file, and nothing else
  deletes them, so the image directory is bounded by age (`--image-ttl`) and by
  count (`--keep-images`). The sweep only ever touches names the server itself
  issued, so pointing `--image-dir` at a directory holding anything else is
  safe. Turn both to 0 to keep everything — reasonable for a short-lived local
  server, a slow disk leak for anything longer.

## Configuration

Configs are TOML. Tables are cosmetic grouping only: every key is flattened
into the flat `TrainConfig` namespace, so a key must name a real field and may
appear in exactly one table. Unknown or repeated keys are errors, not silent
no-ops — a typo fails immediately instead of wasting a training run.

```toml
[model]
base_channels = 64
channel_mult = [1, 2, 2]
attn_resolutions = [16]

[diffusion]
num_timesteps = 1000
schedule = "cosine"       # or "linear", which uses beta_start/beta_end
```

### Going faster

Everything here is off or neutral by default, because each one is a change you
should measure on your own hardware rather than inherit:

```toml
[optimisation]
grad_accum = 2          # effective batch of batch_size * grad_accum
lr_schedule = "cosine"  # decay to zero over the run, after the warmup ramp

[bookkeeping]
amp_dtype = "bf16"      # no gradient scaler, no skipped steps
compile = true          # torch.compile the training step
channels_last = true    # measure this one; it can lose on a small model
```

When the limit is memory rather than speed — a wider `base_channels` or a
larger `batch_size` than the card will hold — turn on `grad_checkpoint`. It
drops the U-Net's intermediate activations and recomputes them during the
backward pass, costing roughly a third more time per step and buying back most
of the activation memory. Nothing else changes: same weights, same loss, and a
checkpoint trained with it on resumes fine with it off. Sampling is untouched,
since there is no backward pass there to save for.

Measured on an RTX 5060 at the `configs/mnist.toml` settings (batch 128, 32px,
`base_channels = 64`), 25 steps after a warmup:

| Setting | ms/step | vs default |
| --- | --- | --- |
| fp16 (the default) | 203 | — |
| bf16 | 206 | 0.98x |
| fp16 + `channels_last` | 182 | **1.11x** |
| bf16 + `channels_last` | 183 | 1.11x |

So on that card `channels_last` is worth about 11% and `amp_dtype = "bf16"`
buys no throughput — its argument is stability, not speed: no loss scaler, and
so no skipped steps. Both are worth re-measuring on your own card and at your
own width; these numbers do not transfer.

`compile` wraps the network for the training step only. The checkpoint, the
EMA and every sampler keep the eager module, which shares its parameters — so a
compiled run writes ordinary checkpoints rather than ones whose keys all carry
a `_orig_mod.` prefix, and the first batch pays the compile cost once.

> **On Windows, `compile` needs Triton**, which the PyTorch Windows wheels do
> not ship: `pip install triton-windows`. Without it the run says so on
> startup and trains eagerly, rather than failing on the first batch several
> frames inside dynamo. It is therefore unmeasured here.

`amp_dtype = "bf16"` turns the gradient scaler off, since bfloat16 has
float32's exponent range and nothing to overflow. That also means no skipped
steps: `train/skipped_step` stays at zero, and it is not hiding anything. On a
GPU without bfloat16 the run says so and falls back to fp16.

`grad_accum` buys an effective batch larger than VRAM allows. Each group is
averaged over the batches it actually holds, including the ragged one a
non-dividing loader leaves at the end of an epoch, so the last update of an
epoch is not quietly a fraction of the others. `lr_warmup` and `lr_schedule`
count optimiser steps, so raising `grad_accum` covers proportionally more data
per step of the schedule.

The optimiser is AdamW. At the default `weight_decay = 0.0` that is Adam
exactly — decoupled decay is the only thing the two differ in — so turning it
on is an opt-in, and `betas` is there for the runs that need it.

### Choosing a dataset

`dataset` names an entry in the registry in `tinydiffusion/data/datasets.py`:

| Name | Channels | Native size | Classes | Flips |
| --- | --- | --- | --- | --- |
| `mnist` | 1 | 28 | 10 | no |
| `fashion_mnist` | 1 | 28 | 10 | yes |
| `cifar10` | 3 | 32 | 10 | yes |

Nothing downstream hard-codes a channel count: the U-Net's input and output
width, the shape the samplers draw, and the reference side of a FID all read it
from the spec the config names. Switching datasets is therefore the one key,
plus whatever `num_classes` and `image_size` the new one implies:

```bash
./run.sh train --config configs/cifar10.toml
./run.sh train --config configs/mnist.toml --dataset fashion_mnist
```

`num_classes` has to match the dataset's label space exactly, and a mismatch is
refused when the config is read — the labels come from the dataset, so a
smaller count would index past the embedding table the first time a batch
carried a higher one.

The "flips" column is whether a random horizontal flip preserves the label. It
does for natural images and does not for digits, so MNIST opts out. The flip is
applied to the training split only: a scored split never gets one, or a
held-out number would move for reasons that have nothing to do with the
weights.

A checkpoint records the dataset it was trained on, and `--resume` refuses to
carry a run across a change to it — the channel count is part of the shape of
every tensor in the state dict.

### Validation and `best.pt`

After each epoch the EMA weights are scored on a fixed slice of the held-out
test split, at a pinned grid of timesteps with pinned noise. Everything that
would otherwise make the number jump around is held constant, so `val/loss`
moves only with the weights — which is what makes it usable for picking a best
epoch:

```toml
[validation]
val_every = 1     # epochs between scores; 0 turns validation off
val_steps = 10    # timesteps to score at
val_batches = 4   # batches of the test split to score; 0 for all of it
```

The defaults cost a few percent of an epoch. `val_batches = 0` scores the whole
10k split, which is a better absolute number and a much slower one — and it
buys little here, since the same small slice every epoch already isolates the
weights. The score lands in `metrics.jsonl` as `val/loss`, alongside
`val/best_loss`, and drives `best.pt`.

It is scored with the *EMA* weights, because those are what the sample grids
and every downstream command draw from; scoring the live weights would pick a
best epoch nobody ever samples.

### Choosing the parameterisation

`predict`, `variance` and `objective` pick what the network predicts, where the
reverse variance comes from, and what is optimised. Their defaults —
`epsilon` / `fixed_small` / `mse` — are the DDPM baseline and build the `DDPM`
class itself; anything else builds `GaussianDiffusion`, which implements all
three as explicit choices. The combination worth trying is Nichol & Dhariwal's
improved DDPM, which samples well in far fewer steps:

```toml
[diffusion]
variance = "learned_range"   # the net emits the variance alongside the mean
objective = "rescaled_mse"   # L_simple plus a down-weighted variational term
```

A learned variance doubles the U-Net's output channels, so a checkpoint trained
this way is not interchangeable with a baseline one. Bad combinations — a
learned variance under plain `mse`, which would leave the variance head
untrained — are rejected when the config is read.

### Predicting velocity, and closing the terminal-SNR leak

`predict = "v"` has the network predict the *velocity*
`v = sqrt(abar)*eps - sqrt(1-abar)*x0` (Salimans & Ho 2022). It interpolates
between the two obvious targets — epsilon at high noise, `x_0` at low — so no
timestep is left regressing on something it cannot see the signal in.

It pairs with `zero_snr`, which rescales the beta schedule so the last step
carries no signal at all (Lin et al. 2024). A schedule that stops short leaves
`x_T` holding a trace of the image's mean brightness; training always starts
from a real image and never notices, but sampling starts from pure noise, whose
mean is zero, and the model spends the chain restoring a brightness that was
never there. The symptom is samples that are never fully black or fully white,
and it gets worse the fewer steps you take.

```toml
[diffusion]
predict = "v"
zero_snr = true
```

`zero_snr` with `predict = "epsilon"` is rejected: with no signal at `t = T`
there is no epsilon that says anything about `x_0`, which is exactly why the
rescaling is published alongside `v`. How much it buys depends on the schedule
— `linear` leaves `sqrt(abar_T)` around 0.006 over 1000 steps, while `cosine`
clamps its betas at 0.999 and lands within 5e-5 of zero on its own, so there it
mostly just makes the intent exact.

### Weighting the timesteps

Two independent knobs, both aimed at the same problem: a uniform draw of
timesteps with uniform weights spends most of the gradient on the steps that
need it least.

```toml
[diffusion]
loss_weighting = "min_snr"      # clamp each timestep's weight at min_snr_gamma
min_snr_gamma = 5.0             # the paper's value; not sensitive
timestep_sampler = "loss_second_moment"
```

`loss_weighting = "min_snr"` (Hang et al. 2023) weights each timestep by
`min(SNR(t), gamma)`, expressed in whatever space the network predicts in.
Uniform weighting of an epsilon-space MSE is implicitly `1/SNR` weighting in
`x_0` space, so the low-noise timesteps — where the model is nearly right
already — dominate and pull against the high-noise ones. The clamp stops that,
and usually reaches a given loss in noticeably fewer epochs. It applies to the
MSE term only: under a hybrid objective the variational term keeps its own
scale, and under a pure KL objective there is no MSE term to weight, so the
combination is rejected. The logged `train/loss_q*` buckets stay unweighted, so
they remain comparable with each other and across runs.

`timestep_sampler = "loss_second_moment"` (Nichol & Dhariwal 2021) attacks the
other half: which timesteps get drawn at all. It keeps a ten-deep history of
each timestep's loss, samples in proportion to its RMS, and divides the loss
back through by the sampling probability so the estimator stays unbiased. The
draw is uniform until every timestep has a full history. It earns its keep on
the variational objectives, whose per-timestep terms differ by orders of
magnitude, and does very little for plain `mse`. The history lives in memory
only, so a resumed run re-warms it over its first few hundred batches.

### Class conditioning

`num_classes` opts into class-conditional training. It is off by default, and
on in both shipped configs — the one place they depart from the defaults:

```toml
[conditioning]
num_classes = 10      # MNIST's digits; has to match the dataset's label space
class_dropout = 0.1   # labels replaced by the null token during training
guidance = 2.0        # scale used at sample time
guidance_rescale = 0.0  # correction for the scale guidance inflates
```

The three work together. `num_classes` gives the U-Net a class embedding,
summed into the timestep embedding. That embedding table holds one extra row —
the null token, meaning "no class given" — and `class_dropout` is what trains
it: 10% of training labels are replaced by it, so the one network learns both a
conditional and an unconditional prediction. `guidance` then extrapolates
between them at sample time, and needs the other two to mean anything;
combinations that cannot work, such as `guidance` above 1 with
`class_dropout = 0`, are rejected when the config is read.

`guidance_rescale` corrects the scale that extrapolation inflates, and is what
keeps a high `guidance` from washing images out; see
[Asking for a particular digit](#asking-for-a-particular-digit) for what it does and when to
reach for it. It is 0 by default, and setting it above 0 at `guidance = 1.0` is
rejected rather than silently ignored, since there is no extrapolation there to
correct.

The label costs almost nothing: one embedding row per class, and the sample
grids become class-matched — each generated digit sits directly above a real
one of the same class instead of above an unrelated digit.

Conditional and unconditional checkpoints are not interchangeable, since the
state dict differs. Turning conditioning on part way through a run means
starting it over, not `--resume`.

| Field | Default | Notes |
| --- | --- | --- |
| `dataset` | `mnist` | Also `fashion_mnist`, `cifar10`; see [Choosing a dataset](#choosing-a-dataset) |
| `data_root` | `data` | The dataset is downloaded here on first use |
| `image_size` | 32 | 32 keeps 28x28 digits intact and halves exactly |
| `batch_size` | 128 | 8 GB of VRAM has room for 256 |
| `num_workers` | 4 | 0 when debugging |
| `base_channels` | 64 | Width; DDPM uses 128 for CIFAR-10 |
| `channel_mult` | `[1, 2, 2]` | One entry per resolution level |
| `num_res_blocks` | 2 | Residual blocks per level |
| `attn_resolutions` | `[16]` | Spatial sizes that get self-attention |
| `dropout` | 0.1 | Inside ResBlocks |
| `num_classes` | unset | Classes to condition on; 10 for MNIST |
| `class_dropout` | 0.1 | Labels replaced by the null token in training |
| `guidance` | 1.0 | Guidance scale; 1.0 is plain conditional |
| `guidance_rescale` | 0.0 | Corrects the scale guidance inflates; 0.7 above `guidance` 3 |
| `num_timesteps` | 1000 | Length of the diffusion schedule |
| `schedule` | `cosine` | `cosine` or `linear` |
| `beta_start` / `beta_end` | 1e-4 / 0.02 | Linear schedule only |
| `predict` | `epsilon` | Also `v` (velocity), `start_x`, `previous_x` |
| `variance` | `fixed_small` | Also `fixed_large`, `learned_range`, `learned` |
| `objective` | `mse` | Also `rescaled_mse` (hybrid), `kl`, `rescaled_kl` |
| `zero_snr` | `false` | Rescale the schedule to zero terminal SNR; needs `predict = "v"` or `"start_x"` |
| `loss_weighting` | `uniform` | Or `min_snr`, clamping each timestep's weight |
| `min_snr_gamma` | 5.0 | The clamp, under `min_snr` weighting |
| `timestep_sampler` | `uniform` | Or `loss_second_moment`, importance sampling the draw |
| `num_epochs` | 30 | |
| `lr` | 2e-4 | Adam |
| `lr_warmup` | 500 | Optimiser steps to ramp the LR over; 0 disables |
| `lr_schedule` | `constant` | Or `cosine`, decaying to zero after the ramp |
| `betas` | `[0.9, 0.999]` | AdamW moment decays |
| `weight_decay` | 0.0 | Decoupled; at 0 AdamW is Adam |
| `grad_accum` | 1 | Micro-batches per optimiser step |
| `grad_clip` | 1.0 | 0 disables clipping |
| `ema_decay` | 0.9999 | Sample quality depends on this |
| `ema_warmup` | 2000 | Steps over which the decay ramps in |
| `seed` | 0 | Python, NumPy and torch RNGs; with the epoch index, fixes the batch order |
| `deterministic` | `false` | Force deterministic CUDA kernels and disable the cuDNN autotuner, at a throughput cost |
| `amp` | `true` | Autocast; ignored off CUDA |
| `amp_dtype` | `fp16` | Or `bf16`, which runs unscaled but wants Ampere or newer |
| `compile` | `false` | `torch.compile` the training step |
| `channels_last` | `false` | Worth measuring rather than assuming |
| `grad_checkpoint` | `false` | Recompute activations in the backward pass: roughly a third more compute for a large cut in memory |
| `sample_every` | 1 | Epochs between sample grids; 0 disables |
| `num_samples` | 16 | Images per grid |
| `sampler` | `ddim` | Or `dpmpp` (DPM-Solver++(2M)), which needs about a third of the steps |
| `sample_steps` | 50 | Denoising steps for those grids; 15-20 is plenty for `dpmpp` |
| `out_dir` | `contents` | Sample grids |
| `ckpt_dir` | `checkpoints` | Checkpoints |
| `device` | auto | `cuda` when available, else `cpu` |
| `log_dir` | `runs/mnist` | `metrics.jsonl`, and `tb/` when TensorBoard is on |
| `log_console` | `true` | Per-epoch metrics table |
| `log_jsonl` | `true` | Append `metrics.jsonl` |
| `tensorboard` | `false` | TensorBoard events; needs the `tracking` extra |

`configs/mnist.toml` lists every field with its default, `[conditioning]`
aside;
`TrainConfig` in `src/tinydiffusion/training/config.py` is the source of truth.

## Troubleshooting

**`ModuleNotFoundError: No module named 'tinydiffusion'`**
The interpreter you used is not one the package was installed into. A second
virtualenv in the repo is the usual culprit — `uv` manages `.venv` and nothing
else, so an environment under any other name will not have the package however
recently you activated it. Check with `echo $env:VIRTUAL_ENV` (PowerShell) or
`echo $VIRTUAL_ENV` (bash): if it names anything but `.venv`, deactivate and
reopen the terminal. Use `./run.sh` / `.\run.ps1`, which only ever pick
`.venv`, or install into the interpreter you want with
`python -m pip install -e .`.

The same mix-up shows up in an editor as unresolved imports — VS Code reads
`VIRTUAL_ENV` when choosing an interpreter, so a stale value points Pylance at
the wrong environment. Fix it with **Python: Select Interpreter** →
`.venv\Scripts\python.exe`, after restarting the editor so it stops inheriting
the old variable.

**`ModuleNotFoundError: No module named 'torch._weights_only_unpickler'`** (or
any other missing submodule of an installed package)
The install is corrupt, not misconfigured — typically an uninstall that was
interrupted partway. `uv` reports the same thing as
`Failed to uninstall … due to missing RECORD file`. Repair it by reinstalling:

```powershell
uv pip install --reinstall --torch-backend=auto torch torchvision
```

Avoid running `uv sync`/`uv run` against the project while something else is
using `.venv`; that race is a common way to get here.

**Training says `device cpu` on a machine with a GPU**
The installed torch is a CPU-only build. See [Using a GPU](#using-a-gpu).

**`no CUDA device visible, falling back to CPU`**
`--device cuda` was asked for but torch cannot see a GPU. The run continues on
the CPU rather than failing; same fix as above.

**CUDA out of memory**
Lower `batch_size`, then `base_channels`. Sampling `num_samples` also
allocates a batch at once.

**Windows: the install step takes minutes and disk usage doubles**
uv's cache is on a different drive from `.venv/`. See
[uv's cache and your disk](#uvs-cache-and-your-disk).

## uv command reference

Environment:

| Command | What it does |
| --- | --- |
| `uv sync --all-extras --dev` | Create/refresh `.venv` to match `uv.lock`. **Reverts a manually installed CUDA torch.** |
| `uv run --no-sync <cmd>` | Run inside `.venv` without syncing first — safe with a CUDA torch installed. |
| `uv run <cmd>` | Syncs, then runs. Convenient, but undoes an out-of-lockfile install. |
| `uv pip install -e .` | Install this project into the active environment. |
| `uv pip install --reinstall --torch-backend=auto torch torchvision` | Fetch a CUDA build matching your driver; also the repair for a corrupt install. |
| `uv pip list` | What is actually installed. |
| `uv lock --upgrade` | Re-resolve dependencies and update `uv.lock`. |

Cache and disk:

| Command | What it does |
| --- | --- |
| `uv cache dir` | Where the cache lives. |
| `uv cache size` | Bytes held. Experimental in uv 0.12; it warns unless you pass `--preview-features cache-size`. |
| `uv cache prune` | Remove dangling entries and cached environments. Safe, keeps what is in use. |
| `uv cache clean [PACKAGE…]` | Wipe the cache, or just the named packages. |
| `uv cache prune --ci` | CI-oriented trim; keeps built wheels. |
| `UV_CACHE_DIR=<path>` | Move the cache — put it on the same volume as `.venv` to enable hardlinking. |
| `UV_LINK_MODE=copy` | Silence the hardlink warning when a cross-volume cache is intentional. |

Project checks:

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```
