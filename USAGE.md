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
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [uv command reference](#uv-command-reference)

## Install

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
comparable — and a resumable `last.pt` to `ckpt_dir`. The checkpoint holds the
model, EMA shadow weights, optimiser moments, AMP scaler state, and the config,
so a resumed run continues rather than restarts:

```bash
./run.sh train --config configs/mnist.toml --resume checkpoints/last.pt
```

Flags override the config file when passed: `--seed`, `--device`, `--epochs`,
and the logging flags in [Metrics and logging](#metrics-and-logging).
`--config` itself is optional — omit it to run the built-in defaults.

```bash
./run.sh train --config configs/mnist.toml --device cpu --epochs 1 --seed 7
```

### Stopping a run early

`Ctrl+C` does not kill the run outright. At the next batch boundary training
pauses and asks whether to stop, and if so whether to write `last.pt` first:

```text
stop training? [y/N] y
save a checkpoint so training can resume later? [Y/n] y
saved checkpoints/last.pt (3 epochs complete)
resume with: tinydiffusion train --resume checkpoints/last.pt
```

Answering `n` to the first question resumes training where it left off. A
checkpoint saved mid-epoch records the last *completed* epoch, so resuming
replays the interrupted one in full. A second `Ctrl+C` while the questions are
on screen quits immediately, and a run without a terminal attached — a CI job,
a `nohup`ed script — saves and exits without asking.

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
------------------------------------
```

Per-batch values (`train/loss`, `train/grad_norm`, the quartiles) are averaged
over the epoch; states (`train/lr`, `train/amp_scale`, the timings) are recorded
as they stood at the end of it.

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
| `--steps` | the checkpoint's `sample_steps` | DDIM steps; fewer is faster, coarser |
| `--eta` | 0.0 | 0 is deterministic DDIM, 1 is ancestral DDPM |
| `--out` | `contents/samples.png` | Where to write the grid |
| `--seed` | 0 | Seed applied before sampling |
| `--device` | auto | `cuda`, `cpu`, `cuda:1`, … |

Checkpoints embed the config they were trained with, so this reconstructs the
architecture from the `.pt` alone — the TOML that produced it is not needed.
Sampling always uses the EMA weights, which is what the training grids are
drawn from.

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
directly — the published metric for that is FID, which is not implemented here.
Keep looking at the grids.

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

| Field | Default | Notes |
| --- | --- | --- |
| `data_root` | `data` | MNIST is downloaded here on first use |
| `image_size` | 32 | 32 keeps 28x28 digits intact and halves exactly |
| `batch_size` | 128 | 8 GB of VRAM has room for 256 |
| `num_workers` | 4 | 0 when debugging |
| `base_channels` | 64 | Width; DDPM uses 128 for CIFAR-10 |
| `channel_mult` | `[1, 2, 2]` | One entry per resolution level |
| `num_res_blocks` | 2 | Residual blocks per level |
| `attn_resolutions` | `[16]` | Spatial sizes that get self-attention |
| `dropout` | 0.1 | Inside ResBlocks |
| `num_timesteps` | 1000 | Length of the diffusion schedule |
| `schedule` | `cosine` | `cosine` or `linear` |
| `beta_start` / `beta_end` | 1e-4 / 0.02 | Linear schedule only |
| `num_epochs` | 30 | |
| `lr` | 2e-4 | Adam |
| `grad_clip` | 1.0 | 0 disables clipping |
| `ema_decay` | 0.9999 | Sample quality depends on this |
| `ema_warmup` | 2000 | Steps over which the decay ramps in |
| `seed` | 0 | Python, NumPy and torch RNGs |
| `amp` | `true` | fp16 autocast; ignored off CUDA |
| `sample_every` | 1 | Epochs between sample grids; 0 disables |
| `num_samples` | 16 | Images per grid |
| `sample_steps` | 50 | DDIM steps for those grids |
| `out_dir` | `contents` | Sample grids |
| `ckpt_dir` | `checkpoints` | Checkpoints |
| `device` | auto | `cuda` when available, else `cpu` |
| `log_dir` | `runs/mnist` | `metrics.jsonl`, and `tb/` when TensorBoard is on |
| `log_console` | `true` | Per-epoch metrics table |
| `log_jsonl` | `true` | Append `metrics.jsonl` |
| `tensorboard` | `false` | TensorBoard events; needs the `tracking` extra |

`configs/mnist.toml` lists every field with its default;
`TrainConfig` in `src/tinydiffusion/training/config.py` is the source of truth.

## Troubleshooting

**`ModuleNotFoundError: No module named 'tinydiffusion'`**
The interpreter you used is not one the package was installed into. A second
virtualenv in the repo is the usual culprit — `uv` manages `.venv`, so an
activated `venv` will not have it. Use `./run.sh` / `.\run.ps1`, which pick a
working interpreter, or install into the one you want with
`python -m pip install -e .`.

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
