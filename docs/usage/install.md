# Installing and updating

Setting up the environment, confirming the GPU is actually being used,
and what lands on disk. A fuller walkthrough lives in
[INSTALL.md](../INSTALL.md).

Part of [Usage](../../USAGE.md).

- [Install](#install)
- [Using a GPU](#using-a-gpu)
- [What gets downloaded, and where](#what-gets-downloaded-and-where)
- [uv command reference](#uv-command-reference)

## Install

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras --dev
```

That creates `.venv/` in the repo and installs the project into it in editable
mode. It also installs a **CUDA** build of PyTorch on Windows and Linux —
`pyproject.toml` points the Windows wheels at PyTorch's own index, because
PyPI's Windows wheel is CPU-only — so a machine with an NVIDIA GPU is ready to
use it with no second step. Add `--no-sources` for a CPU-only environment,
which is what CI uses:

```bash
uv sync --all-extras --dev --no-sources
```

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
as `cu124` installs happily and then never sees the GPU. Both builds are
recorded in `uv.lock` and chosen by platform marker, so a later `uv sync` — or
the implicit one a bare `uv run` does — cannot silently put a CPU wheel back.
[INSTALL.md](../INSTALL.md#how-the-gpu-build-is-selected) has the
details, including how to move to a different CUDA version.

Training then names the device it is actually using on its first line:

```
6.95M parameters | mnist 32px x1 | device cuda (NVIDIA GeForce RTX 5060 Laptop GPU) | amp fp16 | ...
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

## uv command reference

Environment:

| Command | What it does |
| --- | --- |
| `uv sync --all-extras --dev` | Create/refresh `.venv` to match `uv.lock`. **Reverts a manually installed CUDA torch.** |
| `uv run --no-sync <cmd>` | Run inside `.venv` without syncing first — the fastest path once installed. |
| `uv run <cmd>` | Syncs, then runs. Convenient, and safe: the lockfile pins the CUDA build. |
| `uv pip install -e .` | Install this project into the active environment. |
| `uv sync --reinstall-package torch --reinstall-package torchvision` | Reinstall the locked torch build — the repair for a corrupt install. |
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
