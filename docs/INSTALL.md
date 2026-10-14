# Installation

One command, and an NVIDIA GPU is used automatically if you have one:

```bash
git clone https://github.com/HilthonTT/TinyDiffusion.git
cd TinyDiffusion
uv sync --all-extras --dev
```

That creates `.venv/` in the repo and installs the project into it in editable
mode. For scale, one MNIST epoch at the shipped settings is roughly **1.8
minutes on an RTX 5060** and **29 minutes on a CPU** — worth confirming you got
the GPU before starting a real run.

Part of [Usage](../USAGE.md).

- [Requirements](#requirements)
- [Using a GPU](#using-a-gpu)
- [Run something](#run-something)
- [Installing from PyPI](#installing-from-pypi)
- [How the GPU build is selected](#how-the-gpu-build-is-selected)
- [What gets downloaded, and where](#what-gets-downloaded-and-where)
- [uv command reference](#uv-command-reference)
- [Troubleshooting](#troubleshooting)
- [Continuous integration](#continuous-integration)
- [Uninstalling](#uninstalling)

## Requirements

| | |
| --- | --- |
| Python | 3.14 (pinned in `.python-version`) |
| [uv](https://docs.astral.sh/uv/) | 0.12 or newer |
| Disk | ~3.5 GB with CUDA, ~0.7 GB with a CPU-only torch |
| GPU (optional) | NVIDIA, with a driver new enough for CUDA 13.2 |

You do **not** need a system CUDA toolkit — the PyTorch wheels bundle the
runtime. You do need a current NVIDIA **driver**; check it with `nvidia-smi`.

## Using a GPU

Training picks the GPU automatically when one is visible and falls back to the
CPU when it is not, so there is no flag to set — but the wheel has to support
your card. Check what you have:

```bash
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

A working CUDA install prints a `+cu` version, `True`, and a non-empty arch
list:

```
2.13.0+cu132 True ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

A CPU build prints `2.13.0+cpu False []` — the empty arch list is the tell.

**The arch list has to contain your card's compute capability.** Blackwell
(RTX 50-series) is `sm_120` and needs CUDA 12.8 or newer; an older build such
as `cu124` installs happily and then never sees the card.

Training also names the device on its first line:

```
6.95M parameters | mnist 32px x1 | device cuda (NVIDIA GeForce RTX 5060 Laptop GPU) | amp fp16 | ...
```

On CUDA the loop also turns on cuDNN autotuning, TF32 matmuls, and fp16
autocast (`amp = true` in the config, ignored on CPU). Measured on an RTX 5060
Laptop, one full epoch of `configs/smoke.toml` — training, sampling, and
checkpointing — takes **28 s on the GPU against 206 s on the CPU**. The gap
widens sharply for `configs/mnist.toml`, which is ~40x more compute per step.

## Run something

This is a `src/` layout, so `python src/tinydiffusion/cli/commands.py` does **not**
work — it puts that file's own directory on `sys.path` rather than `src/`. Use
the wrappers, which find the environment the package is installed in and hand
off to it:

```bash
./scripts/run.sh train --config configs/smoke.toml     # Linux, macOS
```

```powershell
.\scripts\run.ps1 train --config configs/smoke.toml    # Windows
```

`configs/smoke.toml` is a deliberately tiny run — about 28 s per epoch on an
RTX 5060, 3.5 minutes on a CPU — for checking the pipeline end to end before
committing to a real one. `configs/mnist.toml` is the real run.

## Installing from PyPI

```bash
pip install tinydiffusion                 # add [server], [plots], [tracking] or [tui] as needed
```

The wheel holds the package and nothing else: **`configs/` is not part of it.**
Every `--config configs/...` in these docs assumes a checkout. Without one,
download the TOML files you want from the
[repository](https://github.com/HilthonTT/TinyDiffusion/tree/main/configs), or
skip `--config` and set what you need with flags and `--set`.

pip also knows nothing about the CUDA index below, so on Windows it installs
PyPI's CPU-only torch. Install the CUDA build first if you want the GPU:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132`.

## How the GPU build is selected

PyPI's **Windows** torch wheel is CPU-only; the CUDA builds live on PyTorch's
own index. (PyPI's Linux wheel already bundles CUDA, so this is a Windows
problem only.) `pyproject.toml` points the Windows wheels at that index:

```toml
[[tool.uv.index]]
name = "pytorch-cu132"
url = "https://download.pytorch.org/whl/cu132"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu132", marker = "sys_platform == 'win32'" }]
torchvision = [{ index = "pytorch-cu132", marker = "sys_platform == 'win32'" }]
```

`explicit = true` means only the packages named there come from that index;
everything else stays on PyPI. Both builds are recorded in `uv.lock` and picked
by platform marker, so `uv sync` gives CUDA on Windows and Linux and CPU on
macOS.

Because this is in the lockfile, **a later `uv sync` or `uv run` cannot undo
it**. That was the failure mode before: the CUDA build had to be installed by
hand with `uv pip install --torch-backend=auto`, which sat outside the
lockfile, and the next sync — including the implicit one a bare `uv run` does —
silently put the CPU wheel back.

### Skipping the CUDA index

`--no-sources` ignores the table above, so every package comes from PyPI:

```bash
uv sync --all-extras --dev --no-sources
```

What that gives you depends on the platform, because it only changes where the
**Windows** wheels come from:

| Platform | `uv sync` | `uv sync --no-sources` |
| --- | --- | --- |
| Windows | CUDA (PyTorch's index) | CPU-only (PyPI) |
| Linux | CUDA (PyPI) | CUDA (PyPI) — unchanged |
| macOS | CPU / MPS (PyPI) | CPU / MPS (PyPI) — unchanged |

So on Linux it is **not** a CPU-only install: PyPI's Linux torch pulls in the
`nvidia-*` runtime packages either way. That build runs fine on a machine with
no GPU — it is just larger. `--no-sources` also re-resolves without the Windows
entries, which rewrites `uv.lock`; don't commit the result.

### Optional: `torch.compile` on Windows

The `compile` config option needs Triton, which PyTorch's Windows wheels do not
include:

```bash
uv pip install triton-windows
```

Without it, a run with `compile = true` reports that it is training eagerly
instead. Nothing else needs it.

### Moving to a different CUDA version

Change the index URL and its name in `pyproject.toml` — say `cu132` to
`cu130` — then re-run `uv lock`.

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

MNIST downloads itself the first time you train and is reused afterwards. In a
multi-GPU run one process per machine downloads it while the others wait.
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
| `uv lock --check` | Fail if `uv.lock` is out of date with `pyproject.toml`, without changing it. CI runs this. |

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

## Troubleshooting

### `torch.cuda.is_available()` is `False`

Check the version string first. `+cpu` means a CPU wheel is installed: run
`uv sync --all-extras --dev` (without `--no-sources`) and check again. `+cu132`
with `False` is usually a driver too old for the CUDA runtime — compare
`nvidia-smi` against the CUDA version in the wheel.

### It worked, then went back to CPU

This should no longer happen — the CUDA build is in the lockfile. If it does,
something ran `--no-sources`, or `uv pip install`ed torch over the top. Re-run
`uv sync --all-extras --dev`.

### CUDA is available but the GPU is idle and training is slow

Check the arch list, as under [Using a GPU](#using-a-gpu). A wheel that does
not include your card's compute capability reports `True` and then falls back.
For an RTX 50-series card you need `sm_120`, which means CUDA 12.8 or newer.

### `DLL load failed`, `WinError 126`, or `Installation may result in an incomplete environment`

An interrupted install left CPU and CUDA files mixed in `.venv/`. uv warns with
`Failed to uninstall package ... due to missing RECORD file`. Rebuild:

```powershell
Remove-Item -Recurse -Force .venv
uv sync --all-extras --dev
```

### `ModuleNotFoundError: No module named 'tinydiffusion'`

You ran `python src/tinydiffusion/cli/commands.py`. Use `./scripts/run.sh` or `.\scripts\run.ps1`, or
`uv run --no-sync tinydiffusion ...`.

### `run.ps1: tinydiffusion is not installed in ...`

The wrapper found a Python without the package. Run `uv sync --all-extras
--dev`, or point it at the right interpreter with `$env:PYTHON`.

### `did not find executable at 'C:\Python314\python.exe'`

Or any other path in that message — and on Linux or macOS the same fault reads
`No such file or directory`. The venv in `.venv/` is stale: it records the
interpreter it was built against in `.venv/pyvenv.cfg`, and delegates to it
every time it starts. Upgrade Python, move the install, or uninstall it, and
every `python` in the venv stops working — the error names the *base*
interpreter, which is why it does not obviously point at `.venv/` at all.

The catch is that **`uv sync` does not fix this**: it reuses the existing venv
rather than rebuilding it, so the install appears to succeed and the wrapper
still fails. Delete the venv first:

```powershell
Remove-Item -Recurse -Force .venv
uv sync --all-extras --dev
```

```bash
rm -rf .venv && uv sync --all-extras --dev
```

The wrappers detect this case and print those commands for you.

### The dataset downloads every time

It should not — MNIST lands in `data/` (63 MB) and is reused. If `data_root`
points somewhere transient, or you run from a different working directory each
time, it will re-download.

## Continuous integration

CI syncs with `--no-sources`, so the Windows runner gets PyPI's CPU wheel
rather than gigabytes of CUDA from PyTorch's index. The Linux runners still get
PyPI's CUDA-bundling wheel — see [Skipping the CUDA index](#skipping-the-cuda-index)
— which is larger than needed but runs on a machine without a GPU. Every job
checks `uv lock --check` before it syncs, since the sync rewrites the
lockfile. The test suite is CPU-only; anything needing a GPU is marked `gpu`
and deselected with `-m "not gpu"`.

## Uninstalling

Everything lives in `.venv/` and, for datasets, `data/`. Delete both to reclaim
the space; uv's wheel cache is separate and cleared with `uv cache clean`.
