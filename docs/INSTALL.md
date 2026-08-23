# Installation

One command, and an NVIDIA GPU is used automatically if you have one:

```bash
git clone https://github.com/HilthonTT/TinyDiffusion.git
cd TinyDiffusion
uv sync --all-extras --dev
```

For scale, one MNIST epoch at the shipped settings is roughly **1.8 minutes on
an RTX 5060** and **29 minutes on a CPU** — worth confirming you got the GPU
before starting a real run.

## Requirements

| | |
| --- | --- |
| Python | 3.14 |
| [uv](https://docs.astral.sh/uv/) | 0.12 or newer |
| Disk | ~3.5 GB with CUDA, ~0.7 GB CPU-only |
| GPU (optional) | NVIDIA, with a driver new enough for CUDA 13.2 |

You do **not** need a system CUDA toolkit — the PyTorch wheels bundle the
runtime. You do need a current NVIDIA **driver**; check it with `nvidia-smi`.

## Verify

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
(RTX 50-series) is `sm_120`; an older CUDA build such as `cu124` installs
happily and then never sees the card.

Training also names the device on its first line:

```
6.95M parameters | mnist 32px x1 | device cuda (NVIDIA GeForce RTX 5060 Laptop GPU) | amp fp16 | ...
```

## Run something

This is a `src/` layout, so `python src/tinydiffusion/cli.py` does **not**
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

### Forcing a CPU-only environment

`--no-sources` ignores the table above and resolves PyPI's CPU wheels. This is
what CI uses, and what you want on a machine with no NVIDIA GPU:

```bash
uv sync --all-extras --dev --no-sources
```

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

Check the arch list, as under [Verify](#verify). A wheel that does not include
your card's compute capability reports `True` and then falls back. For an RTX
50-series card you need `sm_120`, which means CUDA 12.8 or newer.

### `DLL load failed`, `WinError 126`, or `Installation may result in an incomplete environment`

An interrupted install left CPU and CUDA files mixed in `.venv/`. uv warns with
`Failed to uninstall package ... due to missing RECORD file`. Rebuild:

```powershell
Remove-Item -Recurse -Force .venv
uv sync --all-extras --dev
```

### `ModuleNotFoundError: No module named 'tinydiffusion'`

You ran `python src/tinydiffusion/cli.py`. Use `./scripts/run.sh` or `.\scripts\run.ps1`, or
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

CI syncs with `--no-sources` so hosted runners, which have no GPU, get PyPI's
CPU wheels rather than gigabytes of CUDA. The test suite is CPU-only; anything
needing a GPU is marked `gpu` and deselected with `-m "not gpu"`.

## Uninstalling

Everything lives in `.venv/` and, for datasets, `data/`. Delete both to reclaim
the space; uv's wheel cache is separate and cleared with `uv cache clean`.
