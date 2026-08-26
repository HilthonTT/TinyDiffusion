# Troubleshooting

The failures that come up most, and what each one actually means.

Part of [Usage](../../USAGE.md).

## Troubleshooting

**`ModuleNotFoundError: No module named 'tinydiffusion'`**
The interpreter you used is not one the package was installed into. A second
virtualenv in the repo is the usual culprit — `uv` manages `.venv` and nothing
else, so an environment under any other name will not have the package however
recently you activated it. Check with `echo $env:VIRTUAL_ENV` (PowerShell) or
`echo $VIRTUAL_ENV` (bash): if it names anything but `.venv`, deactivate and
reopen the terminal. Use `./scripts/run.sh` / `.\scripts\run.ps1`, which only ever pick
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

```bash
uv sync --all-extras --dev --reinstall-package torch --reinstall-package torchvision
```

Repairing through `uv sync` rather than `uv pip install` keeps the lockfile's
CUDA index in play, so the replacement is the same build the project asks for.

Avoid running `uv sync`/`uv run` against the project while something else is
using `.venv`; that race is a common way to get here.

**Training says `device cpu` on a machine with a GPU**
The installed torch is a CPU-only build — usually the result of a sync that
passed `--no-sources`. Re-run `uv sync --all-extras --dev` without it, then see
[Using a GPU](install.md#using-a-gpu) to confirm.

**`no CUDA device visible, falling back to CPU`**
`--device cuda` was asked for but torch cannot see a GPU. The run continues on
the CPU rather than failing; same fix as above.

**CUDA out of memory**
Lower `batch_size`, then `base_channels`. Sampling `num_samples` also
allocates a batch at once.

**Windows: the install step takes minutes and disk usage doubles**
uv's cache is on a different drive from `.venv/`. See
[uv's cache and your disk](install.md#uvs-cache-and-your-disk).
