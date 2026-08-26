# Running the CLI

How to invoke `tinydiffusion` — the wrapper scripts, and what they do that a
bare `python` cannot.

Part of [Usage](../../USAGE.md).

## Running the CLI

The wrappers locate an interpreter that actually has the package installed and
forward everything else to the CLI. Use `scripts/run.ps1` from PowerShell,
`scripts/run.sh` from Git Bash, WSL, Linux, or macOS:

```powershell
.\scripts\run.ps1 train  --config configs\mnist.toml
.\scripts\run.ps1 sample --checkpoint checkpoints\last.pt --num-images 8
```

```bash
./scripts/run.sh train  --config configs/mnist.toml
./scripts/run.sh sample --checkpoint checkpoints/last.pt --num-images 8
```

Set `PYTHON` to force a specific interpreter: `PYTHON=/usr/bin/python3 ./scripts/run.sh …`.
With no arguments a wrapper prints the CLI help. `--version` (or `-V`) prints
the installed version and exits:

```bash
./scripts/run.sh --version      # tinydiffusion 0.1.0
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
- `bash .\scripts\run.sh` — bash reads the backslashes as escapes and
  looks for a file named `.scriptsrun.sh`. Use `./scripts/run.sh` or
  `bash scripts/run.sh`.
