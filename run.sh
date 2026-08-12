#!/usr/bin/env bash
# Run the TinyDiffusion CLI with the project's interpreter.
#
#   ./run.sh train  --config configs/smoke.toml
#   ./run.sh sample --checkpoint runs/smoke/checkpoints/last.pt --out runs/smoke/gen.png
#
# This is a src-layout project, so `tinydiffusion` is importable only from an
# environment it has been installed into -- running the .py files by path fails
# with ModuleNotFoundError. This script finds such an environment and hands off.
# Override the choice with PYTHON=/path/to/python ./run.sh ...

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

find_python() {
    if [[ -n "${PYTHON:-}" ]]; then
        printf '%s\n' "$PYTHON"
        return
    fi
    # Both Unix (bin/) and Windows (Scripts/) venv layouts, .venv before venv.
    local candidate
    for candidate in \
        "$root/.venv/bin/python" \
        "$root/.venv/Scripts/python.exe" \
        "$root/venv/bin/python" \
        "$root/venv/Scripts/python.exe"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return
        fi
    done
    # No project venv: fall back to whatever is on PATH and let the import
    # check below decide whether that is good enough.
    command -v python3 || command -v python || true
}

python_bin="$(find_python)"

if [[ -z "$python_bin" ]]; then
    echo "run.sh: no Python interpreter found. Try: uv sync --all-extras --dev" >&2
    exit 1
fi

if ! "$python_bin" -c 'import tinydiffusion' >/dev/null 2>&1; then
    echo "run.sh: tinydiffusion is not installed in $python_bin" >&2
    echo "        install it with:  uv sync --all-extras --dev" >&2
    echo "                     or:  $python_bin -m pip install -e ." >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    set -- --help
fi

exec "$python_bin" -m tinydiffusion.cli "$@"
