#!/usr/bin/env bash

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

python_bin=""
runnable=""

find_python() {
    local candidate
    for candidate in "$@"; do
        [[ -n "$candidate" ]] || continue
        "$candidate" -c 'pass' >/dev/null 2>&1 || continue
        if "$candidate" -c 'import tinydiffusion' >/dev/null 2>&1; then
            python_bin="$candidate"
            return 0
        fi
        [[ -n "$runnable" ]] || runnable="$candidate"
    done
    return 1
}

if [[ -n "${PYTHON:-}" ]]; then
    candidates=("$PYTHON")
else
    candidates=(
        "$root/.venv/bin/python"
        "$root/.venv/Scripts/python.exe"
        "$(command -v python3 || true)"
        "$(command -v python || true)"
    )
fi

stale_venv_home() {
    local cfg="$root/.venv/pyvenv.cfg" base
    [[ -f "$cfg" ]] || return 1
    base="$(sed -n 's/^[[:space:]]*home[[:space:]]*=[[:space:]]*\(.*[^[:space:]]\)[[:space:]]*$/\1/p' "$cfg" | head -n 1)"
    [[ -n "$base" && ! -d "$base" ]] || return 1
    printf '%s\n' "$base"
}

if ! find_python "${candidates[@]}"; then
    if stale_home="$(stale_venv_home)"; then
        echo "run.sh: $root/.venv was built against a Python that is no longer there:" >&2
        echo "        $stale_home" >&2
        echo "        The venv cannot start, and 'uv sync' will reuse it as is. Rebuild it:" >&2
        echo "          rm -rf '$root/.venv'" >&2
        echo "          uv sync --all-extras --dev" >&2
    elif [[ -n "$runnable" ]]; then
        echo "run.sh: tinydiffusion is not installed in $runnable" >&2
        echo "        install it with:  uv sync --all-extras --dev" >&2
        echo "                     or:  $runnable -m pip install -e ." >&2
    else
        echo "run.sh: found no Python that this shell can run." >&2
        if grep -qi microsoft /proc/version 2>/dev/null; then
            echo "        You are in WSL, which cannot launch the Windows .venv" >&2
            echo "        under $root/.venv/Scripts/." >&2
            echo "        Use PowerShell (.\\run.ps1) or Git Bash instead, or make" >&2
            echo "        a Linux env here:  uv sync --all-extras --dev" >&2
        else
            echo "        Set one up with:  uv sync --all-extras --dev" >&2
        fi
    fi
    exit 1
fi

if [[ $# -eq 0 ]]; then
    set -- --help
fi

exec "$python_bin" -m tinydiffusion.cli "$@"
