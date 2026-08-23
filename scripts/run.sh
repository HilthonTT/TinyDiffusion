#!/usr/bin/env bash
# Run the TinyDiffusion CLI with the project's interpreter.
#
#   ./scripts/run.sh train  --config configs/smoke.toml
#   ./scripts/run.sh sample --checkpoint runs/smoke/checkpoints/last.pt --out runs/smoke/gen.png
#
# This is a src-layout project, so `tinydiffusion` is importable only from an
# environment it has been installed into -- running the .py files by path fails
# with ModuleNotFoundError. This script finds such an environment and hands off.
# Override the choice with PYTHON=/path/to/python ./scripts/run.sh ...
#
# On Windows use PowerShell (.\scripts\run.ps1) or Git Bash. WSL cannot use a Windows
# .venv: its python.exe is a Windows binary, and WSL only runs those when
# interop is enabled.

set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

python_bin=""
# Set when an interpreter runs but lacks the package, so the error message can
# name the one worth installing into.
runnable=""

# Results come back through globals rather than stdout: a command substitution
# would run this in a subshell, where the assignment to `runnable` would be
# thrown away along with it.
#
# Candidates are probed by execution, not by a -x test: every file on a Windows
# drive mounted into WSL reports as executable, including .exe files that WSL
# cannot actually launch.
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
    # Unix layout first: in WSL it is the only one that can run. Only .venv is
    # probed, since that is what `uv sync` creates; a second environment in the
    # repo is the thing this script used to silently fall back into.
    candidates=(
        "$root/.venv/bin/python"
        "$root/.venv/Scripts/python.exe"
        "$(command -v python3 || true)"
        "$(command -v python || true)"
    )
fi

# A venv records its base interpreter in pyvenv.cfg and delegates to it at
# startup. Move or uninstall that Python and every interpreter in the venv stops
# working, with an error naming the base rather than the venv that is actually
# the problem -- and `uv sync` reuses such a venv rather than rebuilding it, so
# the obvious fix does not work either.
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
