#!/usr/bin/env bash
# Train in the terminal dashboard -- live loss, progress and samples.
#
#   ./scripts/tui.sh mnist                 # opens paused; press s to start
#   ./scripts/tui.sh mnist --start         # start straight away
#   ./scripts/tui.sh cifar10 --device cuda
#   ./scripts/tui.sh smoke --start --epochs 1
#
# Same arguments as train.sh: the first names a config in configs/, the rest is
# forwarded to `tinydiffusion tui`. Needs the 'tui' extra --
# `uv sync --extra tui` if the dashboard reports itself missing.

source "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

args=(tui)
if [[ $# -gt 0 && "$1" != -* ]]; then
    resolve_config "$1"
    shift
    args+=(--config "$resolved")
fi

run "${args[@]}" "$@"
