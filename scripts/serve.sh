#!/usr/bin/env bash
# Serve a checkpoint over HTTP.
#
#   ./scripts/serve.sh                       # checkpoints/last.pt on loopback
#   ./scripts/serve.sh cifar10 --port 8080
#   ./scripts/serve.sh mnist --cors-origin http://localhost:5173
#
# Checkpoint naming works as it does in sample.sh; everything after it is
# forwarded to `tinydiffusion serve`. Needs the 'server' extra --
# `uv sync --extra server`. The API is unauthenticated, so --host is worth
# widening only behind something that is not.

source "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

checkpoint="checkpoints/last.pt"
if [[ $# -gt 0 && "$1" != -* ]]; then
    checkpoint="$1"
    shift
fi
resolve_checkpoint "$checkpoint"

run serve --checkpoint "$resolved" "$@"
