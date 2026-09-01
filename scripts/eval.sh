#!/usr/bin/env bash

source "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

checkpoint="checkpoints/last.pt"
if [[ $# -gt 0 && "$1" != -* ]]; then
    checkpoint="$1"
    shift
fi
resolve_checkpoint "$checkpoint"

run eval --checkpoint "$resolved" "$@"
