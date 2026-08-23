#!/usr/bin/env bash
# Walk between two latents and sample every point along the way.
#
#   ./scripts/interpolate.sh                                  # checkpoints/last.pt
#   ./scripts/interpolate.sh mnist --labels 7 --steps 10
#   ./scripts/interpolate.sh cifar10 --seed-start 1 --seed-end 2 \
#       --out contents/walk.png
#
# Checkpoint naming works as it does in sample.sh; everything after it is
# forwarded to `tinydiffusion interpolate`.

source "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

checkpoint="checkpoints/last.pt"
if [[ $# -gt 0 && "$1" != -* ]]; then
    checkpoint="$1"
    shift
fi
resolve_checkpoint "$checkpoint"

run interpolate --checkpoint "$resolved" "$@"
