#!/usr/bin/env bash
# Score a checkpoint's likelihood on held-out data.
#
#   ./scripts/eval.sh                          # checkpoints/last.pt, test split
#   ./scripts/eval.sh cifar10
#   ./scripts/eval.sh mnist --split train
#   ./scripts/eval.sh mnist --no-ema --device cpu
#
# Checkpoint naming works as it does in sample.sh; everything after it is
# forwarded to `tinydiffusion eval`.

source "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

checkpoint="checkpoints/last.pt"
if [[ $# -gt 0 && "$1" != -* ]]; then
    checkpoint="$1"
    shift
fi
resolve_checkpoint "$checkpoint"

run eval --checkpoint "$resolved" "$@"
