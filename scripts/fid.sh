#!/usr/bin/env bash
# Score a checkpoint's samples against real data: FID, and optionally KID and
# precision/recall.
#
#   ./scripts/fid.sh                                   # checkpoints/last.pt
#   ./scripts/fid.sh mnist --num-images 2000 --kid
#   ./scripts/fid.sh cifar10 --num-images 10000
#   ./scripts/fid.sh mnist --guidance 3 --precision-recall
#
# Checkpoint naming works as it does in sample.sh; everything after it is
# forwarded to `tinydiffusion fid`. Below a few thousand images the score is
# mostly its own bias -- --kid is the one that stays comparable down there.

source "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

checkpoint="checkpoints/last.pt"
if [[ $# -gt 0 && "$1" != -* ]]; then
    checkpoint="$1"
    shift
fi
resolve_checkpoint "$checkpoint"

run fid --checkpoint "$resolved" "$@"
