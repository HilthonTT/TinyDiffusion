#!/usr/bin/env bash
# Draw a grid of images from a checkpoint.
#
#   ./scripts/sample.sh                                # checkpoints/last.pt
#   ./scripts/sample.sh cifar10 --num-images 16        # checkpoints/cifar10/last.pt
#   ./scripts/sample.sh runs/smoke/checkpoints/last.pt
#   ./scripts/sample.sh mnist --labels 7 --guidance 4 --out contents/sevens.png
#   ./scripts/sample.sh mnist --sampler dpmpp --steps 20
#
# The first argument names the checkpoint -- a file, a directory holding a
# last.pt, or a bare word for checkpoints/<word>/last.pt -- and defaults to
# checkpoints/last.pt, which is where configs/mnist.toml writes. Everything
# after it is forwarded to `tinydiffusion sample`.

source "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

checkpoint="checkpoints/last.pt"
if [[ $# -gt 0 && "$1" != -* ]]; then
    checkpoint="$1"
    shift
fi
resolve_checkpoint "$checkpoint"

run sample --checkpoint "$resolved" "$@"
