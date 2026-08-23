#!/usr/bin/env bash
# Train a model from one of the configs in configs/.
#
#   ./scripts/train.sh                     # the CLI's own defaults, no config
#   ./scripts/train.sh smoke               # configs/smoke.toml, a minute or so
#   ./scripts/train.sh mnist
#   ./scripts/train.sh cifar10 --epochs 5 --device cpu
#   ./scripts/train.sh mnist --dataset fashion_mnist --set lr=1e-4
#   ./scripts/train.sh path/to/own.toml --tensorboard
#
# The first argument names the config; everything after it is forwarded to
# `tinydiffusion train` as given, so --resume, --set, --seed and the rest all
# still work.

source "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

args=(train)
if [[ $# -gt 0 && "$1" != -* ]]; then
    resolve_config "$1"
    shift
    args+=(--config "$resolved")
fi

run "${args[@]}" "$@"
