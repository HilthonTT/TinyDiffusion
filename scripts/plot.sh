#!/usr/bin/env bash
# Draw a run's metrics as a figure.
#
#   ./scripts/plot.sh                                  # runs/mnist
#   ./scripts/plot.sh cifar10 --out contents/cifar.png
#   ./scripts/plot.sh baseline min_snr                 # both on shared axes
#   ./scripts/plot.sh runs/smoke/metrics.jsonl --dpi 200
#
# Every leading argument names a run: a bare word is runs/<word>, a path or a
# .jsonl file is taken as given. More than one draws them together, which is
# how a sweep is compared. The rest is forwarded to `tinydiffusion plot`.

source "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

runs=()
while [[ $# -gt 0 && "$1" != -* ]]; do
    resolve_run "$1"
    runs+=("$resolved")
    shift
done

if [[ ${#runs[@]} -eq 0 ]]; then
    resolve_run mnist
    runs=("$resolved")
fi

run plot "${runs[@]}" "$@"
