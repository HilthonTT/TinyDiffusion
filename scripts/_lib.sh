#!/usr/bin/env bash
# Shared helpers for the task wrappers in this directory. Sourced by them,
# never run on its own.
#
# Each wrapper is a thin front end over run.sh: it resolves one shorthand
# argument -- a config name, a checkpoint, a run directory -- and forwards
# everything else to the CLI untouched. Whatever the CLI accepts still works,
# PYTHON= still picks the interpreter, and the exit code is still the CLI's,
# because run.sh is what ends up running.

set -euo pipefail

lib_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$lib_dir/.." && pwd)"
me="$(basename -- "${BASH_SOURCE[1]:-$0}")"

# The paths inside a config -- ckpt_dir, log_dir, data_root -- are written
# relative to the repo root, so a run started from somewhere else would scatter
# its output across the filesystem. Everything below, and the CLI itself, runs
# from the root instead. A path given on the command line is therefore read
# from the root too, not from the directory you happen to be standing in.
cd -- "$root"

die() {
    local line
    printf '%s: %s\n' "$me" "$1" >&2
    shift
    for line in "$@"; do
        printf '%*s  %s\n' "${#me}" '' "$line" >&2
    done
    exit 1
}

# Hands off to the wrapper that finds an interpreter with the package in it.
# exec, so signals and the exit code pass straight through -- Ctrl-C during
# training has to reach Python, not a shell sitting in front of it.
run() {
    exec "$lib_dir/run.sh" "$@"
}

config_names() {
    local path names=()
    for path in configs/*.toml; do
        [[ -e "$path" ]] || continue
        names+=("$(basename -- "$path" .toml)")
    done
    printf '%s' "${names[*]}"
}

# A bare word names a file in configs/; anything with a separator in it or a
# .toml suffix is a path, and is taken as given.
resolve_config() {
    case "$1" in
        */* | *\\* | *.toml) resolved="$1" ;;
        *) resolved="configs/$1.toml" ;;
    esac
    [[ -f "$resolved" ]] || die "no config at $resolved" "configs/ holds: $(config_names)"
}

# Checkpoints are named the way you would say them out loud: a file as given, a
# directory whose last.pt is wanted, or a bare word naming a run's directory
# under checkpoints/ -- which is where the configs put them.
resolve_checkpoint() {
    local candidate
    for candidate in "$1" "$1/last.pt" "checkpoints/$1/last.pt" "checkpoints/$1.pt"; do
        if [[ -f "$candidate" ]]; then
            resolved="$candidate"
            return
        fi
    done
    die "no checkpoint for '$1'" \
        "tried: $1, $1/last.pt, checkpoints/$1/last.pt, checkpoints/$1.pt" \
        "train one first:  ./scripts/train.sh mnist"
}

# Run directories hold metrics.jsonl. A bare word is one under runs/, which is
# where the configs' log_dir points.
resolve_run() {
    case "$1" in
        */* | *\\* | *.jsonl) resolved="$1" ;;
        *) resolved="runs/$1" ;;
    esac
    [[ -e "$resolved" ]] || die "no run at $resolved" \
        "a run appears there once training has written a metrics.jsonl"
}
