#!/usr/bin/env bash

set -euo pipefail

lib_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$lib_dir/.." && pwd)"
me="$(basename -- "${BASH_SOURCE[1]:-$0}")"

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

resolve_config() {
    case "$1" in
        */* | *\\* | *.toml) resolved="$1" ;;
        *) resolved="configs/$1.toml" ;;
    esac
    [[ -f "$resolved" ]] || die "no config at $resolved" "configs/ holds: $(config_names)"
}

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

resolve_run() {
    case "$1" in
        */* | *\\* | *.jsonl) resolved="$1" ;;
        *) resolved="runs/$1" ;;
    esac
    [[ -e "$resolved" ]] || die "no run at $resolved" \
        "a run appears there once training has written a metrics.jsonl"
}
