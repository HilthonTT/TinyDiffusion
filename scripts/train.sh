#!/usr/bin/env bash

source "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

args=(train)
if [[ $# -gt 0 && "$1" != -* ]]; then
    resolve_config "$1"
    shift
    args+=(--config "$resolved")
fi

run "${args[@]}" "$@"
