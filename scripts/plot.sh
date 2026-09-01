#!/usr/bin/env bash

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
