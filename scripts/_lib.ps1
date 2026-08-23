<#
.SYNOPSIS
    Shared helpers for the task wrappers in this directory.

.DESCRIPTION
    Dot-sourced by them, never run on its own.

    Each wrapper is a thin front end over run.ps1: it resolves one shorthand
    argument -- a config name, a checkpoint, a run directory -- and forwards
    everything else to the CLI untouched. Whatever the CLI accepts still works,
    $env:PYTHON still picks the interpreter, and the exit code is still the
    CLI's, because run.ps1 is what ends up running.
#>

$ErrorActionPreference = 'Stop'

$script:LibDir = $PSScriptRoot
$script:Root = Split-Path -Parent $PSScriptRoot

# Error messages name the wrapper the user actually typed, not this file. The
# first frame with a script behind it that is not this one is that wrapper.
function Get-WrapperName {
    foreach ($frame in Get-PSCallStack) {
        if ($frame.ScriptName -and (Split-Path -Leaf $frame.ScriptName) -ne '_lib.ps1') {
            return (Split-Path -Leaf $frame.ScriptName)
        }
    }
    return 'tinydiffusion'
}

function Stop-Wrapper {
    param([string]$Message, [string[]]$Detail = @())

    $me = Get-WrapperName
    Write-Host "${me}: $Message"
    foreach ($line in $Detail) { Write-Host ((' ' * ($me.Length + 1)) + " $line") }
    exit 1
}

function Get-RootedPath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return (Join-Path $script:Root $Path)
}

# A bare word names a file in configs\; anything with a separator in it or a
# .toml suffix is a path, and is taken as given.
function Resolve-ConfigPath {
    param([string]$Name)

    $path = if ($Name -match '[\\/]' -or $Name -like '*.toml') { $Name } else { "configs\$Name.toml" }
    if (-not (Test-Path (Get-RootedPath $path) -PathType Leaf)) {
        $names = (Get-ChildItem (Join-Path $script:Root 'configs') -Filter *.toml |
            ForEach-Object { $_.BaseName }) -join ' '
        Stop-Wrapper "no config at $path" @("configs\ holds: $names")
    }
    return $path
}

# Checkpoints are named the way you would say them out loud: a file as given, a
# directory whose last.pt is wanted, or a bare word naming a run's directory
# under checkpoints\ -- which is where the configs put them.
function Resolve-CheckpointPath {
    param([string]$Name)

    $candidates = @($Name, "$Name\last.pt", "checkpoints\$Name\last.pt", "checkpoints\$Name.pt")
    foreach ($candidate in $candidates) {
        if (Test-Path (Get-RootedPath $candidate) -PathType Leaf) { return $candidate }
    }
    Stop-Wrapper "no checkpoint for '$Name'" @(
        "tried: $($candidates -join ', ')",
        'train one first:  .\scripts\train.ps1 mnist'
    )
}

# Run directories hold metrics.jsonl. A bare word is one under runs\, which is
# where the configs' log_dir points.
function Resolve-RunPath {
    param([string]$Name)

    $path = if ($Name -match '[\\/]' -or $Name -like '*.jsonl') { $Name } else { "runs\$Name" }
    if (-not (Test-Path (Get-RootedPath $path))) {
        Stop-Wrapper "no run at $path" @(
            'a run appears there once training has written a metrics.jsonl'
        )
    }
    return $path
}

# Hands off to the wrapper that finds an interpreter with the package in it.
#
# The paths inside a config -- ckpt_dir, log_dir, data_root -- are written
# relative to the repo root, so a run started from somewhere else would scatter
# its output across the filesystem. The CLI runs from the root instead, which
# means a path given on the command line is read from the root too, not from
# the directory you happen to be standing in. Push/Pop rather than Set-Location
# because a script's working directory is the session's: leaving the caller
# somewhere they did not ask to be is a rude way to end.
function Invoke-TinyDiffusion {
    param([string[]]$Arguments)

    Push-Location $script:Root
    try {
        & (Join-Path $script:LibDir 'run.ps1') @Arguments
        $code = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    exit $code
}
