$ErrorActionPreference = 'Stop'

$script:LibDir = $PSScriptRoot
$script:Root = Split-Path -Parent $PSScriptRoot

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
