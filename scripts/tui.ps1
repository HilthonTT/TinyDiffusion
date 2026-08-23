<#
.SYNOPSIS
    Train in the terminal dashboard -- live loss, progress and samples.

.DESCRIPTION
    Same arguments as train.ps1: the first names a config in configs\, the rest
    is forwarded to `tinydiffusion tui`. The dashboard opens paused and starts
    on 's', or immediately with --start.

    Needs the 'tui' extra -- `uv sync --extra tui` if it reports itself
    missing.

.EXAMPLE
    .\scripts\tui.ps1 mnist

.EXAMPLE
    .\scripts\tui.ps1 mnist --start

.EXAMPLE
    .\scripts\tui.ps1 cifar10 --device cuda
#>

. (Join-Path $PSScriptRoot '_lib.ps1')

$forwarded = @($args)
$cli = @('tui')
if ($forwarded.Count -gt 0 -and $forwarded[0] -notlike '-*') {
    $cli += @('--config', (Resolve-ConfigPath $forwarded[0]))
    $forwarded = @($forwarded | Select-Object -Skip 1)
}

Invoke-TinyDiffusion ($cli + $forwarded)
