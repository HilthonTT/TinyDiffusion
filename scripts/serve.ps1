<#
.SYNOPSIS
    Serve a checkpoint over HTTP.

.DESCRIPTION
    Checkpoint naming works as it does in sample.ps1; everything after it is
    forwarded to `tinydiffusion serve`. Needs the 'server' extra --
    `uv sync --extra server`. The API is unauthenticated, so --host is worth
    widening only behind something that is not.

.EXAMPLE
    .\scripts\serve.ps1

.EXAMPLE
    .\scripts\serve.ps1 cifar10 --port 8080
#>

. (Join-Path $PSScriptRoot '_lib.ps1')

$forwarded = @($args)
$checkpoint = 'checkpoints\last.pt'
if ($forwarded.Count -gt 0 -and $forwarded[0] -notlike '-*') {
    $checkpoint = $forwarded[0]
    $forwarded = @($forwarded | Select-Object -Skip 1)
}

$cli = @('serve', '--checkpoint', (Resolve-CheckpointPath $checkpoint))
Invoke-TinyDiffusion ($cli + $forwarded)
