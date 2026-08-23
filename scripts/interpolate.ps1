<#
.SYNOPSIS
    Walk between two latents and sample every point along the way.

.DESCRIPTION
    Checkpoint naming works as it does in sample.ps1; everything after it is
    forwarded to `tinydiffusion interpolate`.

.EXAMPLE
    .\scripts\interpolate.ps1 mnist --labels 7 --steps 10

.EXAMPLE
    .\scripts\interpolate.ps1 cifar10 --seed-start 1 --seed-end 2 --out contents\walk.png
#>

. (Join-Path $PSScriptRoot '_lib.ps1')

$forwarded = @($args)
$checkpoint = 'checkpoints\last.pt'
if ($forwarded.Count -gt 0 -and $forwarded[0] -notlike '-*') {
    $checkpoint = $forwarded[0]
    $forwarded = @($forwarded | Select-Object -Skip 1)
}

$cli = @('interpolate', '--checkpoint', (Resolve-CheckpointPath $checkpoint))
Invoke-TinyDiffusion ($cli + $forwarded)
