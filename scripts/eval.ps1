<#
.SYNOPSIS
    Score a checkpoint's likelihood on held-out data.

.DESCRIPTION
    Checkpoint naming works as it does in sample.ps1; everything after it is
    forwarded to `tinydiffusion eval`.

.EXAMPLE
    .\scripts\eval.ps1

.EXAMPLE
    .\scripts\eval.ps1 cifar10 --split train
#>

. (Join-Path $PSScriptRoot '_lib.ps1')

$forwarded = @($args)
$checkpoint = 'checkpoints\last.pt'
if ($forwarded.Count -gt 0 -and $forwarded[0] -notlike '-*') {
    $checkpoint = $forwarded[0]
    $forwarded = @($forwarded | Select-Object -Skip 1)
}

$cli = @('eval', '--checkpoint', (Resolve-CheckpointPath $checkpoint))
Invoke-TinyDiffusion ($cli + $forwarded)
