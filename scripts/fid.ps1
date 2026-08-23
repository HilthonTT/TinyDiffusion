<#
.SYNOPSIS
    Score a checkpoint's samples against real data: FID, and optionally KID and
    precision/recall.

.DESCRIPTION
    Checkpoint naming works as it does in sample.ps1; everything after it is
    forwarded to `tinydiffusion fid`. Below a few thousand images the score is
    mostly its own bias -- --kid is the one that stays comparable down there.

.EXAMPLE
    .\scripts\fid.ps1 mnist --num-images 2000 --kid

.EXAMPLE
    .\scripts\fid.ps1 cifar10 --num-images 10000
#>

. (Join-Path $PSScriptRoot '_lib.ps1')

$forwarded = @($args)
$checkpoint = 'checkpoints\last.pt'
if ($forwarded.Count -gt 0 -and $forwarded[0] -notlike '-*') {
    $checkpoint = $forwarded[0]
    $forwarded = @($forwarded | Select-Object -Skip 1)
}

$cli = @('fid', '--checkpoint', (Resolve-CheckpointPath $checkpoint))
Invoke-TinyDiffusion ($cli + $forwarded)
