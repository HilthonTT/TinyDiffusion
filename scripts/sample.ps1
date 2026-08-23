<#
.SYNOPSIS
    Draw a grid of images from a checkpoint.

.DESCRIPTION
    The first argument names the checkpoint -- a file, a directory holding a
    last.pt, or a bare word for checkpoints\<word>\last.pt -- and defaults to
    checkpoints\last.pt, which is where configs\mnist.toml writes. Everything
    after it is forwarded to `tinydiffusion sample`.

.EXAMPLE
    .\scripts\sample.ps1

.EXAMPLE
    .\scripts\sample.ps1 cifar10 --num-images 16

.EXAMPLE
    .\scripts\sample.ps1 mnist --labels 7 --guidance 4 --out contents\sevens.png
#>

. (Join-Path $PSScriptRoot '_lib.ps1')

$forwarded = @($args)
$checkpoint = 'checkpoints\last.pt'
if ($forwarded.Count -gt 0 -and $forwarded[0] -notlike '-*') {
    $checkpoint = $forwarded[0]
    $forwarded = @($forwarded | Select-Object -Skip 1)
}

$cli = @('sample', '--checkpoint', (Resolve-CheckpointPath $checkpoint))
Invoke-TinyDiffusion ($cli + $forwarded)
