<#
.SYNOPSIS
    Draw a run's metrics as a figure.

.DESCRIPTION
    Every leading argument names a run: a bare word is runs\<word>, a path or a
    .jsonl file is taken as given. More than one draws them together on shared
    axes, which is how a sweep is compared. Defaults to runs\mnist. The rest is
    forwarded to `tinydiffusion plot`.

.EXAMPLE
    .\scripts\plot.ps1 cifar10 --out contents\cifar.png

.EXAMPLE
    .\scripts\plot.ps1 baseline min_snr
#>

. (Join-Path $PSScriptRoot '_lib.ps1')

$forwarded = @($args)
$runs = @()
while ($forwarded.Count -gt 0 -and $forwarded[0] -notlike '-*') {
    $runs += (Resolve-RunPath $forwarded[0])
    $forwarded = @($forwarded | Select-Object -Skip 1)
}
if ($runs.Count -eq 0) { $runs = @((Resolve-RunPath 'mnist')) }

Invoke-TinyDiffusion (@('plot') + $runs + $forwarded)
