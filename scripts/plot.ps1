. (Join-Path $PSScriptRoot '_lib.ps1')

$forwarded = @($args)
$runs = @()
while ($forwarded.Count -gt 0 -and $forwarded[0] -notlike '-*') {
    $runs += (Resolve-RunPath $forwarded[0])
    $forwarded = @($forwarded | Select-Object -Skip 1)
}
if ($runs.Count -eq 0) { $runs = @((Resolve-RunPath 'mnist')) }

Invoke-TinyDiffusion (@('plot') + $runs + $forwarded)
