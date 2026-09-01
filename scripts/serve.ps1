. (Join-Path $PSScriptRoot '_lib.ps1')

$forwarded = @($args)
$checkpoint = 'checkpoints\last.pt'
if ($forwarded.Count -gt 0 -and $forwarded[0] -notlike '-*') {
    $checkpoint = $forwarded[0]
    $forwarded = @($forwarded | Select-Object -Skip 1)
}

$cli = @('serve', '--checkpoint', (Resolve-CheckpointPath $checkpoint))
Invoke-TinyDiffusion ($cli + $forwarded)
