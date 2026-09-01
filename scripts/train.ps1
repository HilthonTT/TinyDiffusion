. (Join-Path $PSScriptRoot '_lib.ps1')

$forwarded = @($args)
$cli = @('train')
if ($forwarded.Count -gt 0 -and $forwarded[0] -notlike '-*') {
    $cli += @('--config', (Resolve-ConfigPath $forwarded[0]))
    $forwarded = @($forwarded | Select-Object -Skip 1)
}

Invoke-TinyDiffusion ($cli + $forwarded)
