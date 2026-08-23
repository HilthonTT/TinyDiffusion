<#
.SYNOPSIS
    Train a model from one of the configs in configs\.

.DESCRIPTION
    The first argument names the config: a bare word is configs\<word>.toml,
    anything with a separator in it or a .toml suffix is a path. Everything
    after it is forwarded to `tinydiffusion train` as given, so --resume,
    --set, --seed and the rest all still work. Omit it for the CLI's own
    defaults.

.EXAMPLE
    .\scripts\train.ps1 smoke

.EXAMPLE
    .\scripts\train.ps1 cifar10 --epochs 5 --device cpu

.EXAMPLE
    .\scripts\train.ps1 mnist --dataset fashion_mnist --set lr=1e-4
#>

. (Join-Path $PSScriptRoot '_lib.ps1')

$forwarded = @($args)
$cli = @('train')
if ($forwarded.Count -gt 0 -and $forwarded[0] -notlike '-*') {
    $cli += @('--config', (Resolve-ConfigPath $forwarded[0]))
    $forwarded = @($forwarded | Select-Object -Skip 1)
}

Invoke-TinyDiffusion ($cli + $forwarded)
