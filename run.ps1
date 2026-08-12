<#
.SYNOPSIS
    Run the TinyDiffusion CLI with the project's interpreter.

.DESCRIPTION
    This is a src-layout project, so `tinydiffusion` is importable only from an
    environment it has been installed into -- `python src\tinydiffusion\cli.py`
    fails with ModuleNotFoundError because that puts the file's own directory on
    sys.path rather than src\. This script finds an environment that has the
    package and hands off to it.

    Set $env:PYTHON to force a particular interpreter.

.EXAMPLE
    .\run.ps1 train  --config configs\smoke.toml

.EXAMPLE
    .\run.ps1 sample --checkpoint runs\smoke\checkpoints\last.pt --out runs\smoke\gen.png
#>

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($env:PYTHON) {
    $candidates = @($env:PYTHON)
}
else {
    $candidates = @(
        (Join-Path $root '.venv\Scripts\python.exe'),
        (Join-Path $root 'venv\Scripts\python.exe')
    )
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { $candidates += $onPath.Source }
}

# Probed by running each one: an interpreter that exists is not necessarily one
# that has the package installed. find_spec rather than a bare import, so a
# miss costs an exit code instead of a traceback on stderr -- which PowerShell
# would surface as a NativeCommandError.
$probe = "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('tinydiffusion') else 1)"
$interpreter = $null
$runnable = $null
foreach ($candidate in $candidates) {
    if (-not (Test-Path $candidate)) { continue }
    & $candidate -c $probe
    if ($LASTEXITCODE -eq 0) { $interpreter = $candidate; break }
    if (-not $runnable) { $runnable = $candidate }
}

if (-not $interpreter) {
    if ($runnable) {
        Write-Host "run.ps1: tinydiffusion is not installed in $runnable"
        Write-Host "         install it with:  uv sync --all-extras --dev"
        Write-Host "                      or:  & '$runnable' -m pip install -e ."
    }
    else {
        Write-Host 'run.ps1: found no Python with tinydiffusion installed.'
        Write-Host '         Set one up with:  uv sync --all-extras --dev'
    }
    exit 1
}

$forwarded = $args
if ($forwarded.Count -eq 0) { $forwarded = @('--help') }

# tqdm draws the progress bar on stderr; with 'Stop' still in force PowerShell
# would treat that as a fatal error partway through training.
$ErrorActionPreference = 'Continue'

& $interpreter -m tinydiffusion.cli @forwarded
exit $LASTEXITCODE
