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
    # Only .venv, which is what `uv sync` creates. A second environment in the
    # repo is exactly what this script used to silently fall back into.
    $candidates = @((Join-Path $root '.venv\Scripts\python.exe'))
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { $candidates += $onPath.Source }
}

# Probed by running each one, in two stages. A file that exists is not
# necessarily an interpreter that starts -- a venv's python.exe is a shim that
# fails outright once the base interpreter it was built against is gone -- and
# one that starts does not necessarily have the package. Collapsing the two
# into a single probe reports a broken interpreter as a missing package, which
# sends you off reinstalling something that is already installed.
#
# find_spec rather than a bare import, so a miss costs an exit code instead of
# a traceback on stderr -- which PowerShell would surface as a
# NativeCommandError.
$probe = "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('tinydiffusion') else 1)"
$interpreter = $null
$runnable = $null
$broken = @()

# 'Continue' for the probes: a stage-1 failure writes the launcher's complaint
# to stderr, and redirecting a native command's stderr under 'Stop' turns each
# line into a terminating NativeCommandError. The point of the probe is to
# survive that and carry on to the next candidate.
$strict = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
foreach ($candidate in $candidates) {
    if (-not (Test-Path $candidate)) { continue }
    # Stage 1: does it start at all? Output discarded; only the code matters.
    & $candidate -c 'pass' 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { $broken += $candidate; continue }
    # Stage 2: does it have the package?
    & $candidate -c $probe 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { $interpreter = $candidate; break }
    if (-not $runnable) { $runnable = $candidate }
}
$ErrorActionPreference = $strict

# A venv records its base interpreter in pyvenv.cfg and delegates to it at
# startup. Move or uninstall that Python -- a version upgrade relocating from
# C:\Python314 to %LOCALAPPDATA% will do it -- and every python.exe in the venv
# dies with "did not find executable at ...", which says nothing about the venv
# that is actually the problem. `uv sync` reuses such a venv rather than
# rebuilding it, so the obvious fix does not work either.
function Get-StaleVenvHome {
    param([string]$VenvRoot)

    $cfg = Join-Path $VenvRoot 'pyvenv.cfg'
    if (-not (Test-Path $cfg)) { return $null }
    # Not $home: that is an automatic variable, and shadowing it in a script
    # anyone might dot-source is a rude thing to do.
    $match = Get-Content $cfg | Select-String -Pattern '^\s*home\s*=\s*(.+?)\s*$' | Select-Object -First 1
    if (-not $match) { return $null }
    $base = $match.Matches[0].Groups[1].Value
    if ($base -and -not (Test-Path $base)) { return $base }
    return $null
}

if (-not $interpreter) {
    $venv = Join-Path $root '.venv'
    $staleHome = if ($broken) { Get-StaleVenvHome $venv } else { $null }

    if ($staleHome) {
        Write-Host "run.ps1: $venv was built against a Python that is no longer there:"
        Write-Host "         $staleHome"
        Write-Host '         The venv cannot start, and `uv sync` will reuse it as is. Rebuild it:'
        Write-Host "           Remove-Item -Recurse -Force '$venv'"
        Write-Host '           uv sync --all-extras --dev'
    }
    elseif ($runnable) {
        Write-Host "run.ps1: tinydiffusion is not installed in $runnable"
        Write-Host "         install it with:  uv sync --all-extras --dev"
        Write-Host "                      or:  & '$runnable' -m pip install -e ."
    }
    elseif ($broken) {
        Write-Host "run.ps1: $($broken[0]) exists but will not start."
        Write-Host '         If it is the project venv, rebuild it:'
        Write-Host "           Remove-Item -Recurse -Force '$venv'"
        Write-Host '           uv sync --all-extras --dev'
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
