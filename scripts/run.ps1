$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

if ($env:PYTHON) {
    $candidates = @($env:PYTHON)
}
else {
    $candidates = @((Join-Path $root '.venv\Scripts\python.exe'))
    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { $candidates += $onPath.Source }
}

$probe = "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('tinydiffusion') else 1)"
$interpreter = $null
$runnable = $null
$broken = @()

$strict = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
foreach ($candidate in $candidates) {
    if (-not (Test-Path $candidate)) { continue }
    & $candidate -c 'pass' 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { $broken += $candidate; continue }
    & $candidate -c $probe 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { $interpreter = $candidate; break }
    if (-not $runnable) { $runnable = $candidate }
}
$ErrorActionPreference = $strict

function Get-StaleVenvHome {
    param([string]$VenvRoot)

    $cfg = Join-Path $VenvRoot 'pyvenv.cfg'
    if (-not (Test-Path $cfg)) { return $null }
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

$ErrorActionPreference = 'Continue'

& $interpreter -m tinydiffusion.cli @forwarded
exit $LASTEXITCODE
