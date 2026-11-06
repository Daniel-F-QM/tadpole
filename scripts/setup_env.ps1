# Tadpole: one-shot development setup (Windows PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
# 1. .venv with Python 3.13 and the pinned requirements (+ dev/test tools)
# 2. the Julia package's own environment (julia/TadpoleRetrieval), precompiled
# 3. juliacall's environment (julia/env: PythonCall + a dev of the package)
#
# Needs: Python 3.13 from python.org (the `py` launcher) and Julia >= 1.10 on
# PATH (juliaup is fine). Re-running is safe; finished steps are quick.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv (Python 3.13)..."
    py -3.13 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "could not create .venv" }
}
$py = Join-Path $root ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt
& $py -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

if (-not (Get-Command julia -ErrorAction SilentlyContinue)) {
    throw "julia not found on PATH. Install it (winget install julia -s msstore) or set TADPOLE_JULIA."
}
Write-Host "Instantiating julia/TadpoleRetrieval..."
julia --project=julia/TadpoleRetrieval -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
if ($LASTEXITCODE -ne 0) { throw "Julia package setup failed" }

Write-Host "Setting up juliacall's environment (first time: a few minutes)..."
& $py scripts\setup_julia.py
if ($LASTEXITCODE -ne 0) { throw "juliacall setup failed" }

Write-Host ""
Write-Host "Done. Run the app with:  .venv\Scripts\python main.py"
Write-Host "Tests:  .venv\Scripts\python -m pytest   and   julia --project=julia/TadpoleRetrieval -e 'using Pkg; Pkg.test()'"
