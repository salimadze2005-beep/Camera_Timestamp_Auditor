param(
    [Parameter(Mandatory=$true)][string]$ImagesDir,
    [Parameter(Mandatory=$true)][string]$Manifest,
    [string]$OutDir = "results",
    [switch]$Gpu,
    [switch]$SkipEnhanced
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) { py -3.12 -m venv .venv }
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
$modelArgs = @()
if ($Gpu) { $modelArgs += "--gpu" }
& ".venv\Scripts\python.exe" download_models.py @modelArgs
$argsList = @("run_pipeline.py", "--images-dir", $ImagesDir, "--manifest", $Manifest, "--out-dir", $OutDir)
if ($Gpu) { $argsList += "--gpu" }
if ($SkipEnhanced) { $argsList += "--skip-enhanced" }
& ".venv\Scripts\python.exe" @argsList
