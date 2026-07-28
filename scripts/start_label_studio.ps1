$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$localFilesRoot = (Resolve-Path (Join-Path $repoRoot "data\raw")).Path
$labelStudio = Join-Path $repoRoot ".venv-label-studio\Scripts\label-studio.exe"

if (-not (Test-Path -LiteralPath $labelStudio)) {
    throw "Label Studio executable not found: $labelStudio"
}

$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = $localFilesRoot

Write-Host "Local files root: $localFilesRoot"
Write-Host "Starting Label Studio at http://localhost:8080"

Set-Location $repoRoot
& $labelStudio
