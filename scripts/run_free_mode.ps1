param(
    [switch]$SkipTrain
)

Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "Created .env from .env.example"
    }
}

$env:FREE_MODE = "true"
$env:API_HOST = "127.0.0.1"
$env:UI_HOST = "127.0.0.1"
$env:API_PORT = if ($env:API_PORT) { $env:API_PORT } else { "8000" }
$env:UI_PORT = if ($env:UI_PORT) { $env:UI_PORT } else { "8501" }
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

if ($SkipTrain) {
    powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1 -SkipTrain
} else {
    powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
}
