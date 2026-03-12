param(
    [switch]$SkipTrain
)

Set-Location (Join-Path $PSScriptRoot "..")

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*$" -or $_ -match "^\s*#") {
            return
        }
        $pair = $_ -split "=", 2
        if ($pair.Length -eq 2) {
            [System.Environment]::SetEnvironmentVariable($pair[0].Trim(), $pair[1].Trim())
        }
    }
}

$configPath = if ($env:CONFIG_PATH) { $env:CONFIG_PATH } else { "config/settings.yaml" }
$apiHost = if ($env:API_HOST) { $env:API_HOST } else { "127.0.0.1" }
$apiPort = if ($env:API_PORT) { [int]$env:API_PORT } else { 8000 }
$uiHost = if ($env:UI_HOST) { $env:UI_HOST } else { "127.0.0.1" }
$uiPort = if ($env:UI_PORT) { [int]$env:UI_PORT } else { 8501 }

$env:PYTHONPATH = "src"
$env:CONFIG_PATH = $configPath
$env:FREE_MODE = if ($env:FREE_MODE) { $env:FREE_MODE } else { "true" }
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

if (-not $SkipTrain) {
    python src/orchestration/pipeline_train.py
    if ($LASTEXITCODE -ne 0) {
        throw "Training pipeline failed."
    }
}

python src/orchestration/pipeline_predict.py
if ($LASTEXITCODE -ne 0) {
    throw "Prediction pipeline failed."
}

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts/start_api.ps1",
    "-ApiHost", $apiHost,
    "-ApiPort", "$apiPort"
)

Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts/start_ui.ps1",
    "-UiHost", $uiHost,
    "-UiPort", "$uiPort"
)

Write-Host "Pipeline done. API at http://$apiHost`:$apiPort and UI at http://$uiHost`:$uiPort"
