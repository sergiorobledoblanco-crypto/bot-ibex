param(
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8000
)

Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = "src"

python -m uvicorn app.api:app --host $ApiHost --port $ApiPort
