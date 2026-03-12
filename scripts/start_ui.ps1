param(
    [string]$UiHost = "127.0.0.1",
    [int]$UiPort = 8501
)

Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = "src"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

# Avoid first-run interactive prompt in unattended runs.
"" | python -m streamlit run src/app/ui_streamlit.py --server.address $UiHost --server.port $UiPort
