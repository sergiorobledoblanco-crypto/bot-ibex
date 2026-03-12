Set-Location (Join-Path $PSScriptRoot "..")

$patterns = @(
    "uvicorn app.api:app",
    "streamlit run src/app/ui_streamlit.py"
)

Get-CimInstance Win32_Process | ForEach-Object {
    $cmd = $_.CommandLine
    if ($null -eq $cmd) {
        return
    }
    foreach ($pattern in $patterns) {
        if ($cmd -like "*$pattern*") {
            try {
                Stop-Process -Id $_.ProcessId -Force
            } catch {
                Write-Host "Could not stop PID $($_.ProcessId): $($_.Exception.Message)"
            }
        }
    }
}

Write-Host "API/UI processes stopped (if running)."
