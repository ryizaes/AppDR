$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $projectRoot 'backend'
$python = Join-Path $backendDir '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Backend Python environment was not found at $python"
}

$env:MODEL_MODE = 'dual_model_screening_hybrid_severity'
$routedInterfaces = Get-NetRoute -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue |
    Where-Object { $_.NextHop -ne '0.0.0.0' } |
    Sort-Object RouteMetric |
    Select-Object -ExpandProperty InterfaceIndex -Unique
$phoneAddresses = foreach ($interfaceIndex in $routedInterfaces) {
    Get-NetIPAddress -InterfaceIndex $interfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
            $_.IPAddress -notlike '127.*' -and
            $_.IPAddress -notlike '169.254.*'
        } |
        Select-Object -ExpandProperty IPAddress
}

Write-Host 'Starting OPTIMEYE backend in dual-model mode.' -ForegroundColor Cyan
Write-Host 'Laptop: http://127.0.0.1:8000/health'
foreach ($address in $phoneAddresses | Select-Object -Unique) {
    Write-Host "Phone:  http://${address}:8000/health"
}
Write-Host 'Keep this window open while using the app.' -ForegroundColor Yellow

Push-Location $backendDir
try {
    & $python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
}
finally {
    Pop-Location
}
