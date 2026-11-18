# Diabetic Retinopathy CDSS Launcher (PowerShell)
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  Diabetic Retinopathy Clinical Decision Support System (DR-CDSS)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

$py = "C:\Users\tanma\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Starting FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectRoot'; & '$py' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

Start-Sleep -Seconds 3

Write-Host "Starting Streamlit Frontend on http://localhost:8501 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectRoot'; & '$py' -m streamlit run frontend/app.py --server.port 8501"

Start-Sleep -Seconds 2
Start-Process "http://localhost:8501"

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Yellow
Write-Host "  Services launched successfully!" -ForegroundColor Yellow
Write-Host "  - Streamlit Dashboard : http://localhost:8501" -ForegroundColor Yellow
Write-Host "  - FastAPI Swagger Docs: http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Yellow
