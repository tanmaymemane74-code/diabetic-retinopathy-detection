@echo off
cd /d "%~dp0"
title Diabetic Retinopathy CDSS Launcher
echo =================================================================
echo   Diabetic Retinopathy Clinical Decision Support System (DR-CDSS)
echo =================================================================
echo Project Directory: %CD%
echo.

set PY_EXE=C:\Users\tanma\AppData\Local\Programs\Python\Python312\python.exe
if not exist "%PY_EXE%" (
    set PY_EXE=python
)

echo Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "DR-CDSS FastAPI Backend" /d "%~dp0" cmd /k ""%PY_EXE%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak > nul

echo Starting Streamlit Frontend on http://localhost:8501 ...
start "DR-CDSS Streamlit Dashboard" /d "%~dp0" cmd /k ""%PY_EXE%" -m streamlit run frontend/app.py --server.port 8501"

timeout /t 2 /nobreak > nul
start http://localhost:8501

echo.
echo =================================================================
echo   Both services are now running!
echo   - Streamlit Dashboard : http://localhost:8501
echo   - FastAPI API Docs    : http://127.0.0.1:8000/docs
echo =================================================================
pause
