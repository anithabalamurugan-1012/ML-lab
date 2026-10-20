@echo off
title Retinal Disease Screening Lab
cd /d "%~dp0"

echo ========================================================
echo   Retinal Disease Screening Lab - Web App Launcher
echo ========================================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo.
    echo Please install Python (3.10 or 3.11 recommended):
    echo 1. Download from: https://www.python.org/downloads/
    echo 2. During installation, CHECK "Add python.exe to PATH"
    echo.
    pause
    exit /b 1
)

echo [1/2] Checking and installing dependencies...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Encountered an issue installing requirements. Trying to proceed...
)

echo.
echo [2/2] Starting Streamlit server...
echo App URL: http://localhost:8501
echo.
python -m streamlit run app.py --server.port=8501 --browser.gatherUsageStats=false

pause
