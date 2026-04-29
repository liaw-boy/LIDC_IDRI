@echo off
setlocal
:: Ensure we are in the script's directory
cd /d "%~dp0"
title Lung Nodule System - Setup

echo ======================================================
echo   Lung Nodule AI Diagnostic System - Setup Wizard
echo ======================================================
echo.

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8+ and add it to PATH.
    pause
    exit /b
)

:: Create virtual environment inside the project folder
if not exist "%~dp0.venv" (
    echo [INFO] Creating virtual environment...
    python -m venv "%~dp0.venv"
)

:: Activate virtual environment and install dependencies
echo [INFO] Installing dependencies from requirements.txt...
call "%~dp0.venv\Scripts\activate"
python -m pip install --upgrade pip
pip install -r "%~dp0requirements.txt"

:: Check for model files
if not exist "%~dp0models\best.pt" (
    echo [WARNING] YOLO model (models\best.pt) not found.
)
if not exist "%~dp0models\dual_input_final_model.pth" (
    echo [WARNING] CNN model (models\dual_input_final_model.pth) not found.
)

:: Create necessary directories
if not exist "%~dp0output" mkdir "%~dp0output"
if not exist "%~dp0data\sample" mkdir "%~dp0data\sample"

echo.
echo ======================================================
echo   Setup Complete! 
echo   To run the app: python -m gui_app.cnn_detector_v1
echo ======================================================
echo.
pause
