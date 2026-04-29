@echo off
setlocal
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

:: Create virtual environment
if not exist ".venv" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
)

:: Activate virtual environment and install dependencies
echo [INFO] Installing dependencies from requirements.txt...
call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

:: Check for model files
if not exist "models\best.pt" (
    echo [WARNING] YOLO model (models\best.pt) not found.
)
if not exist "models\dual_input_final_model.pth" (
    echo [WARNING] CNN model (models\dual_input_final_model.pth) not found.
)

:: Create necessary directories
if not exist "output" mkdir output
if not exist "data\sample" mkdir data\sample

echo.
echo ======================================================
echo   Setup Complete! 
echo   To run the app: python -m gui_app.cnn_detector_v1
echo ======================================================
echo.
pause
