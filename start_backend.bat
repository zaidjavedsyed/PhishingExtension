@echo off
REM Startup script for Phishing Detection API (Windows)

echo 🚀 Starting Phishing Detection API...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python first.
    pause
    exit /b 1
)

REM Check if model file exists
if not exist "XGBoostClassifier.pickle.dat" (
    echo ❌ XGBoostClassifier.pickle.dat not found!
    echo Please make sure the model file is in the same directory as this script.
    pause
    exit /b 1
)

REM Install requirements if needed
echo 📦 Installing Python dependencies...
pip install -r requirements.txt

REM Start the FastAPI server
echo 🌐 Starting FastAPI server on http://localhost:8000
echo 📊 API Documentation available at http://localhost:8000/docs
echo 🔍 Health check available at http://localhost:8000/health
echo.
echo Press Ctrl+C to stop the server
echo.

python backend.py
pause

