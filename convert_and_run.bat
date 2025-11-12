@echo off
REM Model Conversion Script for Windows

echo 🔧 XGBoost Model Conversion Tool
echo ================================

echo.
echo 📋 Step 1: Converting your XGBoost model...
python convert_model.py

if %errorlevel% equ 0 (
    echo.
    echo ✅ Model conversion completed successfully!
    echo.
    echo 📋 Step 2: Starting the backend server...
    python backend.py
) else (
    echo.
    echo ❌ Model conversion failed!
    echo.
    echo 💡 Suggestions:
    echo    1. Make sure XGBoostClassifier.pickle.dat exists
    echo    2. Try updating XGBoost: pip install --upgrade xgboost
    echo    3. The backend will create a fallback model
    echo.
    echo 📋 Starting backend with fallback model...
    python backend.py
)

pause
