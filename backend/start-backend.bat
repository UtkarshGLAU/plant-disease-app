@echo off
echo.
echo ========================================
echo   PlantGuard AI - Backend Server
echo   Plant Disease Detection API v2.0
echo ========================================
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created!
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo Starting Flask server...
echo.
echo ========================================
echo   API will be available at:
echo   http://localhost:5000
echo ========================================
echo.

python app.py

pause
