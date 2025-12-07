@echo off
echo Starting Plant Disease Prediction Backend...
echo.

cd /d "%~dp0"

echo Activating virtual environment...
call venv\Scripts\activate.bat

if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

echo Starting Flask server...
echo.
echo Backend will be available at:
echo http://localhost:5000 (local access)
echo http://172.16.144.249:5000 (mobile access)
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause