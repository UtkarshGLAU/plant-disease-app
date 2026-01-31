@echo off
echo.
echo ============================================================
echo   PlantGuard AI - Complete Setup Script
echo   Plant Disease Detection System
echo   BTech Final Year Project
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/4] Setting up Backend...
echo ----------------------------------------
cd backend

if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing Python dependencies...
pip install -r requirements.txt

echo Backend setup complete!
echo.

echo [2/4] Setting up Frontend...
echo ----------------------------------------
cd ..\frontend

echo Installing Node.js dependencies...
call npm install

echo Frontend setup complete!
echo.

cd ..

echo [3/4] Checking Model File...
echo ----------------------------------------
if exist "model\plantDisease-resnet34.pth" (
    echo Model file found!
) else (
    echo WARNING: Model file not found at model\plantDisease-resnet34.pth
    echo Please ensure the model file is in the correct location.
)
echo.

echo [4/4] Setup Complete!
echo ============================================================
echo.
echo To start the application:
echo.
echo   1. Start Backend:
echo      cd backend
echo      start-backend.bat
echo.
echo   2. Start Frontend (in a new terminal):
echo      cd frontend
echo      start-frontend.bat
echo.
echo   3. Open browser:
echo      http://localhost:3000
echo.
echo ============================================================
echo.

pause
