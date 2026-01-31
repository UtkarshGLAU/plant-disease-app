@echo off
echo.
echo ========================================
echo   PlantGuard AI - Frontend Server
echo   React Development Server
echo ========================================
echo.

cd /d "%~dp0"

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing dependencies...
    npm install
    echo Dependencies installed!
    echo.
)

echo.
echo Starting React development server...
echo.
echo ========================================
echo   App will be available at:
echo   http://localhost:3000
echo ========================================
echo.

npm start

pause
