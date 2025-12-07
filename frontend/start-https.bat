@echo off
echo Setting up HTTPS for React development...
echo.

REM Stop any running React server
taskkill /F /IM node.exe 2>nul

echo Starting React development server with HTTPS...
echo.
echo IMPORTANT: 
echo 1. Your browser will show a security warning because of the self-signed certificate
echo 2. Click "Advanced" and then "Proceed to localhost (unsafe)" 
echo 3. This is safe for development purposes
echo.
echo The server will be available at:
echo https://localhost:3000 (for local access)
echo https://[your-ip]:3000 (for mobile access)
echo.

cd /d "%~dp0"
set HTTPS=true
npm start

pause