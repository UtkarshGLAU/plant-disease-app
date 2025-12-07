@echo off
echo Starting HTTP server for mobile testing...
echo.
echo IMPORTANT: This works with most mobile browsers EXCEPT Firefox Mobile
echo Firefox Mobile requires HTTPS for camera access
echo.
echo Your server will be available at:
echo http://localhost:3000 (local)
echo http://172.16.144.249:3000 (mobile - use Chrome/Edge/Safari)
echo.

cd /d "%~dp0"
npm start

pause