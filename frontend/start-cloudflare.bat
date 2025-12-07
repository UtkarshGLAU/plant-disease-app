@echo off
echo Setting up Cloudflare tunnel for secure HTTPS access...
echo.

echo Checking if cloudflared is installed...
where cloudflared >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo cloudflared is not installed. Installing via winget...
    winget install --id Cloudflare.cloudflared
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install cloudflared. Please install manually:
        echo Go to: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/
        pause
        exit /b 1
    )
)

echo Starting React development server on HTTP...
start /B npm start

echo Waiting for React server to start...
timeout /t 10 /nobreak >nul

echo Starting Cloudflare tunnel...
echo.
echo This will create a secure HTTPS URL that you can use on mobile
echo The tunnel will remain active until you close this window
echo.

cloudflared tunnel --url http://localhost:3000