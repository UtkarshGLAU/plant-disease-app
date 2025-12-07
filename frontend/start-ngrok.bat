@echo off
echo Setting up secure HTTPS tunnel for mobile testing...
echo.

echo Checking if ngrok is installed...
where ngrok >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ngrok is not installed. Please install it first:
    echo.
    echo 1. Go to https://ngrok.com/download
    echo 2. Download ngrok for Windows
    echo 3. Extract it to a folder in your PATH
    echo 4. Sign up for a free account at https://ngrok.com
    echo 5. Get your auth token and run: ngrok config add-authtoken YOUR_TOKEN
    echo.
    pause
    exit /b 1
)

echo Starting React development server on HTTP...
start /B npm start

echo Waiting for React server to start...
timeout /t 10 /nobreak >nul

echo Starting ngrok tunnel...
echo.
echo This will create a secure HTTPS URL that you can use on mobile
echo The tunnel will remain active until you close this window
echo.

ngrok http 3000