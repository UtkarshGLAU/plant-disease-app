@echo off
echo Starting Plant Disease Detection System...
echo.

echo Creating Python virtual environment for backend...
cd backend
python -m venv venv
call venv\Scripts\activate.bat

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Backend setup complete!
echo.
echo To start the system:
echo 1. In this window, run: python app.py
echo 2. In a new terminal, navigate to frontend folder and run: npm install
echo 3. Then run: npm start
echo.
echo The backend will run on http://localhost:5000
echo The frontend will run on http://localhost:3000
echo.
pause