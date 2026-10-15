@echo off
title OpsMate - Your AI Operations Companion
cd /d "%~dp0"
echo ==================================================
echo   OpsMate - Your AI Operations Companion
echo ==================================================
echo.
echo Step 1 of 2: Checking the required libraries (first time takes 1-2 minutes)...
python -m pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
  echo.
  echo Could not install the libraries. Check your internet connection and that Python is installed.
  pause
  exit /b
)
echo.
echo Step 2 of 2: Starting the app. Your browser will open at http://localhost:8501
echo.
echo   KEEP THIS WINDOW OPEN while you use the app.
echo   To stop the app, close this window or press Ctrl + C.
echo.
python -m streamlit run app.py --browser.gatherUsageStats false
pause
