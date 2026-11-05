@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install from https://www.python.org/downloads/ and check "Add python.exe to PATH".
  pause
  exit /b 1
)
if not exist .venv python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Install failed.
  pause
  exit /b 1
)
if not exist .env copy .env.example .env >nul
echo.
echo Setup complete. Next: run_daily.bat, then dashboard.bat
pause
