@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python -m stockai run >> data\run.log 2>&1
if errorlevel 1 (
  echo Run failed. See data\run.log
  exit /b 1
)
echo Done. Open dashboard.bat to see results.
