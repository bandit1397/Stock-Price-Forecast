@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python -m stockai scan >> data\run.log 2>&1
