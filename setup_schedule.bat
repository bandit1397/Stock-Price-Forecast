@echo off
rem Optional: use this only if you do NOT use the GitHub Actions schedule, or alerts arrive twice.
chcp 65001 >nul
cd /d "%~dp0"
schtasks /Create /F /TN "StockAI Daily KR" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:50 /TR "\"%~dp0run_daily.bat\""
schtasks /Create /F /TN "StockAI Daily US" /SC WEEKLY /D TUE,WED,THU,FRI,SAT /ST 06:30 /TR "\"%~dp0run_daily.bat\""
schtasks /Create /F /TN "StockAI Scan" /SC HOURLY /TR "\"%~dp0run_scan.bat\""
echo Done. Remove with: schtasks /Delete /TN "StockAI Daily KR" /F  (same for the others)
pause
