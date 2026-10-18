@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "src\download_data.py"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" "src\prepare_data.py"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" "src\train.py"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" "src\generate_combined.py"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" "src\verify.py"
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" "src\write_report.py"
if errorlevel 1 exit /b 1
