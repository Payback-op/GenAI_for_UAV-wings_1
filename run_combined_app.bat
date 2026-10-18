@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "src\combined_app.py"
if errorlevel 1 pause
