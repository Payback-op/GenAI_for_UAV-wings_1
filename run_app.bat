@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "src\design_app.py"
if errorlevel 1 pause
