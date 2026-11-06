@echo off
REM Double-click this file (or run it in a terminal) after all members recorded data.
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
python run_all.py %*
pause
