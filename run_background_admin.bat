@echo off
echo Starting Groq Whisperer with administrator privileges...
cd /d %~dp0
echo Activating virtual environment...
call .venv\Scripts\activate
echo Starting script in background...
powershell -Command "Start-Process python -ArgumentList 'main.py' -Verb RunAs -WindowStyle Hidden -WorkingDirectory '%CD%'"
echo Script started successfully!
echo You can close this window - the script will continue running in background.
echo To stop the script, use Task Manager and find python.exe process.
pause
exit 