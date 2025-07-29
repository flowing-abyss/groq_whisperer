@echo off
echo Starting Groq Whisperer in the background...
cd /d %~dp0
echo Activating virtual environment...
call .venv\Scripts\activate
echo Starting script...
start "GroqWhisperer" /B python main.py
echo Script is running in the background.
echo To stop it, close this window or use Task Manager to find the python.exe process.
exit