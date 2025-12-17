@echo off
REM Backend startup script for Windows

echo Setting up backend environment...

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run the backend server
echo Starting backend server...
uvicorn app.main:app --reload --port 8000