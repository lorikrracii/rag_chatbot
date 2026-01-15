@echo off
setlocal

REM Run the Streamlit chat UI
cd /d "%~dp0\.."

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

echo [INFO] Starting Streamlit app...
streamlit run app/streamlit_app.py

pause
