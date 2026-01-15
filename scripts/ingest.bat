@echo off
setlocal

REM Run ingestion (PDF -> chunks -> embeddings -> Chroma)
cd /d "%~dp0\.."

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

echo [INFO] Running ingestion...
python -m rag.ingest

echo.
echo [DONE] Ingestion completed successfully.
pause
