@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONPATH=%~dp0"
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "http_proxy="
set "https_proxy="
set "ALL_PROXY="
title shopkeeper-brain  IMPORT service  :8000
echo ============================================================
echo  Import service ^-^>  http://127.0.0.1:8000
echo  Import page    ^-^>  http://127.0.0.1:8000/import
echo  API docs       ^-^>  http://127.0.0.1:8000/docs
echo  (first start needs about 60s; wait for 'Application startup complete')
echo ============================================================
echo.
"%~dp0knowledge\.venv\Scripts\python.exe" -m uvicorn knowledge.processor.import_process.api.import_router:create_app --factory --host 127.0.0.1 --port 8000
echo.
echo service stopped. press any key to close.
pause >nul
