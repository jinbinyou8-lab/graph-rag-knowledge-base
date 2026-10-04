@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONPATH=%~dp0"
set "HTTP_PROXY="
set "HTTPS_PROXY="
set "http_proxy="
set "https_proxy="
set "ALL_PROXY="
title graph-rag-knowledge-base  QUERY service  :8001
echo ============================================================
echo  Query service  ^-^>  http://127.0.0.1:8001
echo  Chat page      ^-^>  http://127.0.0.1:8001/
echo  API docs       ^-^>  http://127.0.0.1:8001/docs
echo  (first start needs about 60s; wait for 'Application startup complete')
echo ============================================================
echo.
"%~dp0knowledge\.venv\Scripts\python.exe" -m uvicorn knowledge.processor.query_process.api.query_router:create_app --factory --host 127.0.0.1 --port 8001
echo.
echo service stopped. press any key to close.
pause >nul
