@echo off
chcp 65001 >nul
REM ============================================
REM  Beauty RAG Engine - Startup Script (Windows)
REM ============================================

echo.
echo  [Beauty RAG] Starting up...
echo ========================================
echo.

REM 1. Check .env
if not exist .env (
    echo  [ERROR] .env file not found! Copy .env.example to .env first.
    pause
    exit /b
)
echo  [OK] .env config found

REM 2. Start Milvus via Docker
docker ps --format "{{.Names}}" 2>nul | findstr milvus >nul
if %ERRORLEVEL% neq 0 (
    echo  [..] Starting Milvus database...
    docker compose up -d
    echo  [..] Waiting 15s for Milvus...
    timeout /t 15 /nobreak >nul
) else (
    echo  [OK] Milvus is running
)

set PORT=8877
REM 3. Start Backend (port %PORT%)
echo.
echo  [..] Starting backend (port %PORT%)...
start "Beauty-Backend" cmd /c "uvicorn main:app --host 0.0.0.0 --port %PORT% --reload"
echo  [..] Waiting 10s for backend...
timeout /t 10 /nobreak >nul

REM 4. Start Frontend (port 8501)
echo.
echo  [..] Starting frontend (port 8501)...
start "Beauty-Frontend" cmd /c "streamlit run app_frontend.py"

echo.
echo  ========================================
echo  [OK] All services started!
echo  Frontend: http://localhost:8501
echo  Backend:  http://localhost:%PORT%
echo  ========================================
echo.
pause
