@echo off
setlocal
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

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Python not found in PATH.
    pause
    exit /b 1
)

where docker >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] Docker not found in PATH.
    pause
    exit /b 1
)

REM 2. Start Milvus via Docker
docker ps --format "{{.Names}}" 2>nul | findstr milvus >nul
if %ERRORLEVEL% neq 0 (
    echo  [..] Starting Milvus database...
    docker compose up -d
    if %ERRORLEVEL% neq 0 (
        echo  [ERROR] Failed to start Milvus.
        pause
        exit /b 1
    )
    echo  [..] Waiting 15s for Milvus...
    timeout /t 15 /nobreak >nul
) else (
    echo  [OK] Milvus is running
)

set "PORT=8877"
REM 3. Start Backend (port %PORT%)
echo.
echo  [..] Starting backend (port %PORT%)...
start "Beauty-Backend" cmd /c "python -m uvicorn main:app --host 127.0.0.1 --port %PORT%"
echo  [..] Waiting for backend health check...
for /l %%N in (1,1,60) do (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $health = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:%PORT%/health; $privacy = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:%PORT%/api/v1/privacy; if ($health.StatusCode -eq 200 -and $privacy.StatusCode -eq 200) { exit 0 } } catch {} ; exit 1" >nul 2>nul
    if not errorlevel 1 goto backend_ready
    timeout /t 1 /nobreak >nul
)
echo  [ERROR] Backend did not become ready within 60 seconds.
echo  Please close the old backend process on port %PORT% and run start.bat again.
pause
exit /b 1

:backend_ready
echo  [OK] Backend health and privacy routes are ready

REM 4. Start Frontend (port 8501)
echo.
echo  [..] Starting frontend (port 8501)...
start "Beauty-Frontend" cmd /c "python -m streamlit run app_frontend.py --server.maxUploadSize 50"

echo.
echo  ========================================
echo  [OK] All services started!
echo  Frontend: http://localhost:8501
echo  Backend:  http://localhost:%PORT%
echo  ========================================
echo.
pause
