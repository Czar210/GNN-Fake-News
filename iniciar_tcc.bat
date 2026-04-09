@echo off
title Truth GNN Analytics - Launcher
color 0B

echo ==================================================
echo       INICIANDO O TRUTH GNN ANALYTICS (TCC)
echo ==================================================
echo.

:: Resolve o diretorio raiz a partir de onde o .bat esta
set "ROOT=%~dp0"

:: ── Checagem de Python ──────────────────────────────────────────────────────
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERRO] Python nao encontrado no PATH. Instale Python 3.11+ e tente novamente.
    pause
    exit /b 1
)

:: ── Checagem de Node.js ─────────────────────────────────────────────────────
where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERRO] Node.js nao encontrado no PATH. Instale Node 20+ e tente novamente.
    pause
    exit /b 1
)

:: ── Venv: cria se nao existir ───────────────────────────────────────────────
if not exist "%ROOT%.venv\Scripts\activate.bat" (
    echo [SETUP] Criando ambiente virtual (.venv)...
    python -m venv "%ROOT%.venv"
    echo [SETUP] Instalando dependencias Python...
    "%ROOT%.venv\Scripts\pip.exe" install -r "%ROOT%requirements.txt"
)

:: ── Node modules: instala se nao existir ────────────────────────────────────
if not exist "%ROOT%Interface\frontend\web\node_modules" (
    echo [SETUP] Instalando dependencias do frontend (npm install)...
    cd /d "%ROOT%Interface\frontend\web"
    npm install
    cd /d "%ROOT%"
)

echo.
echo [1/2] Ligando o Motor Neural do Backend (FastAPI, PyTorch)...
start "Backend - FastAPI" cmd /k "cd /d "%ROOT%Interface\frontend\api" && "%ROOT%.venv\Scripts\python.exe" -m uvicorn main:app --reload"

timeout /t 3 /nobreak >nul

echo [2/2] Construindo Dashboard Premium (Next.js)...
start "Frontend - Next.js" cmd /k "cd /d "%ROOT%Interface\frontend\web" && npm run dev"

echo.
echo ==================================================
echo   TUDO PRONTO! O sistema foi engatado com sucesso.
echo.
echo   - Sua API esta rodando na porta 8000.
echo   - Seu Dashboard pode ser acessado em:
echo     http://localhost:3000
echo.
echo   Para desligar, feche as duas janelas que abriram.
echo ==================================================
pause
