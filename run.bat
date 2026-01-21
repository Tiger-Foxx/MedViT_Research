@echo off
REM ============================================================================
REM MedViT-CAMIL Run Script V2 (Windows)
REM ============================================================================
REM Usage:
REM   run.bat test              # Mode test (donnees synthetiques)
REM   run.bat proxy             # Mode proxy (NoduleMNIST3D)
REM   run.bat real              # Mode real (HyperKvasir)
REM   run.bat test --epochs 5   # Avec arguments
REM   run.bat --dry-run         # Verification sans entrainement
REM ============================================================================

echo.
echo ============================================================
echo      MedViT-CAMIL V2: Medical Video Analysis
echo      3 Modes: TEST / PROXY / REAL
echo ============================================================
echo.

cd /d "%~dp0"

REM Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python non trouve!
    exit /b 1
)

echo [INFO] Python:
python --version

REM Dependances
python -c "import torch; print(f'[INFO] PyTorch: {torch.__version__}')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PyTorch manquant. pip install -r requirements.txt
    exit /b 1
)

python -c "import timm; print(f'[INFO] timm: {timm.__version__}')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] timm manquant. pip install timm
    exit /b 1
)

REM GPU
python -c "import torch; print(f'[INFO] CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

REM Repertoires
if not exist "results" mkdir results
if not exist "data" mkdir data

REM Mode
set MODE=%1
if "%MODE%"=="" set MODE=test

REM Arguments
set ARGS=
:loop
shift
if "%1"=="" goto endloop
set ARGS=%ARGS% %1
goto loop
:endloop

echo.
echo [INFO] Lancement: MODE=%MODE%
echo.

REM Lancer
python -m src.main --mode %MODE% %ARGS%

echo.
echo [SUCCESS] Termine! Resultats: .\results\
echo.
