@echo off
REM ============================================================================
REM MedViT-CAMIL Run Script (Windows)
REM ============================================================================
REM Script universel pour lancer l'entraînement
REM
REM Usage:
REM   run.bat test              # Mode test (données synthétiques, rapide)
REM   run.bat real              # Mode real (NoduleMNIST3D)
REM   run.bat test --epochs 5   # Mode test avec 5 époques
REM   run.bat --dry-run         # Vérifier la config sans entraîner
REM
REM ============================================================================

echo.
echo ============================================================
echo      MedViT-CAMIL: Medical Video Analysis
echo      Context-Aware Multiple Instance Learning
echo ============================================================
echo.

REM Changer vers le répertoire du script
cd /d "%~dp0"

REM Vérifier Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python n'est pas installé ou pas dans le PATH!
    exit /b 1
)

echo [INFO] Python version:
python --version

REM Vérifier PyTorch
python -c "import torch; print(f'[INFO] PyTorch: {torch.__version__}')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PyTorch n'est pas installé. Executez: pip install -r requirements.txt
    exit /b 1
)

REM Vérifier timm
python -c "import timm; print(f'[INFO] timm: {timm.__version__}')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] timm n'est pas installe. Executez: pip install timm
    exit /b 1
)

REM Vérifier GPU
python -c "import torch; print(f'[INFO] CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Non disponible (CPU)\"}')"

REM Créer les répertoires
if not exist "results" mkdir results
if not exist "data" mkdir data

REM Mode par défaut
set MODE=%1
if "%MODE%"=="" set MODE=test

REM Préparer les arguments restants
set ARGS=
:loop
shift
if "%1"=="" goto endloop
set ARGS=%ARGS% %1
goto loop
:endloop

echo.
echo [INFO] Lancement en mode: %MODE%
echo.

REM Lancer l'entraînement
python src/main.py --mode %MODE% %ARGS%

echo.
echo [SUCCESS] Execution terminee!
echo [INFO] Resultats dans: .\results\
echo.
