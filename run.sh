#!/bin/bash
# ============================================================================
# MedViT-CAMIL Run Script
# ============================================================================
# Script universel pour lancer l'entraînement
#
# Usage:
#   ./run.sh test              # Mode test (données synthétiques, rapide)
#   ./run.sh real              # Mode real (NoduleMNIST3D)
#   ./run.sh test --epochs 5   # Mode test avec 5 époques
#   ./run.sh --dry-run         # Vérifier la config sans entraîner
#
# ============================================================================

set -e  # Exit on error

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction de logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo ""
echo "============================================================"
echo "     MedViT-CAMIL: Medical Video Analysis"
echo "     Context-Aware Multiple Instance Learning"
echo "============================================================"
echo ""

# Déterminer le répertoire du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Vérifier Python
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        log_error "Python n'est pas installé!"
        exit 1
    fi
else
    PYTHON_CMD="python"
fi

log_info "Python: $($PYTHON_CMD --version)"

# Vérifier les dépendances critiques
log_info "Vérification des dépendances..."

$PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || {
    log_error "PyTorch n'est pas installé. Exécutez: pip install -r requirements.txt"
    exit 1
}

$PYTHON_CMD -c "import timm; print(f'timm: {timm.__version__}')" 2>/dev/null || {
    log_error "timm n'est pas installé. Exécutez: pip install timm"
    exit 1
}

# Vérifier GPU
$PYTHON_CMD -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA: Non disponible (utilisation CPU)')
"

# Créer les répertoires nécessaires
mkdir -p results
mkdir -p data

# Mode par défaut
MODE="${1:-test}"

# Shift pour passer les arguments restants
if [[ "$MODE" == "test" ]] || [[ "$MODE" == "real" ]]; then
    shift || true
fi

# Lancer l'entraînement
echo ""
log_info "Lancement en mode: ${MODE^^}"
echo ""

$PYTHON_CMD src/main.py --mode "$MODE" "$@"

echo ""
log_success "Exécution terminée!"
log_info "Résultats dans: ./results/"
echo ""
