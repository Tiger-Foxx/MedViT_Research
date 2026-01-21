#!/bin/bash
# ============================================================================
# MedViT-CAMIL Run Script V2
# ============================================================================
# Usage:
#   ./run.sh test              # Mode test (données synthétiques)
#   ./run.sh proxy             # Mode proxy (NoduleMNIST3D)
#   ./run.sh real              # Mode real (HyperKvasir + vidéos)
#   ./run.sh test --epochs 5   # Avec arguments personnalisés
#   ./run.sh --dry-run         # Vérifier la config sans entraîner
# ============================================================================

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo ""
echo "============================================================"
echo "     MedViT-CAMIL V2: Medical Video Analysis"
echo "     3 Modes: TEST | PROXY | REAL"
echo "============================================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Python
PYTHON_CMD="python"
command -v python3 &> /dev/null && PYTHON_CMD="python3"
log_info "Python: $($PYTHON_CMD --version)"

# Dépendances
log_info "Vérification des dépendances..."
$PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}')" || exit 1
$PYTHON_CMD -c "import timm; print(f'timm: {timm.__version__}')" || exit 1

# GPU
$PYTHON_CMD -c "import torch; print(f'CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Répertoires
mkdir -p results data

# Mode
MODE="${1:-test}"

# Mode REAL: téléchargement HyperKvasir
if [ "$MODE" = "real" ]; then
    DATA_DIR="./data/hyperkvasir"
    if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
        log_warning "Mode REAL: Téléchargement HyperKvasir (~2 Go)..."
        mkdir -p $DATA_DIR
        wget -q --show-progress https://datasets.simula.no/hyper-kvasir/hyper-kvasir-labeled-images.zip -O dataset.zip
        log_info "Extraction..."
        unzip -q dataset.zip -d temp_unzip
        mkdir -p $DATA_DIR/abnormal $DATA_DIR/normal
        mv temp_unzip/*/lower-gi-tract/pathological-findings/*/* $DATA_DIR/abnormal/ 2>/dev/null || true
        mv temp_unzip/*/lower-gi-tract/quality-of-mucosal-views/*/* $DATA_DIR/normal/ 2>/dev/null || true
        rm -rf dataset.zip temp_unzip
        log_success "Données installées!"
    fi
fi

# Lancement
if [[ "$MODE" == "test" ]] || [[ "$MODE" == "proxy" ]] || [[ "$MODE" == "real" ]]; then
    shift || true
fi

echo ""
log_info "Lancement: MODE=${MODE^^}"
echo ""

$PYTHON_CMD -m src.main --mode "$MODE" "$@"

echo ""
log_success "Terminé! Résultats: ./results/"
