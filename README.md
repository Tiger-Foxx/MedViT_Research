# MedViT-CAMIL V2

**Context-Aware Multiple Instance Learning for Medical Video Analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Problème de Recherche

Détection d'**anomalies rares** dans des séquences médicales sur appareils à ressources limitées.

**Défi** : L'anomalie n'apparaît que sur quelques frames (5/200 → dilution ×40 avec average pooling).

## 🧬 Architecture CAMIL

```
Vidéo (T, 3, H, W)
       │
       ▼
┌─────────────────┐
│  MobileViT (❄️)  │  ← Backbone gelé
└────────┬────────┘
         │ Features (T, D)
         ▼
┌─────────────────┐
│  Conv1D (k=3)   │  ← Contexte local [t-1, t, t+1]
└────────┬────────┘
         ▼
┌─────────────────┐
│ Gated Attention │  ← V: Tanh | U: Sigmoid
│      MIL        │
└────────┬────────┘
         ▼
┌─────────────────┐
│   Classifier    │  → Prédiction
└─────────────────┘
```

## 📦 3 Modes d'Exécution

| Mode | Données | Usage | Epochs |
|------|---------|-------|--------|
| **TEST** | Synthétiques (speckle + lésions) | Validation locale rapide | 3 |
| **PROXY** | NoduleMNIST3D (CT pulmonaire) | Preuve scientifique | 15 |
| **REAL** | HyperKvasir + vidéos OpenCV | Serveur production | 50 |

## 🚀 Installation

```bash
cd MedViT_Research
pip install -r requirements.txt
```

## 💻 Utilisation

### Mode TEST (Laptop - Rapide)
```bash
# Windows
run.bat test

# Linux/Mac
./run.sh test

# Python direct
python -m src.main --mode test
```

### Mode PROXY (NoduleMNIST3D)
```bash
run.bat proxy
# ou
python -m src.main --mode proxy --epochs 15
```

### Mode REAL (Serveur avec GPU)
```bash
./run.sh real
# Télécharge automatiquement HyperKvasir (~2 Go)
```

### Options
```bash
python -m src.main --mode test --epochs 5 --batch_size 8 --lr 0.0001
python -m src.main --mode test --dry-run  # Vérification sans entraînement
```

## 🐳 Docker

```bash
# Build
docker build -t medvit-camil .

# Run
docker run --gpus all -v $(pwd)/results:/app/results medvit-camil test
docker run --gpus all -v $(pwd)/data:/app/data medvit-camil real
```

## 📊 Résultats Attendus

| Modèle | Mode TEST | Mode PROXY |
|--------|-----------|------------|
| Baseline (Avg) | ~50-60% | ~60-70% |
| **CAMIL** | ~70-85% | ~80-90% |

### Visualisations générées
- `training_curves_*.png` : Loss/Accuracy comparatives
- `attention_comparison_*.png` : Heatmaps Baseline vs CAMIL
- `results_*.json` : Métriques détaillées

## 📁 Structure

```
MedViT_Research/
├── Dockerfile
├── requirements.txt
├── run.sh / run.bat
├── src/
│   ├── config.py      # 3 modes: test/proxy/real
│   ├── dataset.py     # Synthétique + MedMNIST + OpenCV
│   ├── model.py       # Baseline + CAMIL
│   └── main.py        # Training loop
├── data/              # Auto-téléchargé
└── results/           # Graphiques + JSON
```

## 📚 Références

- **MobileViT**: Mehta & Rastegari, ICLR 2022
- **Gated Attention MIL**: Ilse et al., ICML 2018
- **MedMNIST**: Yang et al., Nature Scientific Data 2023
- **HyperKvasir**: Borgli et al., Scientific Data 2020

## 👥 Auteurs

Projet M2 Recherche - ENSPY (École Nationale Supérieure Polytechnique de Yaoundé)

*"Design of Next-Generation Generative and Agentic AI Architectures"*
