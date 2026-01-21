# MedViT-CAMIL

**Context-Aware Multiple Instance Learning for Medical Video Analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Problème de Recherche

Détection d'**anomalies rares** dans des séquences médicales (vidéos/volumes 3D) sur des appareils à ressources limitées.

**Défi principal** : L'anomalie n'apparaît que sur quelques frames (ex: 5/200), ce qui pose le problème de "l'aiguille dans la botte de foin". Les méthodes classiques (average pooling) diluent le signal par un facteur de ~40x.

## 🧬 Architecture Proposée

```
┌─────────────────────────────────────────────────────────────┐
│                    MedViT-CAMIL                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Vidéo (T, 3, H, W)                                        │
│         │                                                   │
│         ▼                                                   │
│   ┌─────────────────┐                                       │
│   │  MobileViT (❄️)  │  ← Backbone gelé, pré-entraîné       │
│   │  Feature Extractor│                                     │
│   └────────┬────────┘                                       │
│            │ Features (T, D)                                │
│            ▼                                                │
│   ┌─────────────────┐                                       │
│   │   Conv1D (k=3)  │  ← Contexte local [t-1, t, t+1]      │
│   │   Temporal      │                                       │
│   └────────┬────────┘                                       │
│            │                                                │
│            ▼                                                │
│   ┌─────────────────┐                                       │
│   │ Gated Attention │  ← V: Tanh (contenu)                 │
│   │      MIL        │  ← U: Sigmoid (gate)                 │
│   └────────┬────────┘                                       │
│            │ Aggregated (D,)                                │
│            ▼                                                │
│   ┌─────────────────┐                                       │
│   │   Classifier    │  → Logits (num_classes)              │
│   └─────────────────┘                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Contribution Scientifique

Le module **CAMIL** (Context-Aware Multiple Instance Learning) combine :

1. **Conv1D temporelle** : Vérifie la cohérence locale sur 3 frames consécutives
2. **Gated Attention** (Ilse et al., ICML 2018) : Le gate sigmoid peut "fermer" l'attention sur les frames bruitées, contrairement au softmax standard qui distribue toujours de l'attention

## 📁 Structure du Projet

```
MedViT_Research/
├── Dockerfile              # Image Docker pour déploiement
├── requirements.txt        # Dépendances Python
├── run.sh                  # Script de lancement (Linux/Mac)
├── run.bat                 # Script de lancement (Windows)
├── README.md               # Ce fichier
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration (modes TEST/REAL)
│   ├── dataset.py          # Datasets synthétiques + MedMNIST
│   ├── model.py            # Architectures Baseline et CAMIL
│   └── main.py             # Entraînement et évaluation
├── data/                   # Données (auto-téléchargées)
└── results/                # Résultats et visualisations
```

## 🚀 Installation

### Prérequis
- Python 3.8+
- PyTorch 2.0+
- CUDA (optionnel, pour GPU)

### Installation des dépendances

```bash
cd MedViT_Research
pip install -r requirements.txt
```

## 💻 Utilisation

### Mode TEST (Validation locale rapide)

Utilise des **données synthétiques** avec bruit speckle simulé :

```bash
# Windows
run.bat test

# Linux/Mac
./run.sh test

# Ou directement
python src/main.py --mode test
```

**Paramètres TEST** : 16 frames, 3 époques, batch=4, ~100 samples

### Mode REAL (Production)

Utilise **NoduleMNIST3D** (nodules pulmonaires CT) :

```bash
# Windows
run.bat real

# Linux/Mac
./run.sh real

# Ou directement
python src/main.py --mode real
```

**Paramètres REAL** : 28 slices, 15 époques, batch=8, ~1600 samples

### Options supplémentaires

```bash
# Dry-run (vérifie la config sans entraîner)
python src/main.py --mode test --dry-run

# Custom epochs
python src/main.py --mode test --epochs 10

# Custom batch size
python src/main.py --mode real --batch-size 4

# Custom learning rate
python src/main.py --mode test --lr 0.0001
```

## 📊 Résultats Attendus

| Modèle | Mode | Accuracy Attendue |
|--------|------|-------------------|
| Baseline (Avg Pool) | TEST | ~50-60% |
| MedViT-CAMIL | TEST | ~70-85% |
| Baseline (Avg Pool) | REAL | ~60-70% |
| MedViT-CAMIL | REAL | ~80-90% |

### Visualisations générées

- `training_curves_*.png` : Courbes loss/accuracy comparatives
- `attention_heatmap_*.png` : Heatmaps d'attention temporelle
- `attention_comparison_*.png` : Comparaison Baseline vs CAMIL
- `results_*.json` : Métriques détaillées

## 🔬 Dataset: NoduleMNIST3D

[NoduleMNIST3D](https://medmnist.com/) fait partie de MedMNIST v2 (Nature Scientific Data 2023).

| Caractéristique | Valeur |
|-----------------|--------|
| Modalité | Chest CT |
| Tâche | Classification binaire (malin/bénin) |
| Dimensions | 28×28×28 |
| Train/Val/Test | 1,158 / 165 / 310 |
| Téléchargement | Automatique (~50 Mo) |

> **Note** : NoduleMNIST3D est un volume 3D spatial, pas une vidéo temporelle. Cependant, mathématiquement le tenseur $(D, H, W)$ se traite identiquement à $(T, H, W)$, ce qui est acceptable pour le PoC.

## 🐳 Docker (Optionnel)

```bash
# Build
docker build -t medvit-camil .

# Run mode test
docker run --gpus all -v $(pwd)/results:/app/results medvit-camil test

# Run mode real
docker run --gpus all -v $(pwd)/results:/app/results medvit-camil real
```

## 📚 Références

- **MobileViT** : Mehta & Rastegari, "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer", ICLR 2022
- **Gated Attention MIL** : Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018
- **MedMNIST** : Yang et al., "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification", Nature Scientific Data 2023

## 📝 Licence

MIT License - Voir [LICENSE](LICENSE)

## 👥 Auteurs

Projet de recherche M2 - ENSPY (École Nationale Supérieure Polytechnique de Yaoundé)

*"Design of Next-Generation Generative and Agentic AI Architectures for Complex, Long-Horizon, and Multimodal Intelligence Tasks"*
