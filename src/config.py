"""
MedViT-CAMIL Configuration Module V2
=====================================
Gère les arguments et configurations pour les 3 modes:
- TEST: Données synthétiques (validation locale rapide)
- PROXY: NoduleMNIST3D (preuve scientifique intermédiaire)
- REAL: HyperKvasir + vraies vidéos OpenCV (serveur professeur)

Usage:
    python -m src.main --mode test    # Laptop - données synthétiques
    python -m src.main --mode proxy   # Validation - MedMNIST3D
    python -m src.main --mode real    # Serveur - HyperKvasir/vidéos réelles
"""

import argparse
import torch
import os
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Config:
    """Configuration centralisée du projet MedViT-CAMIL."""
    
    # Mode d'exécution
    mode: Literal["test", "proxy", "real"]
    
    # Paramètres de séquence
    seq_len: int          # Nombre de frames/slices dans la séquence
    img_size: int         # Taille des images (224 pour MobileViT)
    in_channels: int      # Canaux d'entrée (3 pour RGB)
    
    # Paramètres d'entraînement
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    
    # Paramètres du modèle
    feature_dim: int      # Dimension des features MobileViT
    hidden_dim: int       # Dimension cachée du module CAMIL
    num_classes: int      # Nombre de classes (2 pour binaire)
    
    # Paramètres de données
    num_train_samples: int   # Nombre d'échantillons (mode test uniquement)
    num_val_samples: int     # Nombre d'échantillons (mode test uniquement)
    num_workers: int         # Workers pour DataLoader
    
    # Device
    device: torch.device
    
    # Chemins
    results_dir: str
    data_dir: str
    
    # Reproductibilité
    seed: int


# ============================================================================
# CONFIGURATIONS PRÉDÉFINIES POUR CHAQUE MODE
# ============================================================================

TEST_CONFIG = {
    "seq_len": 16,
    "img_size": 224,
    "in_channels": 3,
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "feature_dim": 512,
    "hidden_dim": 128,
    "num_classes": 2,
    "num_train_samples": 100,
    "num_val_samples": 30,
    "num_workers": 0,        # 0 pour Windows compatibility
    "results_dir": "./results",
    "data_dir": "./data",
    "seed": 42,
}

PROXY_CONFIG = {
    "seq_len": 28,           # Profondeur NoduleMNIST3D
    "img_size": 224,
    "in_channels": 3,
    "epochs": 15,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "feature_dim": 512,
    "hidden_dim": 128,
    "num_classes": 2,
    "num_train_samples": -1,  # Ignoré
    "num_val_samples": -1,
    "num_workers": 0,
    "results_dir": "./results",
    "data_dir": "./data",
    "seed": 42,
}

REAL_CONFIG = {
    "seq_len": 32,           # Séquences plus longues pour vraies vidéos
    "img_size": 224,
    "in_channels": 3,
    "epochs": 50,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "feature_dim": 512,
    "hidden_dim": 128,
    "num_classes": 2,
    "num_train_samples": -1,
    "num_val_samples": -1,
    "num_workers": 4,        # Multi-threading pour serveur
    "results_dir": "./results",
    "data_dir": "/app/data/hyperkvasir",  # Chemin Docker
    "seed": 42,
}


def get_device() -> torch.device:
    """Détecte automatiquement le meilleur device disponible (GPU/CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] CUDA disponible: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Apple MPS disponible")
    else:
        device = torch.device("cpu")
        print("[INFO] Utilisation du CPU")
    return device


def parse_args():
    """Parse les arguments de ligne de commande et retourne une Config."""
    parser = argparse.ArgumentParser(
        description="MedViT-CAMIL: Détection d'anomalies médicales temporelles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["test", "proxy", "real"], 
        default="test",
        help="Mode: 'test' (synthétique), 'proxy' (MedMNIST3D), 'real' (HyperKvasir/vidéos)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Nombre d'époques (override la config par défaut)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Taille du batch (override la config par défaut)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (override la config par défaut)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour la reproductibilité"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Répertoire pour sauvegarder les résultats"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Répertoire des données"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Vérifie la configuration et les données sans entraîner"
    )
    
    args = parser.parse_args()
    
    # Sélectionner la configuration de base selon le mode
    if args.mode == "test":
        base_config = TEST_CONFIG.copy()
    elif args.mode == "proxy":
        base_config = PROXY_CONFIG.copy()
    else:
        base_config = REAL_CONFIG.copy()
    
    # Override avec les arguments CLI
    if args.epochs is not None:
        base_config["epochs"] = args.epochs
    if args.batch_size is not None:
        base_config["batch_size"] = args.batch_size
    if args.lr is not None:
        base_config["learning_rate"] = args.lr
    if args.results_dir is not None:
        base_config["results_dir"] = args.results_dir
    if args.data_dir is not None:
        base_config["data_dir"] = args.data_dir
    
    base_config["seed"] = args.seed
    
    # Créer les répertoires
    os.makedirs(base_config["results_dir"], exist_ok=True)
    os.makedirs(base_config["data_dir"], exist_ok=True)
    
    # Créer l'objet Config
    config = Config(
        mode=args.mode,
        device=get_device(),
        **base_config
    )
    
    return config, args.dry_run


def print_config(config: Config) -> None:
    """Affiche la configuration de manière formatée."""
    mode_desc = {
        "test": "TEST (Données synthétiques - Validation locale)",
        "proxy": "PROXY (NoduleMNIST3D - Preuve intermédiaire)",
        "real": "REAL (HyperKvasir/Vidéos - Serveur Production)"
    }
    
    print("\n" + "=" * 60)
    print("CONFIGURATION MedViT-CAMIL V2")
    print("=" * 60)
    print(f"Mode:              {mode_desc.get(config.mode, config.mode)}")
    print(f"Device:            {config.device}")
    print(f"Seed:              {config.seed}")
    print("-" * 60)
    print("DONNÉES:")
    print(f"  Sequence Length: {config.seq_len}")
    print(f"  Image Size:      {config.img_size}x{config.img_size}")
    print(f"  Channels:        {config.in_channels}")
    print(f"  Data Directory:  {config.data_dir}")
    if config.mode == "test":
        print(f"  Train Samples:   {config.num_train_samples}")
        print(f"  Val Samples:     {config.num_val_samples}")
    elif config.mode == "proxy":
        print("  Dataset:         NoduleMNIST3D (téléchargement auto)")
    else:
        print("  Dataset:         HyperKvasir + vidéos OpenCV")
    print("-" * 60)
    print("ENTRAÎNEMENT:")
    print(f"  Epochs:          {config.epochs}")
    print(f"  Batch Size:      {config.batch_size}")
    print(f"  Learning Rate:   {config.learning_rate}")
    print(f"  Weight Decay:    {config.weight_decay}")
    print(f"  Num Workers:     {config.num_workers}")
    print("-" * 60)
    print("MODÈLE:")
    print(f"  Feature Dim:     {config.feature_dim}")
    print(f"  Hidden Dim:      {config.hidden_dim}")
    print(f"  Num Classes:     {config.num_classes}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    config, dry_run = parse_args()
    print_config(config)
    
    if dry_run:
        print("[DRY-RUN] Configuration validée. Pas d'entraînement.")
