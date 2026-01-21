"""
MedViT-CAMIL Configuration Module
=================================
Gère les arguments et configurations pour les modes TEST et REAL.

Usage:
    python main.py --mode test   # Validation locale rapide
    python main.py --mode real   # Production avec NoduleMNIST3D
"""

import argparse
import torch
from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    """Configuration centralisée du projet MedViT-CAMIL."""
    
    # Mode d'exécution
    mode: Literal["test", "real"]
    
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
    num_train_samples: int   # Nombre d'échantillons d'entraînement (mode test)
    num_val_samples: int     # Nombre d'échantillons de validation (mode test)
    
    # Device
    device: torch.device
    
    # Chemins
    results_dir: str
    data_dir: str
    
    # Reproductibilité
    seed: int


# Configurations prédéfinies
TEST_CONFIG = {
    "seq_len": 16,
    "img_size": 224,
    "in_channels": 3,
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "feature_dim": 512,      # MobileViT-S output dim
    "hidden_dim": 128,
    "num_classes": 2,
    "num_train_samples": 100,
    "num_val_samples": 20,
    "results_dir": "./results",
    "data_dir": "./data",
    "seed": 42,
}

REAL_CONFIG = {
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
    "num_train_samples": -1,  # Ignoré en mode real
    "num_val_samples": -1,
    "results_dir": "./results",
    "data_dir": "./data",
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


def parse_args() -> Config:
    """Parse les arguments de ligne de commande et retourne une Config."""
    parser = argparse.ArgumentParser(
        description="MedViT-CAMIL: Détection d'anomalies médicales temporelles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["test", "real"], 
        default="test",
        help="Mode d'exécution: 'test' pour validation locale, 'real' pour NoduleMNIST3D"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Nombre d'époques (override la config par défaut)"
    )
    
    parser.add_argument(
        "--batch-size",
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
        default="./results",
        help="Répertoire pour sauvegarder les résultats"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Répertoire pour les données"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Vérifie la configuration et les données sans entraîner"
    )
    
    args = parser.parse_args()
    
    # Sélectionner la configuration de base
    base_config = TEST_CONFIG.copy() if args.mode == "test" else REAL_CONFIG.copy()
    
    # Override avec les arguments CLI
    if args.epochs is not None:
        base_config["epochs"] = args.epochs
    if args.batch_size is not None:
        base_config["batch_size"] = args.batch_size
    if args.lr is not None:
        base_config["learning_rate"] = args.lr
    
    base_config["seed"] = args.seed
    base_config["results_dir"] = args.results_dir
    base_config["data_dir"] = args.data_dir
    
    # Créer l'objet Config
    config = Config(
        mode=args.mode,
        device=get_device(),
        **base_config
    )
    
    return config, args.dry_run


def print_config(config: Config) -> None:
    """Affiche la configuration de manière formatée."""
    print("\n" + "=" * 60)
    print("CONFIGURATION MedViT-CAMIL")
    print("=" * 60)
    print(f"Mode:              {config.mode.upper()}")
    print(f"Device:            {config.device}")
    print(f"Seed:              {config.seed}")
    print("-" * 60)
    print("DONNÉES:")
    print(f"  Sequence Length: {config.seq_len}")
    print(f"  Image Size:      {config.img_size}x{config.img_size}")
    print(f"  Channels:        {config.in_channels}")
    if config.mode == "test":
        print(f"  Train Samples:   {config.num_train_samples}")
        print(f"  Val Samples:     {config.num_val_samples}")
    else:
        print("  Dataset:         NoduleMNIST3D (téléchargement auto)")
    print("-" * 60)
    print("ENTRAÎNEMENT:")
    print(f"  Epochs:          {config.epochs}")
    print(f"  Batch Size:      {config.batch_size}")
    print(f"  Learning Rate:   {config.learning_rate}")
    print(f"  Weight Decay:    {config.weight_decay}")
    print("-" * 60)
    print("MODÈLE:")
    print(f"  Feature Dim:     {config.feature_dim}")
    print(f"  Hidden Dim:      {config.hidden_dim}")
    print(f"  Num Classes:     {config.num_classes}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test de la configuration
    config, dry_run = parse_args()
    print_config(config)
    
    if dry_run:
        print("[DRY-RUN] Configuration validée. Pas d'entraînement.")
