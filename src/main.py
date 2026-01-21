"""
MedViT-CAMIL Main Training Script
=================================
Script principal pour entraîner et comparer les modèles:
- Baseline (Average Pooling)
- MedViT-CAMIL (Context-Aware Gated Attention MIL)

Usage:
    python src/main.py --mode test    # Validation locale rapide
    python src/main.py --mode real    # Production avec NoduleMNIST3D
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings

# Imports locaux
from config import parse_args, print_config, Config
from dataset import create_dataloaders
from model import create_model


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def set_seed(seed: int) -> None:
    """Fixe les seeds pour la reproductibilité."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Entraîne le modèle pour une époque.
    
    Returns:
        avg_loss: Perte moyenne sur l'époque
        accuracy: Précision sur l'époque
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for batch_idx, (videos, labels, _) in enumerate(pbar):
        videos = videos.to(device)
        labels = labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits, _ = model(videos)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        
        # Gradient clipping pour la stabilité
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Métriques
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_attention: bool = False
) -> Tuple[float, float, Optional[List[np.ndarray]]]:
    """
    Évalue le modèle sur un ensemble de validation/test.
    
    Returns:
        avg_loss: Perte moyenne
        accuracy: Précision
        attention_maps: Liste des cartes d'attention (si collect_attention=True)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    attention_maps = [] if collect_attention else None
    all_labels = []
    
    pbar = tqdm(val_loader, desc="Evaluating", leave=False)
    
    for videos, labels, gt_attention in pbar:
        videos = videos.to(device)
        labels = labels.to(device)
        
        logits, attention = model(videos)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        if collect_attention:
            attention_maps.append(attention.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    if collect_attention:
        attention_maps = np.concatenate(attention_maps, axis=0)
    
    return avg_loss, accuracy, attention_maps


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(
    history: Dict[str, Dict[str, List[float]]],
    save_path: str
) -> None:
    """
    Trace les courbes d'entraînement comparatives.
    
    Args:
        history: Dict avec les historiques de chaque modèle
        save_path: Chemin pour sauvegarder la figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'baseline': '#e74c3c', 'camil': '#2ecc71'}
    
    # Loss curves
    ax = axes[0]
    for model_name, hist in history.items():
        epochs = range(1, len(hist['train_loss']) + 1)
        ax.plot(epochs, hist['train_loss'], '-', 
                color=colors[model_name], label=f'{model_name.upper()} Train')
        ax.plot(epochs, hist['val_loss'], '--', 
                color=colors[model_name], label=f'{model_name.upper()} Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[1]
    for model_name, hist in history.items():
        epochs = range(1, len(hist['train_acc']) + 1)
        ax.plot(epochs, [a * 100 for a in hist['train_acc']], '-', 
                color=colors[model_name], label=f'{model_name.upper()} Train')
        ax.plot(epochs, [a * 100 for a in hist['val_acc']], '--', 
                color=colors[model_name], label=f'{model_name.upper()} Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training & Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Courbes sauvegardées: {save_path}")


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    model_name: str,
    num_samples: int = 10
) -> None:
    """
    Trace une heatmap des poids d'attention temporelle.
    
    Args:
        attention_weights: Array (N, T) des poids d'attention
        labels: Array (N,) des labels
        save_path: Chemin pour sauvegarder
        model_name: Nom du modèle pour le titre
        num_samples: Nombre d'échantillons à afficher
    """
    # Sélectionner quelques échantillons de chaque classe
    idx_class0 = np.where(labels == 0)[0][:num_samples // 2]
    idx_class1 = np.where(labels == 1)[0][:num_samples // 2]
    indices = np.concatenate([idx_class0, idx_class1])
    
    if len(indices) == 0:
        print("[WARNING] Pas assez d'échantillons pour la heatmap")
        return
    
    selected_attention = attention_weights[indices]
    selected_labels = labels[indices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(selected_attention, aspect='auto', cmap='hot')
    ax.set_xlabel('Frame (temps)')
    ax.set_ylabel('Échantillon')
    ax.set_title(f'Attention Temporelle - {model_name}')
    
    # Ajouter les labels sur l'axe Y
    yticks = [f"{'Anomalie' if l == 1 else 'Normal'} #{i}" 
              for i, l in enumerate(selected_labels)]
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks)
    
    plt.colorbar(im, ax=ax, label='Poids attention')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Heatmap sauvegardée: {save_path}")


def plot_attention_comparison(
    baseline_attention: np.ndarray,
    camil_attention: np.ndarray,
    gt_attention: np.ndarray,
    save_path: str,
    sample_idx: int = 0
) -> None:
    """
    Compare l'attention du Baseline vs CAMIL pour un échantillon.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    T = len(baseline_attention[sample_idx])
    x = np.arange(T)
    
    # Ground truth (si disponible)
    ax = axes[0]
    ax.bar(x, gt_attention[sample_idx], color='green', alpha=0.7)
    ax.set_ylabel('GT Attention')
    ax.set_title('Ground Truth (frames avec anomalie)')
    ax.set_ylim(0, 1.1)
    
    # Baseline
    ax = axes[1]
    ax.bar(x, baseline_attention[sample_idx], color='red', alpha=0.7)
    ax.set_ylabel('Attention')
    ax.set_title('Baseline (Average Pooling) - Attention uniforme')
    ax.set_ylim(0, max(baseline_attention[sample_idx]) * 1.2 + 0.01)
    
    # CAMIL
    ax = axes[2]
    ax.bar(x, camil_attention[sample_idx], color='blue', alpha=0.7)
    ax.set_ylabel('Attention')
    ax.set_xlabel('Frame (temps)')
    ax.set_title('MedViT-CAMIL - Attention apprise')
    ax.set_ylim(0, max(camil_attention[sample_idx]) * 1.2 + 0.01)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Comparaison attention sauvegardée: {save_path}")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Config,
    model_name: str
) -> Dict[str, List[float]]:
    """
    Boucle d'entraînement complète pour un modèle.
    
    Returns:
        history: Dictionnaire avec l'historique d'entraînement
    """
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT: {model_name}")
    print(f"{'='*60}")
    
    # Criterion et Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Seuls les paramètres entraînables (pas le backbone gelé)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.01
    )
    
    # Historique
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(config.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device, epoch
        )
        
        # Validate
        val_loss, val_acc, _ = evaluate(
            model, val_loader, criterion, config.device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(
                config.results_dir, 
                f"{model_name.lower().replace('-', '_')}_best.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    print(f"\n[INFO] Meilleure Val Accuracy: {best_val_acc*100:.2f}%")
    
    return history


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Point d'entrée principal."""
    
    # Parser les arguments
    config, dry_run = parse_args()
    
    # Afficher la configuration
    print_config(config)
    
    # Fixer la seed
    set_seed(config.seed)
    
    # Créer le répertoire de résultats
    os.makedirs(config.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dry run: vérifier la config et les données
    if dry_run:
        print("\n[DRY-RUN] Vérification de la configuration...")
        print("[DRY-RUN] Création d'un batch de test...")
        
        train_loader, val_loader, _ = create_dataloaders(
            mode=config.mode,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            img_size=config.img_size,
            num_train_samples=min(10, config.num_train_samples),
            num_val_samples=min(5, config.num_val_samples),
            data_dir=config.data_dir,
            seed=config.seed
        )
        
        # Test d'un batch
        videos, labels, attention = next(iter(train_loader))
        print(f"[DRY-RUN] Video shape: {videos.shape}")
        print(f"[DRY-RUN] Labels shape: {labels.shape}")
        print(f"[DRY-RUN] Attention shape: {attention.shape}")
        
        # Test du modèle
        print("[DRY-RUN] Test du modèle CAMIL...")
        model = create_model(
            model_type="camil",
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            device=config.device
        )
        
        videos = videos.to(config.device)
        logits, att = model(videos)
        print(f"[DRY-RUN] Output logits: {logits.shape}")
        print(f"[DRY-RUN] Output attention: {att.shape}")
        
        print("\n[DRY-RUN] ✓ Tout fonctionne correctement!")
        return
    
    # Créer les DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        mode=config.mode,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        img_size=config.img_size,
        num_train_samples=config.num_train_samples,
        num_val_samples=config.num_val_samples,
        data_dir=config.data_dir,
        seed=config.seed
    )
    
    # Créer et entraîner les deux modèles
    history = {}
    models = {}
    
    # 1. Baseline (Average Pooling)
    print("\n" + "=" * 60)
    print("MODÈLE 1: BASELINE (Average Pooling)")
    print("=" * 60)
    
    model_baseline = create_model(
        model_type="baseline",
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        device=config.device
    )
    
    history['baseline'] = train_model(
        model_baseline, train_loader, val_loader, config, "Baseline"
    )
    models['baseline'] = model_baseline
    
    # 2. MedViT-CAMIL
    print("\n" + "=" * 60)
    print("MODÈLE 2: MedViT-CAMIL (Context-Aware Gated Attention)")
    print("=" * 60)
    
    model_camil = create_model(
        model_type="camil",
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        device=config.device
    )
    
    history['camil'] = train_model(
        model_camil, train_loader, val_loader, config, "MedViT-CAMIL"
    )
    models['camil'] = model_camil
    
    # ========================================================================
    # ÉVALUATION FINALE ET VISUALISATIONS
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("ÉVALUATION FINALE")
    print("=" * 60)
    
    criterion = nn.CrossEntropyLoss()
    eval_loader = test_loader if test_loader is not None else val_loader
    
    # Collecter les attentions pour la visualisation
    results = {}
    attention_data = {}
    
    for model_name, model in models.items():
        loss, acc, attention = evaluate(
            model, eval_loader, criterion, config.device, collect_attention=True
        )
        results[model_name] = {'loss': loss, 'accuracy': acc}
        attention_data[model_name] = attention
        print(f"{model_name.upper()}: Loss={loss:.4f}, Accuracy={acc*100:.2f}%")
    
    # Collecter les labels et ground truth attention
    all_labels = []
    all_gt_attention = []
    for videos, labels, gt_att in eval_loader:
        all_labels.extend(labels.numpy())
        all_gt_attention.append(gt_att.numpy())
    all_labels = np.array(all_labels)
    all_gt_attention = np.concatenate(all_gt_attention, axis=0)
    
    # ========================================================================
    # GÉNÉRATION DES GRAPHIQUES
    # ========================================================================
    
    print("\n[INFO] Génération des visualisations...")
    
    # 1. Courbes d'entraînement
    plot_training_curves(
        history,
        os.path.join(config.results_dir, f"training_curves_{timestamp}.png")
    )
    
    # 2. Heatmaps d'attention
    for model_name in ['baseline', 'camil']:
        plot_attention_heatmap(
            attention_data[model_name],
            all_labels,
            os.path.join(config.results_dir, f"attention_heatmap_{model_name}_{timestamp}.png"),
            model_name.upper()
        )
    
    # 3. Comparaison d'attention sur un échantillon
    # Trouver un échantillon avec anomalie (label=1)
    anomaly_indices = np.where(all_labels == 1)[0]
    if len(anomaly_indices) > 0:
        sample_idx = anomaly_indices[0]
        plot_attention_comparison(
            attention_data['baseline'],
            attention_data['camil'],
            all_gt_attention,
            os.path.join(config.results_dir, f"attention_comparison_{timestamp}.png"),
            sample_idx=sample_idx
        )
    
    # ========================================================================
    # SAUVEGARDE DES RÉSULTATS
    # ========================================================================
    
    # Sauvegarder les résultats en JSON
    results_summary = {
        'mode': config.mode,
        'timestamp': timestamp,
        'config': {
            'seq_len': config.seq_len,
            'img_size': config.img_size,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
        },
        'results': {
            'baseline': {
                'final_val_acc': results['baseline']['accuracy'] * 100,
                'final_val_loss': results['baseline']['loss'],
            },
            'camil': {
                'final_val_acc': results['camil']['accuracy'] * 100,
                'final_val_loss': results['camil']['loss'],
            }
        },
        'history': {
            model_name: {
                'train_loss': [float(x) for x in hist['train_loss']],
                'train_acc': [float(x) for x in hist['train_acc']],
                'val_loss': [float(x) for x in hist['val_loss']],
                'val_acc': [float(x) for x in hist['val_acc']],
            }
            for model_name, hist in history.items()
        }
    }
    
    results_path = os.path.join(config.results_dir, f"results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"[INFO] Résultats sauvegardés: {results_path}")
    
    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"Mode: {config.mode.upper()}")
    print(f"Device: {config.device}")
    print("-" * 40)
    print(f"BASELINE (Avg Pool):")
    print(f"  - Accuracy: {results['baseline']['accuracy']*100:.2f}%")
    print(f"  - Loss: {results['baseline']['loss']:.4f}")
    print("-" * 40)
    print(f"MedViT-CAMIL:")
    print(f"  - Accuracy: {results['camil']['accuracy']*100:.2f}%")
    print(f"  - Loss: {results['camil']['loss']:.4f}")
    print("-" * 40)
    
    improvement = (results['camil']['accuracy'] - results['baseline']['accuracy']) * 100
    print(f"Amélioration CAMIL vs Baseline: {improvement:+.2f}%")
    print("=" * 60)
    
    print(f"\n[INFO] Tous les résultats sauvegardés dans: {config.results_dir}/")


if __name__ == "__main__":
    main()
