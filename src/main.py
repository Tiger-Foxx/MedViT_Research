"""
MedViT-CAMIL Main Training Script V2
====================================
Script principal avec support des 3 modes:
- TEST: Données synthétiques (validation locale)
- PROXY: NoduleMNIST3D (preuve intermédiaire)
- REAL: HyperKvasir + vidéos OpenCV (serveur)

Usage:
    python -m src.main --mode test
    python -m src.main --mode proxy
    python -m src.main --mode real
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
from typing import Dict, List, Tuple

# Imports locaux - support exécution directe et module
try:
    from src.config import parse_args, print_config, Config
    from src.dataset import get_loaders
    from src.model import create_model
except ImportError:
    from config import parse_args, print_config, Config
    from dataset import get_loaders
    from model import create_model


def set_seed(seed: int) -> None:
    """Fixe les seeds pour la reproductibilité."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
    """Entraîne le modèle pour une époque."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for videos, labels in pbar:
        videos = videos.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(videos)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_attention: bool = False
) -> Tuple[float, float, List]:
    """Évalue le modèle."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    attention_maps = []
    
    for videos, labels in tqdm(val_loader, desc="Evaluating", leave=False):
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
    
    if collect_attention and attention_maps:
        attention_maps = np.concatenate(attention_maps, axis=0)
    
    return total_loss / len(val_loader), correct / total, attention_maps


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Config,
    model_name: str
) -> Dict[str, List[float]]:
    """Boucle d'entraînement complète."""
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT: {model_name}")
    print(f"{'='*60}")
    
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device, epoch
        )
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, config.device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(config.results_dir, f"{model_name.lower().replace('-', '_')}_best.pth"))
    
    print(f"\n[INFO] Meilleure Val Accuracy: {best_val_acc*100:.2f}%")
    return history


def plot_results(
    history: Dict[str, Dict[str, List[float]]],
    save_path: str
) -> None:
    """Trace les courbes d'entraînement comparatives."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'baseline': '#e74c3c', 'camil': '#2ecc71'}
    
    # Loss
    ax = axes[0]
    for model_name, hist in history.items():
        epochs = range(1, len(hist['train_loss']) + 1)
        ax.plot(epochs, hist['train_loss'], '-', color=colors.get(model_name, 'blue'), 
                label=f'{model_name.upper()} Train', linewidth=2)
        ax.plot(epochs, hist['val_loss'], '--', color=colors.get(model_name, 'blue'), 
                label=f'{model_name.upper()} Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[1]
    for model_name, hist in history.items():
        epochs = range(1, len(hist['train_acc']) + 1)
        ax.plot(epochs, [a * 100 for a in hist['train_acc']], '-', 
                color=colors.get(model_name, 'blue'), label=f'{model_name.upper()} Train', linewidth=2)
        ax.plot(epochs, [a * 100 for a in hist['val_acc']], '--', 
                color=colors.get(model_name, 'blue'), label=f'{model_name.upper()} Val', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training & Validation Accuracy', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Courbes sauvegardées: {save_path}")


def plot_attention_heatmap(
    attention_baseline: np.ndarray,
    attention_camil: np.ndarray,
    save_path: str,
    num_samples: int = 8
) -> None:
    """Compare les heatmaps d'attention Baseline vs CAMIL."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    n = min(num_samples, len(attention_baseline), len(attention_camil))
    
    # Baseline
    ax = axes[0]
    im = ax.imshow(attention_baseline[:n], aspect='auto', cmap='Reds')
    ax.set_ylabel('Échantillon')
    ax.set_title('Baseline (Average Pooling) - Attention uniforme', fontsize=12)
    plt.colorbar(im, ax=ax)
    
    # CAMIL
    ax = axes[1]
    im = ax.imshow(attention_camil[:n], aspect='auto', cmap='Greens')
    ax.set_xlabel('Frame (temps)')
    ax.set_ylabel('Échantillon')
    ax.set_title('MedViT-CAMIL - Attention apprise (pics sur anomalies)', fontsize=12)
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Heatmap attention sauvegardée: {save_path}")


def main():
    """Point d'entrée principal."""
    config, dry_run = parse_args()
    print_config(config)
    set_seed(config.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dry run
    if dry_run:
        print("\n[DRY-RUN] Vérification de la configuration...")
        train_loader, val_loader = get_loaders(config)
        
        videos, labels = next(iter(train_loader))
        print(f"[DRY-RUN] Video shape: {videos.shape}")
        print(f"[DRY-RUN] Labels shape: {labels.shape}")
        
        print("[DRY-RUN] Test du modèle CAMIL...")
        model = create_model(model_type="camil", device=config.device)
        
        videos = videos.to(config.device)
        logits, att = model(videos)
        print(f"[DRY-RUN] Output logits: {logits.shape}")
        print(f"[DRY-RUN] Output attention: {att.shape}")
        
        print("\n[DRY-RUN] ✓ Configuration validée!")
        return
    
    # Créer les DataLoaders
    train_loader, val_loader = get_loaders(config)
    
    # Entraîner les deux modèles
    history = {}
    models = {}
    
    # 1. Baseline
    print("\n" + "=" * 60)
    print("MODÈLE 1: BASELINE (Average Pooling)")
    print("=" * 60)
    
    model_baseline = create_model(model_type="baseline", device=config.device)
    history['baseline'] = train_model(model_baseline, train_loader, val_loader, config, "Baseline")
    models['baseline'] = model_baseline
    
    # 2. CAMIL
    print("\n" + "=" * 60)
    print("MODÈLE 2: MedViT-CAMIL (Gated Attention)")
    print("=" * 60)
    
    model_camil = create_model(model_type="camil", device=config.device)
    history['camil'] = train_model(model_camil, train_loader, val_loader, config, "MedViT-CAMIL")
    models['camil'] = model_camil
    
    # Évaluation finale
    print("\n" + "=" * 60)
    print("ÉVALUATION FINALE")
    print("=" * 60)
    
    criterion = nn.CrossEntropyLoss()
    results = {}
    attention_data = {}
    
    for model_name, model in models.items():
        loss, acc, attention = evaluate(model, val_loader, criterion, config.device, collect_attention=True)
        results[model_name] = {'loss': loss, 'accuracy': acc}
        attention_data[model_name] = attention
        print(f"{model_name.upper()}: Loss={loss:.4f}, Accuracy={acc*100:.2f}%")
    
    # Génération des graphiques
    print("\n[INFO] Génération des visualisations...")
    
    plot_results(history, os.path.join(config.results_dir, f"training_curves_{timestamp}.png"))
    
    if len(attention_data.get('baseline', [])) > 0 and len(attention_data.get('camil', [])) > 0:
        plot_attention_heatmap(
            attention_data['baseline'],
            attention_data['camil'],
            os.path.join(config.results_dir, f"attention_comparison_{timestamp}.png")
        )
    
    # Sauvegarder les résultats JSON
    results_summary = {
        'mode': config.mode,
        'timestamp': timestamp,
        'device': str(config.device),
        'config': {
            'seq_len': config.seq_len,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
        },
        'results': {
            name: {'accuracy': r['accuracy'] * 100, 'loss': r['loss']}
            for name, r in results.items()
        },
        'history': {
            name: {k: [float(v) for v in vals] for k, vals in hist.items()}
            for name, hist in history.items()
        }
    }
    
    with open(os.path.join(config.results_dir, f"results_{timestamp}.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Résumé final
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    print("=" * 60)
    print(f"Mode: {config.mode.upper()}")
    print(f"Device: {config.device}")
    print("-" * 40)
    print(f"BASELINE: {results['baseline']['accuracy']*100:.2f}%")
    print(f"CAMIL:    {results['camil']['accuracy']*100:.2f}%")
    print("-" * 40)
    improvement = (results['camil']['accuracy'] - results['baseline']['accuracy']) * 100
    print(f"Amélioration: {improvement:+.2f}%")
    print("=" * 60)
    print(f"\n✅ Résultats sauvegardés dans: {config.results_dir}/")


if __name__ == "__main__":
    main()
