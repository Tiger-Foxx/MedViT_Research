"""
MedViT-CAMIL Dataset Module
===========================
Gère les données pour les modes TEST (synthétique) et REAL (NoduleMNIST3D).

Format de sortie unifié: (B, T, C, H, W)
- B: Batch size
- T: Sequence length (frames temporelles)
- C: Channels (3 pour RGB)
- H, W: Height, Width (224x224)
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional
import warnings


# ============================================================================
# NORMALISATION STANDARD
# ============================================================================

# Normalisation ImageNet (utilisée par MobileViT pré-entraîné)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """Retourne les transformations pour les images."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ============================================================================
# MODE TEST: DONNÉES SYNTHÉTIQUES AVEC BRUIT SPECKLE
# ============================================================================

class SyntheticSpeckleDataset(Dataset):
    """
    Dataset synthétique simulant des vidéos médicales avec:
    - Bruit speckle (typique de l'échographie)
    - Lésions mouvantes apparaissant sur quelques frames seulement
    
    Simule le problème de "l'aiguille dans la botte de foin":
    - Classe 0: Séquence normale (bruit seulement)
    - Classe 1: Séquence avec anomalie (lésion sur quelques frames)
    """
    
    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        img_size: int = 224,
        anomaly_frames: int = 3,  # Nombre de frames avec l'anomalie
        seed: Optional[int] = None
    ):
        """
        Args:
            num_samples: Nombre total d'échantillons
            seq_len: Longueur de la séquence temporelle
            img_size: Taille des images
            anomaly_frames: Nombre de frames contenant l'anomalie (pour classe 1)
            seed: Seed pour reproductibilité
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.img_size = img_size
        self.anomaly_frames = min(anomaly_frames, seq_len)
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Pré-générer les labels (50% normal, 50% anomalie)
        self.labels = torch.zeros(num_samples, dtype=torch.long)
        self.labels[num_samples // 2:] = 1
        
        # Pré-générer les positions des anomalies
        self.anomaly_positions = []
        for i in range(num_samples):
            if self.labels[i] == 1:
                # Choisir des frames consécutives pour l'anomalie
                start = np.random.randint(0, self.seq_len - self.anomaly_frames + 1)
                positions = list(range(start, start + self.anomaly_frames))
            else:
                positions = []
            self.anomaly_positions.append(positions)
    
    def _generate_speckle_noise(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Génère du bruit speckle réaliste (distribution gamma)."""
        # Le bruit speckle suit une distribution gamma
        noise = np.random.gamma(shape=2.0, scale=0.5, size=shape)
        noise = np.clip(noise, 0, 1)
        return noise.astype(np.float32)
    
    def _generate_lesion(self, img_size: int) -> np.ndarray:
        """Génère une lésion elliptique brillante."""
        lesion = np.zeros((img_size, img_size), dtype=np.float32)
        
        # Position et taille aléatoires
        cx = np.random.randint(img_size // 4, 3 * img_size // 4)
        cy = np.random.randint(img_size // 4, 3 * img_size // 4)
        rx = np.random.randint(10, 30)
        ry = np.random.randint(10, 30)
        
        # Créer l'ellipse
        y, x = np.ogrid[:img_size, :img_size]
        mask = ((x - cx) ** 2 / (rx ** 2 + 1e-6) + (y - cy) ** 2 / (ry ** 2 + 1e-6)) <= 1
        
        # Intensité avec gradient (plus brillant au centre)
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        intensity = np.exp(-dist / (max(rx, ry) * 0.5))
        lesion[mask] = intensity[mask] * 0.8
        
        return lesion
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            video: Tensor (T, 3, H, W) - Séquence vidéo
            label: Tensor scalar - 0 ou 1
            attention_mask: Tensor (T,) - 1 pour frames avec anomalie, 0 sinon
        """
        label = self.labels[idx]
        anomaly_pos = self.anomaly_positions[idx]
        
        # Générer la séquence
        frames = []
        attention_mask = torch.zeros(self.seq_len)
        
        for t in range(self.seq_len):
            # Base: bruit speckle
            frame = self._generate_speckle_noise((self.img_size, self.img_size))
            
            # Ajouter une structure de fond (simuler un tissu)
            background = np.sin(np.linspace(0, 4 * np.pi, self.img_size)).reshape(1, -1)
            background = background * np.sin(np.linspace(0, 4 * np.pi, self.img_size)).reshape(-1, 1)
            background = (background + 1) / 4  # Normaliser entre 0 et 0.5
            frame = frame * 0.5 + background * 0.5
            
            # Ajouter la lésion si c'est une frame anomale
            if t in anomaly_pos:
                lesion = self._generate_lesion(self.img_size)
                frame = np.clip(frame + lesion, 0, 1)
                attention_mask[t] = 1.0
            
            # Convertir en RGB (répliquer le canal)
            frame_rgb = np.stack([frame, frame, frame], axis=0)  # (3, H, W)
            frames.append(frame_rgb)
        
        # Stack toutes les frames
        video = np.stack(frames, axis=0)  # (T, 3, H, W)
        video = torch.from_numpy(video).float()
        
        # Normaliser avec les stats ImageNet
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        video = (video - mean) / std
        
        return video, label, attention_mask


# ============================================================================
# MODE REAL: NODULEMNIST3D
# ============================================================================

class NoduleMNIST3DDataset(Dataset):
    """
    Dataset wrapper pour NoduleMNIST3D de MedMNIST.
    
    NoduleMNIST3D: Scans CT 3D de nodules pulmonaires
    - Dimensions originales: 28x28x28
    - Tâche: Classification binaire (nodule malin vs bénin)
    - On traite la dimension de profondeur (D=28) comme la dimension temporelle (T=28)
    
    Transformations appliquées:
    - Resize: 28x28 → 224x224 (requis par MobileViT)
    - Expand channels: 1 → 3 (grayscale → RGB)
    - Normalisation ImageNet
    """
    
    def __init__(
        self,
        split: str = "train",
        data_dir: str = "./data",
        img_size: int = 224,
        download: bool = True
    ):
        """
        Args:
            split: 'train', 'val', ou 'test'
            data_dir: Répertoire pour télécharger/stocker les données
            img_size: Taille cible des images
            download: Télécharger si non présent
        """
        self.img_size = img_size
        self.split = split
        
        # Importer medmnist
        try:
            import medmnist
            from medmnist import NoduleMNIST3D
        except ImportError:
            raise ImportError(
                "medmnist n'est pas installé. "
                "Installez-le avec: pip install medmnist"
            )
        
        # Créer le répertoire de données
        os.makedirs(data_dir, exist_ok=True)
        
        # Charger le dataset
        print(f"[INFO] Chargement de NoduleMNIST3D (split={split})...")
        self.dataset = NoduleMNIST3D(
            split=split,
            root=data_dir,
            download=download
        )
        
        print(f"[INFO] {len(self.dataset)} échantillons chargés")
        print(f"[INFO] Shape originale: {self.dataset.imgs.shape}")
        
        # Préparer les transformations
        self.resize = transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            video: Tensor (T=28, 3, H=224, W=224) - Volume 3D comme pseudo-vidéo
            label: Tensor scalar - 0 ou 1
            attention_mask: Tensor (T,) - Tous à 1 (pas de ground truth frame-level)
        """
        # Obtenir les données brutes
        # NoduleMNIST3D: imgs shape = (N, 28, 28, 28), labels shape = (N, 1)
        volume = self.dataset.imgs[idx]  # (28, 28, 28)
        label = int(self.dataset.labels[idx].item())
        
        # Convertir en tensor et normaliser à [0, 1]
        volume = torch.from_numpy(volume).float()
        volume = volume / 255.0 if volume.max() > 1.0 else volume
        
        # volume shape: (D=28, H=28, W=28) -> traiter D comme T
        seq_len = volume.shape[0]
        
        frames = []
        for t in range(seq_len):
            # Extraire la slice
            frame = volume[t]  # (28, 28)
            
            # Ajouter dimension channel et répliquer pour RGB
            frame = frame.unsqueeze(0)  # (1, 28, 28)
            frame = frame.expand(3, -1, -1)  # (3, 28, 28)
            
            # Resize vers 224x224
            frame = self.resize(frame)  # (3, 224, 224)
            
            frames.append(frame)
        
        # Stack toutes les frames
        video = torch.stack(frames, dim=0)  # (T, 3, 224, 224)
        
        # Normaliser avec les stats ImageNet
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        video = (video - mean) / std
        
        # Pas de ground truth frame-level, donc attention_mask tout à 1
        attention_mask = torch.ones(seq_len)
        
        return video, torch.tensor(label, dtype=torch.long), attention_mask


# ============================================================================
# FACTORY FUNCTION: CRÉER LES DATALOADERS
# ============================================================================

def create_dataloaders(
    mode: str,
    batch_size: int,
    seq_len: int,
    img_size: int = 224,
    num_train_samples: int = 100,
    num_val_samples: int = 20,
    data_dir: str = "./data",
    seed: int = 42,
    num_workers: int = 0  # 0 pour Windows compatibility
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Crée les DataLoaders pour l'entraînement et la validation.
    
    Args:
        mode: 'test' ou 'real'
        batch_size: Taille du batch
        seq_len: Longueur de séquence (utilisé uniquement en mode test)
        img_size: Taille des images
        num_train_samples: Nombre d'échantillons d'entraînement (mode test)
        num_val_samples: Nombre d'échantillons de validation (mode test)
        data_dir: Répertoire des données
        seed: Seed pour reproductibilité
        num_workers: Nombre de workers pour le chargement (0 recommandé sur Windows)
    
    Returns:
        train_loader, val_loader, test_loader (None en mode test)
    """
    print(f"\n[INFO] Création des DataLoaders (mode={mode})...")
    
    if mode == "test":
        # Mode TEST: données synthétiques
        train_dataset = SyntheticSpeckleDataset(
            num_samples=num_train_samples,
            seq_len=seq_len,
            img_size=img_size,
            seed=seed
        )
        
        val_dataset = SyntheticSpeckleDataset(
            num_samples=num_val_samples,
            seq_len=seq_len,
            img_size=img_size,
            seed=seed + 1  # Seed différent pour la validation
        )
        
        test_loader = None
        
    else:
        # Mode REAL: NoduleMNIST3D
        train_dataset = NoduleMNIST3DDataset(
            split="train",
            data_dir=data_dir,
            img_size=img_size,
            download=True
        )
        
        val_dataset = NoduleMNIST3DDataset(
            split="val",
            data_dir=data_dir,
            img_size=img_size,
            download=True
        )
        
        test_dataset = NoduleMNIST3DDataset(
            split="test",
            data_dir=data_dir,
            img_size=img_size,
            download=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Éviter les problèmes de batch incomplet
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"[INFO] Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"[INFO] Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    if mode == "real":
        print(f"[INFO] Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader if mode == "real" else None


# ============================================================================
# TEST MODULE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU MODULE DATASET")
    print("=" * 60)
    
    # Test données synthétiques
    print("\n[TEST] SyntheticSpeckleDataset:")
    synthetic_ds = SyntheticSpeckleDataset(
        num_samples=10,
        seq_len=16,
        img_size=224,
        seed=42
    )
    video, label, attention = synthetic_ds[0]
    print(f"  Video shape: {video.shape}")  # Attendu: (16, 3, 224, 224)
    print(f"  Label: {label.item()}")
    print(f"  Attention mask: {attention}")
    
    # Test NoduleMNIST3D (si disponible)
    print("\n[TEST] NoduleMNIST3DDataset:")
    try:
        nodule_ds = NoduleMNIST3DDataset(
            split="train",
            data_dir="./data",
            img_size=224,
            download=True
        )
        video, label, attention = nodule_ds[0]
        print(f"  Video shape: {video.shape}")  # Attendu: (28, 3, 224, 224)
        print(f"  Label: {label.item()}")
        print(f"  Attention mask shape: {attention.shape}")
    except Exception as e:
        print(f"  [SKIP] Erreur: {e}")
    
    print("\n" + "=" * 60)
    print("TESTS TERMINÉS")
    print("=" * 60)
