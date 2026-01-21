"""
MedViT-CAMIL Dataset Module V2
==============================
Moteur de données hybride supportant 3 modes:
- TEST: Données synthétiques avec bruit speckle
- PROXY: NoduleMNIST3D (volumes 3D → pseudo-vidéo)
- REAL: HyperKvasir images + vraies vidéos OpenCV

Format de sortie unifié: (B, T, C, H, W)
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from typing import Tuple, Optional, List
import warnings

# Import conditionnel pour OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ============================================================================
# NORMALISATION STANDARD (ImageNet pour MobileViT)
# ============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# MODE TEST: DONNÉES SYNTHÉTIQUES
# ============================================================================

class SyntheticDataset(Dataset):
    """
    Dataset synthétique simulant des vidéos médicales avec:
    - Bruit speckle (typique de l'échographie)
    - Lésions mouvantes apparaissant sur quelques frames
    
    Classe 0: Normal (bruit seulement)
    Classe 1: Anomalie (lésion sur quelques frames)
    """
    
    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        img_size: int = 224,
        anomaly_frames: int = 3,
        seed: Optional[int] = None
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.img_size = img_size
        self.anomaly_frames = min(anomaly_frames, seq_len)
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 50% normal, 50% anomalie
        self.labels = torch.zeros(num_samples, dtype=torch.long)
        self.labels[num_samples // 2:] = 1
        
        # Positions des anomalies
        self.anomaly_positions = []
        for i in range(num_samples):
            if self.labels[i] == 1:
                start = np.random.randint(0, self.seq_len - self.anomaly_frames + 1)
                positions = list(range(start, start + self.anomaly_frames))
            else:
                positions = []
            self.anomaly_positions.append(positions)
    
    def _generate_speckle_noise(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Génère du bruit speckle (distribution gamma)."""
        noise = np.random.gamma(shape=2.0, scale=0.5, size=shape)
        return np.clip(noise, 0, 1).astype(np.float32)
    
    def _generate_lesion(self, img_size: int) -> np.ndarray:
        """Génère une lésion elliptique brillante."""
        lesion = np.zeros((img_size, img_size), dtype=np.float32)
        cx = np.random.randint(img_size // 4, 3 * img_size // 4)
        cy = np.random.randint(img_size // 4, 3 * img_size // 4)
        rx = np.random.randint(10, 30)
        ry = np.random.randint(10, 30)
        
        y, x = np.ogrid[:img_size, :img_size]
        mask = ((x - cx) ** 2 / (rx ** 2 + 1e-6) + (y - cy) ** 2 / (ry ** 2 + 1e-6)) <= 1
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        intensity = np.exp(-dist / (max(rx, ry) * 0.5))
        lesion[mask] = intensity[mask] * 0.8
        
        return lesion
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label = self.labels[idx]
        anomaly_pos = self.anomaly_positions[idx]
        
        frames = []
        for t in range(self.seq_len):
            frame = self._generate_speckle_noise((self.img_size, self.img_size))
            
            # Structure de fond
            background = np.sin(np.linspace(0, 4 * np.pi, self.img_size)).reshape(1, -1)
            background = background * np.sin(np.linspace(0, 4 * np.pi, self.img_size)).reshape(-1, 1)
            background = (background + 1) / 4
            frame = frame * 0.5 + background * 0.5
            
            if t in anomaly_pos:
                lesion = self._generate_lesion(self.img_size)
                frame = np.clip(frame + lesion, 0, 1)
            
            frame_rgb = np.stack([frame, frame, frame], axis=0)
            frames.append(frame_rgb)
        
        video = np.stack(frames, axis=0)
        video = torch.from_numpy(video).float()
        
        # Normalisation ImageNet
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        video = (video - mean) / std
        
        return video, label


# ============================================================================
# MODE PROXY: NODULEMNIST3D
# ============================================================================

class ProxyDataset(Dataset):
    """
    Dataset wrapper pour NoduleMNIST3D.
    Traite les volumes 3D (D, H, W) comme des pseudo-vidéos (T, H, W).
    """
    
    def __init__(
        self,
        split: str = "train",
        data_dir: str = "./data",
        img_size: int = 224,
        download: bool = True
    ):
        self.img_size = img_size
        
        try:
            import medmnist
            from medmnist import NoduleMNIST3D
        except ImportError:
            raise ImportError("medmnist non installé. pip install medmnist")
        
        print(f"[INFO] Chargement NoduleMNIST3D (split={split})...")
        self.dataset = NoduleMNIST3D(
            split=split,
            root=data_dir,
            download=download
        )
        print(f"[INFO] {len(self.dataset)} échantillons chargés")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        volume = self.dataset.imgs[idx]  # (28, 28, 28)
        label = int(self.dataset.labels[idx].item())
        
        # Normaliser à [0, 1]
        volume = torch.from_numpy(volume).float()
        if volume.max() > 1.0:
            volume = volume / 255.0
        
        seq_len = volume.shape[0]
        frames = []
        
        for t in range(seq_len):
            frame = volume[t].unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
            frame = F.interpolate(frame, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            frame = frame.squeeze(0).repeat(3, 1, 1)  # (3, 224, 224)
            frames.append(frame)
        
        video = torch.stack(frames, dim=0)  # (T, 3, 224, 224)
        
        # Normalisation ImageNet
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        video = (video - mean) / std
        
        return video, torch.tensor(label, dtype=torch.long)


# ============================================================================
# MODE REAL: HYPERKVASIR + VIDÉOS OPENCV
# ============================================================================

class RealVideoDataset(Dataset):
    """
    Dataset polymorphe pour données réelles:
    - Si fichier .mp4/.avi/.mov → extraction frames via OpenCV
    - Si image .jpg/.png → duplication pour créer une pseudo-séquence
    
    Structure attendue:
    data_dir/
    ├── normal/
    │   ├── video1.mp4 ou image1.jpg
    │   └── ...
    └── abnormal/
        ├── video2.mp4 ou image2.jpg
        └── ...
    """
    
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        seq_len: int = 32,
        img_size: int = 224,
    ):
        self.seq_len = seq_len
        self.img_size = img_size
        self.data_dir = data_dir
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        # Scanner les fichiers
        self.samples = []
        classes = ['normal', 'abnormal']
        
        for label_idx, cls_name in enumerate(classes):
            cls_path = os.path.join(data_dir, cls_name)
            if os.path.exists(cls_path):
                files = glob.glob(os.path.join(cls_path, "*"))
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in self.VIDEO_EXTENSIONS or ext in self.IMAGE_EXTENSIONS:
                        self.samples.append((f, label_idx))
        
        # Split 80/20
        np.random.seed(42)
        np.random.shuffle(self.samples)
        cut = int(0.8 * len(self.samples))
        
        if split == 'train':
            self.samples = self.samples[:cut]
        else:
            self.samples = self.samples[cut:]
        
        print(f"[{split.upper()}] {len(self.samples)} fichiers réels chargés depuis {data_dir}")
    
    def __len__(self) -> int:
        return max(len(self.samples), 1)  # Au moins 1 pour éviter les erreurs
    
    def _load_video_opencv(self, path: str) -> List[torch.Tensor]:
        """Charge une vraie vidéo avec OpenCV."""
        if not CV2_AVAILABLE:
            raise ImportError("opencv-python non installé. pip install opencv-python-headless")
        
        frames = []
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 0:
            # Échantillonner uniformément
            indices = np.linspace(0, total_frames - 1, self.seq_len, dtype=int)
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in indices:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(self.transform(frame))
                frame_idx += 1
        
        cap.release()
        return frames
    
    def _load_image(self, path: str) -> List[torch.Tensor]:
        """Charge une image et la duplique pour créer une séquence."""
        if not CV2_AVAILABLE:
            raise ImportError("opencv-python non installé. pip install opencv-python-headless")
        
        img = cv2.imread(path)
        if img is None:
            return []
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img)
        
        # Dupliquer pour remplir la séquence
        return [tensor for _ in range(self.seq_len)]
    
    def _load_content(self, path: str) -> torch.Tensor:
        """Charge le contenu (vidéo ou image) et retourne un tensor."""
        ext = os.path.splitext(path)[1].lower()
        
        if ext in self.VIDEO_EXTENSIONS:
            frames = self._load_video_opencv(path)
        else:
            frames = self._load_image(path)
        
        # Gestion des erreurs
        if len(frames) == 0:
            return torch.zeros(self.seq_len, 3, self.img_size, self.img_size)
        
        video = torch.stack(frames)
        
        # Padding si nécessaire
        if video.shape[0] < self.seq_len:
            pad = video[-1].unsqueeze(0).repeat(self.seq_len - video.shape[0], 1, 1, 1)
            video = torch.cat([video, pad], dim=0)
        
        return video[:self.seq_len]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.samples) == 0:
            # Dataset vide - retourner un placeholder
            return torch.zeros(self.seq_len, 3, self.img_size, self.img_size), torch.tensor(0, dtype=torch.long)
        
        path, label = self.samples[idx]
        
        try:
            video = self._load_content(path)
            return video, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"[WARNING] Erreur chargement {path}: {e}")
            return torch.zeros(self.seq_len, 3, self.img_size, self.img_size), torch.tensor(label, dtype=torch.long)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_loaders(config) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les DataLoaders selon le mode.
    
    Returns:
        train_loader, val_loader
    """
    print(f"\n[INFO] Création des DataLoaders (mode={config.mode})...")
    
    if config.mode == "test":
        train_dataset = SyntheticDataset(
            num_samples=config.num_train_samples,
            seq_len=config.seq_len,
            img_size=config.img_size,
            seed=config.seed
        )
        val_dataset = SyntheticDataset(
            num_samples=config.num_val_samples,
            seq_len=config.seq_len,
            img_size=config.img_size,
            seed=config.seed + 1
        )
        
    elif config.mode == "proxy":
        train_dataset = ProxyDataset(
            split="train",
            data_dir=config.data_dir,
            img_size=config.img_size
        )
        val_dataset = ProxyDataset(
            split="val",
            data_dir=config.data_dir,
            img_size=config.img_size
        )
        
    else:  # mode == "real"
        # Vérifier OpenCV
        if not CV2_AVAILABLE:
            raise ImportError(
                "opencv-python non installé pour le mode REAL.\n"
                "Installez-le avec: pip install opencv-python-headless"
            )
        
        # Télécharger HyperKvasir VIDÉOS si nécessaire (25 GB)
        # Le code est polymorphe: il gère vidéos .mp4 ET images .jpg
        try:
            from data_utils import prepare_hyperkvasir
        except ImportError:
            from src.data_utils import prepare_hyperkvasir
        
        # use_videos=True pour télécharger les vraies vidéos (25 GB)
        # Le serveur du prof aura les ressources
        real_data_dir = prepare_hyperkvasir(config.data_dir, use_videos=True)
        
        train_dataset = RealVideoDataset(
            data_dir=real_data_dir,
            split="train",
            seq_len=config.seq_len,
            img_size=config.img_size
        )
        val_dataset = RealVideoDataset(
            data_dir=real_data_dir,
            split="val",
            seq_len=config.seq_len,
            img_size=config.img_size
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"[INFO] Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"[INFO] Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


# ============================================================================
# TEST MODULE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU MODULE DATASET V2")
    print("=" * 60)
    
    # Test synthétique
    print("\n[TEST] SyntheticDataset:")
    ds = SyntheticDataset(num_samples=10, seq_len=16, img_size=224, seed=42)
    video, label = ds[0]
    print(f"  Shape: {video.shape}, Label: {label}")
    
    # Test proxy (si medmnist disponible)
    print("\n[TEST] ProxyDataset:")
    try:
        ds = ProxyDataset(split="train", data_dir="./data", img_size=224)
        video, label = ds[0]
        print(f"  Shape: {video.shape}, Label: {label}")
    except Exception as e:
        print(f"  [SKIP] {e}")
    
    print("\n" + "=" * 60)
