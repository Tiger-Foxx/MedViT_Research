"""
MedViT-CAMIL Data Utilities
===========================
Utilitaires pour le téléchargement et la préparation des datasets:
- HyperKvasir (mode real)
- NoduleMNIST3D (mode proxy - géré par medmnist)
"""

import os
import zipfile
import shutil
from pathlib import Path
from typing import Optional
import urllib.request
import ssl


# ============================================================================
# HYPERKVASIR DATASET
# ============================================================================

# Liens officiels: https://datasets.simula.no/hyper-kvasir/
# VIDÉOS labellisées (25.2 GB) - Le vrai dataset pour notre projet vidéo
HYPERKVASIR_VIDEOS_URL = "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-videos.zip"
HYPERKVASIR_VIDEOS_SIZE_GB = 25.2

# Images labellisées (3.9 GB) - Fallback si pas assez d'espace
HYPERKVASIR_IMAGES_URL = "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip"
HYPERKVASIR_IMAGES_SIZE_GB = 3.9


def download_with_progress(url: str, dest_path: str, desc: str = "Téléchargement") -> None:
    """Télécharge un fichier avec barre de progression."""
    
    # Contourner les problèmes SSL sur certains systèmes
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    print(f"[INFO] {desc}...")
    print(f"[INFO] URL: {url}")
    print(f"[INFO] Destination: {dest_path}")
    
    try:
        # Essayer avec tqdm si disponible
        from tqdm import tqdm
        
        with urllib.request.urlopen(url, context=ssl_context) as response:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    except ImportError:
        # Sans tqdm
        urllib.request.urlretrieve(url, dest_path)
    
    print(f"[INFO] Téléchargement terminé: {dest_path}")


def prepare_hyperkvasir(data_dir: str, force_download: bool = False, use_videos: bool = True) -> str:
    """
    Télécharge et prépare le dataset HyperKvasir pour l'entraînement.
    
    Args:
        data_dir: Répertoire de destination
        force_download: Re-télécharger même si existe
        use_videos: True = vidéos (25 GB), False = images (3.9 GB) comme fallback
    
    Structure créée:
    data_dir/hyperkvasir/
    ├── normal/      # Vidéos/Images normales (classe 0)
    └── abnormal/    # Vidéos/Images avec anomalies (classe 1)
    
    Returns:
        Chemin vers le répertoire préparé
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    hyperkvasir_dir = data_dir / "hyperkvasir"
    normal_dir = hyperkvasir_dir / "normal"
    abnormal_dir = hyperkvasir_dir / "abnormal"
    
    # Extensions selon le type de données
    if use_videos:
        extensions = ["*.mp4", "*.avi", "*.mov"]
        url = HYPERKVASIR_VIDEOS_URL
        size_gb = HYPERKVASIR_VIDEOS_SIZE_GB
        zip_name = "hyper-kvasir-labeled-videos.zip"
        data_type = "VIDÉOS"
    else:
        extensions = ["*.jpg", "*.png"]
        url = HYPERKVASIR_IMAGES_URL
        size_gb = HYPERKVASIR_IMAGES_SIZE_GB
        zip_name = "hyper-kvasir-labeled-images.zip"
        data_type = "IMAGES"
    
    # Vérifier si déjà préparé
    if normal_dir.exists() and abnormal_dir.exists() and not force_download:
        file_count = 0
        for ext in extensions:
            file_count += len(list(normal_dir.glob(ext)))
            file_count += len(list(abnormal_dir.glob(ext)))
        
        if file_count > 0:
            print(f"[INFO] HyperKvasir {data_type} déjà préparé: {file_count} fichiers trouvés")
            return str(hyperkvasir_dir)
    
    # Télécharger si nécessaire
    zip_path = data_dir / zip_name
    
    if not zip_path.exists() or force_download:
        print(f"[INFO] Téléchargement de HyperKvasir {data_type} (~{size_gb} GB)...")
        print("[INFO] Cela peut prendre plusieurs minutes selon votre connexion...")
        print(f"[INFO] URL: {url}")
        download_with_progress(url, str(zip_path), f"HyperKvasir {data_type}")
    
    # Extraire
    print("[INFO] Extraction de l'archive...")
    extract_dir = data_dir / "hyper-kvasir-extracted"
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Organiser en normal/abnormal
    print("[INFO] Organisation des données...")
    normal_dir.mkdir(parents=True, exist_ok=True)
    abnormal_dir.mkdir(parents=True, exist_ok=True)
    
    # Classes considérées comme normales (muqueuse saine)
    NORMAL_CLASSES = [
        "normal-cecum",
        "normal-pylorus",
        "normal-z-line",
        "bbps-2-3",  # Bien préparé = normal
    ]
    
    # Classes considérées comme anormales (pathologies)
    ABNORMAL_CLASSES = [
        "polyps",
        "ulcerative-colitis",
        "esophagitis",
        "dyed-lifted-polyps",
        "dyed-resection-margins",
    ]
    
    # Scanner et copier (structure différente pour vidéos vs images)
    # Pour les vidéos: labeled-videos/lower-gi-tract/...
    # Pour les images: labeled-images/lower-gi-tract/...
    
    if use_videos:
        labeled_dir = extract_dir / "labeled-videos"
        file_patterns = ["*.mp4", "*.avi", "*.mov"]
    else:
        labeled_dir = extract_dir / "labeled-images"
        file_patterns = ["*.jpg", "*.png"]
    
    # Chercher dans les sous-dossiers si nécessaire
    if not labeled_dir.exists():
        for subdir in extract_dir.iterdir():
            if subdir.is_dir():
                test_dir = subdir / ("labeled-videos" if use_videos else "labeled-images")
                if test_dir.exists():
                    labeled_dir = test_dir
                    break
    
    if not labeled_dir.exists():
        # Dernier recours: scanner tout le dossier extrait
        labeled_dir = extract_dir
        print(f"[WARNING] Structure non standard, scan de: {labeled_dir}")
    
    copied_normal = 0
    copied_abnormal = 0
    
    # Fonction récursive pour scanner tous les sous-dossiers
    def scan_and_copy(search_dir: Path):
        nonlocal copied_normal, copied_abnormal
        
        for item in search_dir.iterdir():
            if item.is_dir():
                dir_name = item.name.lower()
                
                # Déterminer si normal ou abnormal par le nom du dossier
                is_normal = any(n in dir_name for n in NORMAL_CLASSES)
                is_abnormal = any(a in dir_name for a in ABNORMAL_CLASSES)
                
                if is_normal:
                    dest = normal_dir
                elif is_abnormal:
                    dest = abnormal_dir
                else:
                    # Continuer à descendre dans l'arborescence
                    scan_and_copy(item)
                    continue
                
                # Copier tous les fichiers correspondants
                for pattern in file_patterns:
                    for file in item.glob(pattern):
                        dest_file = dest / f"{dir_name}_{file.name}"
                        if not dest_file.exists():
                            shutil.copy2(file, dest_file)
                            if dest == normal_dir:
                                copied_normal += 1
                            else:
                                copied_abnormal += 1
    
    scan_and_copy(labeled_dir)
    
    print(f"[INFO] Copié: {copied_normal} fichiers normaux, {copied_abnormal} fichiers anormaux")
    
    # Nettoyage optionnel
    # shutil.rmtree(extract_dir)
    # os.remove(zip_path)
    
    return str(hyperkvasir_dir)


def check_opencv() -> bool:
    """Vérifie si OpenCV est installé."""
    try:
        import cv2
        return True
    except ImportError:
        return False


def check_medmnist() -> bool:
    """Vérifie si medmnist est installé."""
    try:
        import medmnist
        return True
    except ImportError:
        return False


def setup_data_for_mode(mode: str, data_dir: str) -> str:
    """
    Configure les données pour le mode spécifié.
    
    Returns:
        Chemin vers les données préparées
    """
    if mode == "test":
        # Pas de données externes nécessaires
        return data_dir
    
    elif mode == "proxy":
        # MedMNIST se télécharge automatiquement via le dataset
        if not check_medmnist():
            raise ImportError(
                "medmnist n'est pas installé.\n"
                "Installez-le avec: pip install medmnist"
            )
        return data_dir
    
    elif mode == "real":
        # Vérifier OpenCV
        if not check_opencv():
            raise ImportError(
                "opencv-python n'est pas installé.\n"
                "Installez-le avec: pip install opencv-python-headless"
            )
        
        # Préparer HyperKvasir
        return prepare_hyperkvasir(data_dir)
    
    return data_dir


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DES UTILITAIRES DE DONNÉES")
    print("=" * 60)
    
    print("\n[CHECK] OpenCV:", "✓" if check_opencv() else "✗")
    print("[CHECK] MedMNIST:", "✓" if check_medmnist() else "✗")
    
    print("\n[INFO] Pour télécharger HyperKvasir:")
    print("  python -c \"from data_utils import prepare_hyperkvasir; prepare_hyperkvasir('./data')\"")
