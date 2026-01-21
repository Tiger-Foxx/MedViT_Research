# MedViT-CAMIL Docker Image
# Base image avec PyTorch et CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Métadonnées
LABEL maintainer="MedViT Research Team"
LABEL description="MedViT-CAMIL: Context-Aware Multiple Instance Learning for Medical Video Analysis"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Répertoire de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements et installer les dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY run.sh .

# Rendre le script exécutable
RUN chmod +x run.sh

# Créer les répertoires pour les données et résultats
RUN mkdir -p /app/data /app/results

# Volume pour persister les données et résultats
VOLUME ["/app/data", "/app/results"]

# Point d'entrée par défaut
ENTRYPOINT ["./run.sh"]

# Mode par défaut: test
CMD ["test"]
