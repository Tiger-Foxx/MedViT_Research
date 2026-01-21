# MedViT-CAMIL Docker Image V2
# Support OpenCV pour vraies vidéos
FROM python:3.10-slim

# Métadonnées
LABEL maintainer="MedViT Research Team"
LABEL description="MedViT-CAMIL V2: 3 modes (TEST/PROXY/REAL)"
LABEL version="2.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Répertoire de travail
WORKDIR /app

# Dépendances système pour OpenCV et téléchargement
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements et installer
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY run.sh .

# Rendre le script exécutable
RUN chmod +x run.sh

# Créer les répertoires
RUN mkdir -p /app/data /app/results

# Volumes
VOLUME ["/app/data", "/app/results"]

# Point d'entrée
ENTRYPOINT ["./run.sh"]

# Mode par défaut: test
CMD ["test"]

