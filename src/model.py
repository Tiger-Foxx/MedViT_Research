"""
MedViT-CAMIL Model Module
=========================
Contient les architectures:
1. MobileViTBackbone: Extracteur de features (gelé)
2. BaselineAvgPooling: Baseline avec moyenne temporelle
3. ContextAwareGatedMIL: Module CAMIL (contribution)
4. MedViTCAMIL: Modèle complet

Architecture CAMIL (Context-Aware Multiple Instance Learning):
- Conv1D (kernel=3): Vérifie la cohérence locale [t-1, t, t+1]
- Gated Attention (Ilse et al. ICML 2018): 
  * Branche V (contenu): Tanh
  * Branche U (gate): Sigmoid
  * Score = w * (Tanh(V) × Sigmoid(U))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import warnings


# ============================================================================
# MOBILEVIT BACKBONE (FROZEN)
# ============================================================================

class MobileViTBackbone(nn.Module):
    """
    Backbone MobileViT pré-entraîné sur ImageNet.
    Extrait des features visuelles pour chaque frame individuellement.
    
    Les poids sont GELÉS (frozen) - ce n'est pas la contribution du projet.
    """
    
    def __init__(
        self,
        model_name: str = "mobilevit_s",
        pretrained: bool = True,
        feature_dim: int = 512
    ):
        """
        Args:
            model_name: Nom du modèle MobileViT ('mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs')
            pretrained: Charger les poids pré-entraînés ImageNet
            feature_dim: Dimension de sortie des features
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Charger MobileViT via timm
        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm n'est pas installé. "
                "Installez-le avec: pip install timm"
            )
        
        print(f"[INFO] Chargement de {model_name} (pretrained={pretrained})...")
        
        # Créer le modèle sans la tête de classification
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Supprime la couche de classification
            global_pool='avg'  # Global average pooling
        )
        
        # Obtenir la dimension de sortie réelle
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.backbone(dummy)
            self.actual_feature_dim = out.shape[-1]
        
        print(f"[INFO] Feature dimension: {self.actual_feature_dim}")
        
        # GELER tous les poids du backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.eval()
        print("[INFO] Backbone MobileViT gelé (frozen)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (B, T, C, H, W) - Batch de séquences vidéo
               ou (B, C, H, W) - Batch d'images simples
        
        Returns:
            features: Tensor (B, T, D) ou (B, D) - Features extraites
        """
        # Gérer les deux cas: vidéo (5D) ou images (4D)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            # Reshape pour traiter toutes les frames en batch
            x = x.view(B * T, C, H, W)  # (B*T, C, H, W)
            
            # Extraire les features
            with torch.no_grad():
                features = self.backbone(x)  # (B*T, D)
            
            # Reshape pour retrouver la structure temporelle
            features = features.view(B, T, -1)  # (B, T, D)
        else:
            # Cas image simple
            with torch.no_grad():
                features = self.backbone(x)  # (B, D)
        
        return features
    
    def train(self, mode: bool = True):
        """Override pour garder le backbone en mode eval."""
        super().train(mode)
        self.backbone.eval()  # Toujours en eval
        return self


# ============================================================================
# BASELINE: AVERAGE POOLING
# ============================================================================

class BaselineAvgPooling(nn.Module):
    """
    Baseline simple: moyenne temporelle des features.
    
    Problème: dilue le signal si l'anomalie n'apparaît que sur quelques frames.
    Exemple: 5 frames anomales sur 200 → signal dilué par 40x
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Args:
            feature_dim: Dimension des features d'entrée (depuis MobileViT)
            hidden_dim: Dimension cachée
            num_classes: Nombre de classes de sortie
            dropout: Taux de dropout
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Projection optionnelle
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Tête de classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self, 
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Tensor (B, T, D) - Features temporelles
        
        Returns:
            logits: Tensor (B, num_classes) - Logits de classification
            attention_weights: Tensor (B, T) - Poids uniformes (1/T)
        """
        B, T, D = features.shape
        
        # Projection
        projected = self.projection(features)  # (B, T, hidden_dim)
        
        # Moyenne temporelle simple
        aggregated = projected.mean(dim=1)  # (B, hidden_dim)
        
        # Classification
        logits = self.classifier(aggregated)  # (B, num_classes)
        
        # Poids d'attention uniformes (pour visualisation)
        attention_weights = torch.ones(B, T, device=features.device) / T
        
        return logits, attention_weights


# ============================================================================
# CONTRIBUTION: CONTEXT-AWARE GATED ATTENTION MIL
# ============================================================================

class ContextAwareGatedMIL(nn.Module):
    """
    Module CAMIL (Context-Aware Multiple Instance Learning).
    
    Architecture:
    1. Conv1D (kernel=3): Capture le contexte local [t-1, t, t+1]
       - Une anomalie isolée sur une seule frame pourrait être du bruit
       - Une anomalie cohérente sur 2-3 frames est plus crédible
    
    2. Gated Attention (Ilse et al. ICML 2018):
       - Branche V (contenu): Tanh → encode "quoi"
       - Branche U (gate): Sigmoid → encode "pertinence"
       - Score = w^T * (Tanh(V*h) ⊙ Sigmoid(U*h))
       - Le gate peut "fermer" complètement l'attention (→ rejeter le bruit)
    
    Avantage vs Softmax standard:
    - Softmax force Σ(attention) = 1, donc distribue toujours de l'attention
    - Gated attention peut donner ~0 à toutes les frames si aucune n'est pertinente
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        context_kernel: int = 3,
        dropout: float = 0.3
    ):
        """
        Args:
            feature_dim: Dimension des features d'entrée
            hidden_dim: Dimension cachée
            num_classes: Nombre de classes
            context_kernel: Taille du kernel Conv1D pour le contexte local
            dropout: Taux de dropout
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 1. Projection initiale
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 2. Contexte temporel local (Conv1D)
        # Padding 'same' pour garder la même longueur de séquence
        self.context_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=context_kernel,
                padding=context_kernel // 2,  # 'same' padding
                bias=False
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # 3. Gated Attention
        # Branche V (contenu) avec Tanh
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Branche U (gate) avec Sigmoid
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Vecteur de score
        self.attention_w = nn.Linear(hidden_dim, 1)
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 5. Tête de classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self, 
        features: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Tensor (B, T, D) - Features temporelles depuis MobileViT
            return_attention: Retourner les poids d'attention
        
        Returns:
            logits: Tensor (B, num_classes) - Logits de classification
            attention_weights: Tensor (B, T) - Poids d'attention normalisés
        """
        B, T, D = features.shape
        
        # 1. Projection initiale
        h = self.input_projection(features)  # (B, T, hidden_dim)
        
        # 2. Contexte temporel via Conv1D
        # Conv1D attend (B, C, T), donc on permute
        h_conv = h.permute(0, 2, 1)  # (B, hidden_dim, T)
        h_context = self.context_conv(h_conv)  # (B, hidden_dim, T)
        h_context = h_context.permute(0, 2, 1)  # (B, T, hidden_dim)
        
        # Résiduel + contexte
        h = h + h_context  # (B, T, hidden_dim)
        h = self.dropout(h)
        
        # 3. Gated Attention
        # Branche contenu (Tanh)
        v = self.attention_V(h)  # (B, T, hidden_dim)
        
        # Branche gate (Sigmoid)
        u = self.attention_U(h)  # (B, T, hidden_dim)
        
        # Combinaison: élément par élément
        gated = v * u  # (B, T, hidden_dim) - Le gate peut "fermer" le signal
        
        # Score d'attention
        attention_scores = self.attention_w(gated).squeeze(-1)  # (B, T)
        
        # Normalisation Softmax pour obtenir des poids
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, T)
        
        # 4. Agrégation pondérée
        # attention_weights: (B, T) → (B, T, 1)
        # h: (B, T, hidden_dim)
        aggregated = torch.bmm(
            attention_weights.unsqueeze(1),  # (B, 1, T)
            h  # (B, T, hidden_dim)
        ).squeeze(1)  # (B, hidden_dim)
        
        # 5. Classification
        logits = self.classifier(aggregated)  # (B, num_classes)
        
        return logits, attention_weights


# ============================================================================
# MODÈLE COMPLET: MedViT-CAMIL
# ============================================================================

class MedViTCAMIL(nn.Module):
    """
    Modèle complet MedViT-CAMIL.
    
    Architecture:
    1. MobileViT Backbone (gelé) → Extraction de features par frame
    2. Module CAMIL → Agrégation temporelle intelligente
    3. Tête de classification → Prédiction finale
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 128,
        num_classes: int = 2,
        backbone_name: str = "mobilevit_s",
        pretrained: bool = True,
        use_camil: bool = True,  # True=CAMIL, False=Baseline
        dropout: float = 0.3
    ):
        """
        Args:
            feature_dim: Dimension des features (doit matcher MobileViT)
            hidden_dim: Dimension cachée
            num_classes: Nombre de classes
            backbone_name: Nom du modèle MobileViT
            pretrained: Utiliser les poids pré-entraînés
            use_camil: Utiliser CAMIL (True) ou Baseline (False)
            dropout: Taux de dropout
        """
        super().__init__()
        
        self.use_camil = use_camil
        
        # Backbone MobileViT (gelé)
        self.backbone = MobileViTBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            feature_dim=feature_dim
        )
        
        # Dimension réelle des features du backbone
        actual_dim = self.backbone.actual_feature_dim
        
        # Module d'agrégation temporelle
        if use_camil:
            self.aggregator = ContextAwareGatedMIL(
                feature_dim=actual_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout
            )
            self.model_name = "MedViT-CAMIL"
        else:
            self.aggregator = BaselineAvgPooling(
                feature_dim=actual_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout
            )
            self.model_name = "MedViT-Baseline"
        
        print(f"[INFO] Modèle créé: {self.model_name}")
    
    def forward(
        self, 
        video: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video: Tensor (B, T, C, H, W) - Batch de vidéos
        
        Returns:
            logits: Tensor (B, num_classes) - Logits de classification
            attention_weights: Tensor (B, T) - Poids d'attention temporelle
        """
        # Extraire les features pour chaque frame
        features = self.backbone(video)  # (B, T, D)
        
        # Agréger et classifier
        logits, attention_weights = self.aggregator(features)
        
        return logits, attention_weights
    
    def get_attention_map(self, video: torch.Tensor) -> torch.Tensor:
        """
        Obtient la carte d'attention temporelle pour l'explicabilité.
        
        Args:
            video: Tensor (B, T, C, H, W)
        
        Returns:
            attention: Tensor (B, T) - Poids d'attention par frame
        """
        self.eval()
        with torch.no_grad():
            _, attention = self.forward(video)
        return attention


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_model(
    model_type: str = "camil",
    feature_dim: int = 512,
    hidden_dim: int = 128,
    num_classes: int = 2,
    pretrained: bool = True,
    device: torch.device = None
) -> nn.Module:
    """
    Factory function pour créer les modèles.
    
    Args:
        model_type: 'camil' ou 'baseline'
        feature_dim: Dimension des features
        hidden_dim: Dimension cachée
        num_classes: Nombre de classes
        pretrained: Utiliser backbone pré-entraîné
        device: Device cible (CPU/GPU)
    
    Returns:
        model: Modèle initialisé et sur le bon device
    """
    use_camil = (model_type.lower() == "camil")
    
    model = MedViTCAMIL(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        pretrained=pretrained,
        use_camil=use_camil
    )
    
    if device is not None:
        model = model.to(device)
    
    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[INFO] Paramètres totaux: {total_params:,}")
    print(f"[INFO] Paramètres entraînables: {trainable_params:,}")
    print(f"[INFO] Backbone gelé: {total_params - trainable_params:,} paramètres")
    
    return model


# ============================================================================
# TEST MODULE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DU MODULE MODEL")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")
    
    # Créer un batch de test
    B, T, C, H, W = 2, 16, 3, 224, 224
    dummy_video = torch.randn(B, T, C, H, W).to(device)
    print(f"[INFO] Input shape: {dummy_video.shape}")
    
    # Test Baseline
    print("\n" + "-" * 40)
    print("TEST: MedViT-Baseline")
    print("-" * 40)
    model_baseline = create_model(
        model_type="baseline",
        device=device
    )
    logits, attention = model_baseline(dummy_video)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Attention shape: {attention.shape}")
    print(f"  Attention sum: {attention.sum(dim=1)}")  # Doit être ~1.0
    
    # Test CAMIL
    print("\n" + "-" * 40)
    print("TEST: MedViT-CAMIL")
    print("-" * 40)
    model_camil = create_model(
        model_type="camil",
        device=device
    )
    logits, attention = model_camil(dummy_video)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Attention shape: {attention.shape}")
    print(f"  Attention sum: {attention.sum(dim=1)}")
    print(f"  Attention distribution: min={attention.min():.4f}, max={attention.max():.4f}")
    
    print("\n" + "=" * 60)
    print("TESTS TERMINÉS")
    print("=" * 60)
