"""
Lightweight backbone models for image quality assessment using torchvision models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, MobileNet_V3_Small_Weights, EfficientNet_B0_Weights


class LightweightIQAModel(nn.Module):
    """Lightweight Image Quality Assessment Model"""
    
    def __init__(self, 
                 backbone='resnet18', 
                 feature_dim=512, 
                 embedding_dim=256,
                 dropout_rate=0.1,
                 pretrained=True):
        super(LightweightIQAModel, self).__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # Initialize backbone
        if backbone == 'resnet18':
            if pretrained:
                weights = ResNet18_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.resnet18(weights=weights)
            # Remove the final classification layer
            self.backbone.fc = nn.Identity()
            backbone_dim = 512
            
        elif backbone == 'mobilenet_v3_small':
            if pretrained:
                weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            # Remove the final classification layer
            self.backbone.classifier = nn.Identity()
            backbone_dim = 576
            
        elif backbone == 'efficientnet_b0':
            if pretrained:
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.efficientnet_b0(weights=weights)
            # Remove the final classification layer
            self.backbone.classifier = nn.Identity()
            backbone_dim = 1280
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature projection layers
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Quality score head (optional direct regression)
        self.quality_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim // 2, 1)
        )
        
        # Initialize positive direction vector for similarity-based scoring
        self.register_parameter('positive_vector', 
                              nn.Parameter(torch.randn(embedding_dim), requires_grad=True))
        
    def forward(self, x, return_features=False, normalize_features=True):
        """
        Forward pass of the model
        
        Args:
            x: Input images [batch_size, 3, height, width]
            return_features: Whether to return intermediate features
            normalize_features: Whether to normalize the embedding features
            
        Returns:
            Dictionary containing embeddings, similarity scores, and optionally direct scores
        """
        # Extract features using backbone
        backbone_features = self.backbone(x)
        
        # Flatten if needed (for models like EfficientNet)
        if len(backbone_features.shape) > 2:
            backbone_features = torch.flatten(backbone_features, 1)
        
        # Project to embedding space
        embeddings = self.feature_projection(backbone_features)
        
        # Normalize embeddings if requested
        if normalize_features:
            embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        else:
            embeddings_normalized = embeddings
        
        # Compute similarity with positive vector
        positive_vector_normalized = F.normalize(self.positive_vector, p=2, dim=0)
        similarity_scores = torch.mm(embeddings_normalized, positive_vector_normalized.unsqueeze(1)).squeeze(1)
        
        # Direct quality regression (alternative approach)
        direct_scores = self.quality_head(embeddings).squeeze(1)
        
        result = {
            'embeddings': embeddings,
            'embeddings_normalized': embeddings_normalized,
            'similarity_scores': similarity_scores,
            'direct_scores': direct_scores,
            'positive_vector': positive_vector_normalized
        }
        
        if return_features:
            result['backbone_features'] = backbone_features
            
        return result
    
    def get_embedding_dim(self):
        """Return the embedding dimension"""
        return self.embedding_dim
    
    def set_positive_vector(self, vector):
        """Set the positive vector for similarity computation"""
        if len(vector.shape) == 1 and vector.size(0) == self.embedding_dim:
            self.positive_vector.data = vector.clone()
        else:
            raise ValueError(f"Vector must have shape ({self.embedding_dim},)")


class MultiBackboneIQAModel(nn.Module):
    """Model that can switch between different backbones for experiments"""
    
    def __init__(self, 
                 backbone_configs,
                 current_backbone='resnet18',
                 shared_embedding_dim=256):
        super(MultiBackboneIQAModel, self).__init__()
        
        self.backbone_configs = backbone_configs
        self.current_backbone = current_backbone
        self.shared_embedding_dim = shared_embedding_dim
        
        # Create multiple models
        self.models = nn.ModuleDict()
        for backbone_name, config in backbone_configs.items():
            self.models[backbone_name] = LightweightIQAModel(
                backbone=backbone_name,
                embedding_dim=shared_embedding_dim,
                **config
            )
    
    def forward(self, x, backbone=None, **kwargs):
        """Forward pass using specified or current backbone"""
        if backbone is None:
            backbone = self.current_backbone
        
        if backbone not in self.models:
            raise ValueError(f"Backbone {backbone} not available")
        
        return self.models[backbone](x, **kwargs)
    
    def switch_backbone(self, backbone_name):
        """Switch to a different backbone"""
        if backbone_name in self.models:
            self.current_backbone = backbone_name
        else:
            raise ValueError(f"Backbone {backbone_name} not available")
    
    def get_available_backbones(self):
        """Get list of available backbones"""
        return list(self.models.keys())


def create_lightweight_model(backbone='resnet18', **kwargs):
    """Factory function to create lightweight IQA models"""
    return LightweightIQAModel(backbone=backbone, **kwargs)


def create_multi_backbone_model(backbones=['resnet18', 'mobilenet_v3_small'], **kwargs):
    """Factory function to create multi-backbone IQA models"""
    backbone_configs = {}
    for backbone in backbones:
        backbone_configs[backbone] = kwargs.copy()
    
    return MultiBackboneIQAModel(
        backbone_configs=backbone_configs,
        current_backbone=backbones[0],
        **kwargs
    )