"""Main ViT model class with flexible head and backbone."""

from typing import Optional, Union, Dict, Any
import torch
from torch import nn

from .components.backbone import ViTBackbone
from .components.heads import BaseHead, IdentityHead


class ViTModel(nn.Module):
    """Main Vision Transformer model with flexible head and backbone.
    
    This class provides a clean, extensible interface for ViT models with:
    - Custom heads (Linear, MLP, or custom implementations)
    - Backbone freezing
    - Attention weight extraction
    - Embedding extraction
    
    Args:
        backbone: ViTBackbone instance
        head: Optional head module. If None, IdentityHead is used (embedding extraction mode)
        freeze_backbone: If True, freeze all backbone parameters at initialization
    """
    
    def __init__(
        self,
        backbone: ViTBackbone,
        head: Optional[BaseHead] = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        
        self.backbone = backbone
        
        # Use IdentityHead if no head provided (embedding extraction mode)
        if head is None:
            self.head = IdentityHead(input_dim=self.backbone.get_embedding_dim())
        else:
            self.head = head
        
        # Apply freezing if requested
        if freeze_backbone:
            self.freeze_backbone(freeze=True)
    
    @property
    def embedding_dim(self) -> int:
        """Returns the embedding dimension of the backbone."""
        return self.backbone.get_embedding_dim()
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze all backbone parameters.
        
        Args:
            freeze: If True, freeze parameters; if False, unfreeze them
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_attentions: bool = False,
        output_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through the model.
        
        Args:
            pixel_values: Input image tensor of shape (batch_size, channels, height, width)
            output_attentions: Whether to return attention weights
            output_embeddings: Whether to return token embeddings for all output tokens
        
        Returns:
            A dict with at least the key 'predictions' (head output tensor). Optional keys:
            - 'attentions': when output_attentions=True; attention weights from the backbone
            - 'last_hidden_state': when output_embeddings=True; tensor of shape
              (batch_size, seq_len, embedding_dim) containing embeddings for all output
              tokens (CLS + any special tokens like distillation/register tokens + patch tokens).
        """
        # Forward through backbone
        backbone_outputs = self.backbone(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=False,
        )
        
        # Extract CLS token embedding (used for predictions)
        cls_embedding = self.backbone.get_cls_token_embedding(backbone_outputs)
        
        # Forward through head
        predictions = self.head(cls_embedding)
        
        # Pack forward results and return
        results = {"predictions": predictions}

        if output_attentions:
            results["attentions"] = backbone_outputs["attentions"]
        
        if output_embeddings:
            results["last_hidden_state"] = backbone_outputs["last_hidden_state"]
        
        return results
