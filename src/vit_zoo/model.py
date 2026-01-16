"""Main ViT model class with flexible head and backbone."""

from typing import Optional, Union, Dict, Any
import torch
from torch import nn

from .backbone import ViTBackbone
from .heads import BaseHead, IdentityHead


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
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Forward pass through the model.
        
        Args:
            pixel_values: Input image tensor of shape (batch_size, channels, height, width)
            output_attentions: Whether to return attention weights
            output_embeddings: Whether to return embeddings
        
        Returns:
            - If output_attentions=False and output_embeddings=False: predictions tensor
            - If output_attentions=True or output_embeddings=True: dict with keys:
              'predictions', 'attentions' (optional), 'embeddings' (optional)
        """
        # Forward through backbone
        backbone_outputs = self.backbone(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=False,
        )
        
        # Extract CLS token embedding
        embeddings = self.backbone.get_cls_token_embedding(backbone_outputs)
        
        # Forward through head
        predictions = self.head(embeddings)
        
        # Return format based on requested outputs
        if output_attentions or output_embeddings:
            result: Dict[str, Any] = {"predictions": predictions}
            
            if output_attentions:
                # Check if attentions are available in backbone outputs
                if "attentions" in backbone_outputs and backbone_outputs["attentions"] is not None:
                    result["attentions"] = backbone_outputs["attentions"]
                else:
                    # If attentions were requested but not available, include None
                    # This allows callers to know attentions were requested but unavailable
                    result["attentions"] = None
            
            if output_embeddings:
                result["embeddings"] = embeddings
            
            return result
        else:
            return predictions
