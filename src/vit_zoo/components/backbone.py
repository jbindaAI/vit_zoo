"""Backbone abstraction layer for ViT models."""

from typing import Type, Dict, Any, Optional
from torch import nn
import torch

from ..types.interfaces import ViTBackboneProtocol


class ViTBackbone(nn.Module):
    """Wraps HuggingFace ViT models and normalizes their outputs.
    
    This class provides a unified interface for different HuggingFace
    ViT model implementations, handling differences in output formats
    and initialization.
    
    Args:
        backbone_cls: HuggingFace model class (e.g., ViTModel, DeiTModel)
        model_name: HuggingFace model identifier or path
        load_pretrained: Whether to load pretrained weights
        config_kwargs: Additional arguments passed to model config or from_pretrained().
                      Can include 'attn_implementation' to control attention mechanism
                      (e.g., 'eager', 'flash_attention_2', 'sdpa').
        backbone_dropout: Dropout probability to apply in backbone
    """
    
    def __init__(
        self,
        backbone_cls: Type[ViTBackboneProtocol],
        model_name: str,
        load_pretrained: bool = True,
        config_kwargs: Optional[Dict[str, Any]] = None,
        backbone_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        config_kwargs = config_kwargs or {}
        
        # Load model (pretrained or from config)
        if load_pretrained:
            self.backbone = backbone_cls.from_pretrained(model_name, **config_kwargs)
        else:
            # Load config and create model without pretrained weights
            config = backbone_cls.config_class.from_pretrained(model_name, **config_kwargs)
            self.backbone = backbone_cls(config)
        
        # Apply dropout if specified
        if backbone_dropout > 0.0:
            def set_dropout(module):
                if isinstance(module, nn.Dropout):
                    module.p = backbone_dropout
            self.backbone.apply(set_dropout)
    
    def get_embedding_dim(self) -> int:
        """Returns the embedding dimension of the backbone.
        
        Returns:
            Hidden size / embedding dimension
        """
        return self.backbone.config.hidden_size
    
    def get_cls_token_embedding(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """Extracts CLS token embedding from model outputs.
        
        Handles different output formats:
        - Models with 'pooler_output' (e.g., some CLIP models)
        - Models with 'last_hidden_state' where CLS token is first (e.g., ViT, DeiT)
        
        Args:
            outputs: Dictionary returned by backbone forward pass
        
        Returns:
            CLS token embedding tensor of shape (batch_size, hidden_size)
        """
        if "pooler_output" in outputs and outputs["pooler_output"] is not None:
            return outputs["pooler_output"]
        elif "last_hidden_state" in outputs:
            # CLS token is typically the first token in the sequence
            return outputs["last_hidden_state"][:, 0, :]
        else:
            raise ValueError(
                "Backbone output must contain either 'pooler_output' or "
                "'last_hidden_state' to extract CLS token embedding"
            )
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass through the backbone.
        
        Args:
            pixel_values: Input image tensor
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            **kwargs: Additional arguments passed to backbone
        
        Returns:
            Dictionary with backbone outputs
        """
        return self.backbone(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
