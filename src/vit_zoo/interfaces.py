"""Protocol definitions for ViT backbone models."""

from typing import Protocol, Dict, Any
import torch


class ViTBackboneProtocol(Protocol):
    """Protocol defining the interface for ViT backbone models.
    
    This protocol abstracts over different HuggingFace ViT model implementations,
    providing a common interface for the backbone wrapper.
    """
    
    config: Any  # Model config object with attributes like hidden_size
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass returning dict with model outputs.
        
        Args:
            pixel_values: Input image tensor of shape (batch_size, channels, height, width)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            **kwargs: Additional arguments passed to the model
        
        Returns:
            Dictionary containing:
            - 'last_hidden_state': Tensor of shape (batch_size, seq_len, hidden_size)
            - 'pooler_output': Optional tensor of shape (batch_size, hidden_size) for CLS token
            - 'attentions': Optional tuple of attention tensors if output_attentions=True
            - 'hidden_states': Optional tuple of hidden state tensors if output_hidden_states=True
        """
        ...
