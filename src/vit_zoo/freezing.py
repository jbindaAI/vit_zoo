"""Utilities for freezing/unfreezing model parameters."""

from typing import List, Iterator
import torch
from torch import nn


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze all backbone parameters.
    
    Args:
        model: Model with a 'backbone' attribute
        freeze: If True, freeze parameters; if False, unfreeze them
    """
    if not hasattr(model, "backbone"):
        raise ValueError("Model must have a 'backbone' attribute")
    
    for param in model.backbone.parameters():
        param.requires_grad = not freeze


def freeze_layers(model: nn.Module, layer_indices: List[int], freeze: bool = True) -> None:
    """Freeze or unfreeze specific transformer layers by index.
    
    Args:
        model: Model with a 'backbone' attribute that has an 'encoder' with 'layer' attribute
        layer_indices: List of layer indices to freeze/unfreeze (0-indexed)
        freeze: If True, freeze parameters; if False, unfreeze them
    
    Raises:
        ValueError: If model structure doesn't match expected format
    """
    if not hasattr(model, "backbone"):
        raise ValueError("Model must have a 'backbone' attribute")
    
    backbone = model.backbone
    
    # Try to access encoder layers
    if not hasattr(backbone, "encoder"):
        raise ValueError("Backbone must have an 'encoder' attribute")
    
    encoder = backbone.encoder
    
    if not hasattr(encoder, "layer"):
        raise ValueError("Encoder must have a 'layer' attribute")
    
    layers = encoder.layer
    
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("Encoder layers must be a ModuleList")
    
    num_layers = len(layers)
    
    for idx in layer_indices:
        if idx < 0 or idx >= num_layers:
            raise ValueError(
                f"Layer index {idx} is out of range. "
                f"Model has {num_layers} layers (indices 0-{num_layers-1})"
            )
        
        for param in layers[idx].parameters():
            param.requires_grad = not freeze


def get_trainable_parameters(model: nn.Module) -> Iterator[torch.nn.Parameter]:
    """Get iterator over trainable parameters.
    
    Args:
        model: PyTorch model
    
    Yields:
        Trainable parameters (parameters where requires_grad=True)
    """
    for param in model.parameters():
        if param.requires_grad:
            yield param
