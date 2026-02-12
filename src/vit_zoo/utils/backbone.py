"""Helper functions for loading and using HuggingFace ViT backbones."""

from typing import Dict, Any, Optional, Type
import torch
from torch import nn

from transformers import AutoModel, AutoConfig


def _load_backbone(
    model_name: str,
    backbone_cls: Optional[Type] = None,
    load_pretrained: bool = True,
    config_kwargs: Optional[Dict[str, Any]] = None,
    backbone_dropout: float = 0.0,
) -> nn.Module:
    """Load a raw HuggingFace ViT backbone (no wrapper). Private helper.

    Uses AutoModel to auto-detect model type from the HuggingFace model
    identifier, or a specific class when provided (e.g., CLIPVisionModel).

    Args:
        model_name: HuggingFace model identifier or path
        backbone_cls: Optional HuggingFace model class (e.g., CLIPVisionModel).
                     When provided, used instead of AutoModel.
        load_pretrained: Whether to load pretrained weights
        config_kwargs: Additional arguments passed to model config or from_pretrained().
        backbone_dropout: Dropout probability to apply in backbone

    Returns:
        Raw HuggingFace backbone module (e.g., ViTModel, CLIPVisionModel)
    """
    config_kwargs = config_kwargs or {}
    model_cls = backbone_cls if backbone_cls is not None else AutoModel
    config_cls = backbone_cls.config_class if backbone_cls is not None else AutoConfig

    if load_pretrained:
        backbone = model_cls.from_pretrained(model_name, **config_kwargs)
    else:
        config = config_cls.from_pretrained(model_name, **config_kwargs)
        backbone = model_cls.from_config(config)

    if backbone_dropout > 0.0:
        def set_dropout(module):
            if isinstance(module, nn.Dropout):
                module.p = backbone_dropout
        backbone.apply(set_dropout)

    return backbone


def get_embedding_dim(backbone: nn.Module) -> int:
    """Returns the embedding dimension of the backbone.

    Args:
        backbone: HuggingFace backbone with a .config.hidden_size attribute

    Returns:
        Hidden size / embedding dimension
    """
    return backbone.config.hidden_size


def get_cls_token_embedding(outputs: Dict[str, Any]) -> torch.Tensor:
    """Extracts CLS token embedding from backbone forward outputs.

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
        return outputs["last_hidden_state"][:, 0, :]
    else:
        raise ValueError(
            "Backbone output must contain either 'pooler_output' or "
            "'last_hidden_state' to extract CLS token embedding"
        )
