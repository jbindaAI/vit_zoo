"""Factory functions for creating ViT models."""

from typing import Optional, Union, Dict, Any, Type

from ..components.backbone import ViTBackbone
from ..model import ViTModel
from ..components.heads import BaseHead, LinearHead


def _create_head_from_config(
    head_config: Union[int, BaseHead],
    backbone_embedding_dim: int
) -> BaseHead:
    """Create a head from simple input formats.
    
    Args:
        head_config: 
            - int: Creates LinearHead with that output dimension
            - BaseHead: Validates that head's input_dim matches backbone embedding dimension
        backbone_embedding_dim: Backbone embedding dimension
    
    Returns:
        BaseHead instance
    
    Raises:
        ValueError: If provided BaseHead's input_dim doesn't match backbone embedding dimension.
    """
    if isinstance(head_config, int):
        return LinearHead(input_dim=backbone_embedding_dim, output_dim=head_config)
    else:
        # Validate input dimension matches
        head_input_dim = head_config.input_dim
        if head_input_dim != backbone_embedding_dim:
            raise ValueError(
                f"Head input dimension ({head_input_dim}) does not match "
                f"backbone embedding dimension ({backbone_embedding_dim}). "
                f"Please create a head with input_dim={backbone_embedding_dim}."
            )
        return head_config


def _create_vit_model(
    model_name: str,
    head: Optional[Union[int, BaseHead]] = None,
    backbone_cls: Optional[Type] = None,
    freeze_backbone: bool = False,
    load_pretrained: bool = True,
    backbone_dropout: float = 0.0,
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> ViTModel:
    """Generic factory function to create ViT models.
    
    Args:
        model_name: HuggingFace model identifier or path
        head: Head configuration (int or BaseHead). If int, creates LinearHead.
              If BaseHead, validates that head.input_dim matches backbone embedding dimension.
        backbone_cls: Optional HuggingFace model class (e.g., CLIPVisionModel).
                     When provided, used instead of AutoModel. Use for multi-modal models like CLIP.
        freeze_backbone: Freeze all backbone parameters
        load_pretrained: Whether to load pretrained weights
        backbone_dropout: Dropout probability for backbone
        config_kwargs: Extra config options passed to model config or from_pretrained().
                      Can include 'attn_implementation' to control attention mechanism
                      (e.g., 'eager' for attention weights, 'flash_attention_2', 'sdpa').
    
    Returns:
        Configured ViTModel instance
    """
    backbone = ViTBackbone(
        model_name=model_name,
        backbone_cls=backbone_cls,
        load_pretrained=load_pretrained,
        config_kwargs=config_kwargs,
        backbone_dropout=backbone_dropout,
    )
    
    # Handle head creation
    head_instance = None
    if head is not None:
        head_instance = _create_head_from_config(head, backbone.get_embedding_dim())
    
    return ViTModel(
        backbone=backbone,
        head=head_instance,
        freeze_backbone=freeze_backbone,
    )


def build_model(
    model_name: str,
    head: Optional[Union[int, BaseHead]] = None,
    backbone_cls: Optional[Type] = None,
    freeze_backbone: bool = False,
    load_pretrained: bool = True,
    backbone_dropout: float = 0.0,
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> ViTModel:
    """Build a ViT model from a HuggingFace model identifier.
    
    Uses AutoModel to auto-detect the model type (ViT, DeiT, DINOv2, CLIP, etc.)
    from the model's config on the HuggingFace Hub. Works with any ViT-compatible
    model hosted on HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier or path (e.g., 'google/vit-base-patch16-224',
                   'facebook/dinov2-base', 'openai/clip-vit-base-patch16').
        head: 
            - int: Creates LinearHead with that output dimension
            - BaseHead: Uses provided head instance. Validates that head.input_dim matches
                       backbone embedding dimension. Users can subclass BaseHead to create
                       custom heads (e.g., MLP, UNET decoder, attention-based, etc.)
            - None: No head (embedding extraction mode)
        backbone_cls: Optional HuggingFace model class (e.g., CLIPVisionModel).
                     When provided, used instead of AutoModel. Required for multi-modal
                     models like CLIP where AutoModel loads the full model.
        freeze_backbone: Freeze all backbone parameters
        load_pretrained: Whether to load pretrained weights
        backbone_dropout: Dropout probability for backbone
        config_kwargs: Extra config options passed to model config or from_pretrained().
                      Can include 'attn_implementation' to control attention mechanism
                      (e.g., 'eager' for attention weights, 'flash_attention_2', 'sdpa').
    
    Returns:
        Configured ViTModel instance
    
    Examples:
        >>> # Simple classification with 10 classes
        >>> model = build_model("google/vit-base-patch16-224", head=10, freeze_backbone=True)
        
        >>> # Use larger model variant
        >>> model = build_model("google/vit-large-patch16-224", head=10)
        
        >>> # Custom MLP head (create it yourself)
        >>> from vit_zoo import MLPHead
        >>> mlp_head = MLPHead(input_dim=768, hidden_dims=[512, 256], output_dim=100)
        >>> model = build_model("facebook/dinov2-base", head=mlp_head)
        
        >>> # Multi-modal models (CLIP) - pass backbone_cls for vision-only
        >>> from transformers import CLIPVisionModel
        >>> model = build_model("openai/clip-vit-base-patch16", backbone_cls=CLIPVisionModel, head=10)
        
        >>> # Custom head instance
        >>> from vit_zoo import LinearHead
        >>> head = LinearHead(input_dim=768, output_dim=10)
        >>> model = build_model("google/vit-base-patch16-224", head=head)
    """
    return _create_vit_model(
        model_name=model_name,
        head=head,
        backbone_cls=backbone_cls,
        freeze_backbone=freeze_backbone,
        load_pretrained=load_pretrained,
        backbone_dropout=backbone_dropout,
        config_kwargs=config_kwargs,
    )
