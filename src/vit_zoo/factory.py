"""Factory functions for creating ViT models."""

from typing import Optional, Union, Type, Tuple, Dict, Any
from transformers import (
    ViTModel as HFViTModel,
    DeiTModel,
    Dinov2Model,
    Dinov2WithRegistersModel,
    CLIPVisionModel,
)

from .model_registry import MODEL_REGISTRY
from .backbone import ViTBackbone
from .model import ViTModel
from .heads import BaseHead, LinearHead
from .interfaces import ViTBackboneProtocol


# Registry stores (backbone_class, default_model_name) tuples
MODEL_REGISTRY.update({
    "vanilla_vit": (HFViTModel, "google/vit-base-patch16-224"),
    "deit_vit": (DeiTModel, "facebook/deit-base-distilled-patch16-224"),
    "dino_vit": (HFViTModel, "facebook/dino-vitb16"),
    "dino_v2_vit": (Dinov2Model, "facebook/dinov2-base"),
    "dinov2_reg_vit": (Dinov2WithRegistersModel, "facebook/dinov2-with-registers-base"),
    "clip_vit": (CLIPVisionModel, "openai/clip-vit-base-patch16"),
    "dinov3_vit": (HFViTModel, "facebook/dinov3-vitb16-pretrain-lvd1689m"),
})


def _create_head_from_config(
    head_config: Union[int, BaseHead],
    input_dim: int
) -> BaseHead:
    """Create a head from simple input formats.
    
    Args:
        head_config: 
            - int: Creates LinearHead with that output dimension
            - BaseHead: Returns the head instance as-is (user is responsible for
                       ensuring input dimension matches backbone embedding dimension)
        input_dim: Input embedding dimension (used when creating LinearHead from int)
    
    Returns:
        BaseHead instance
    
    Raises:
        TypeError: If head_config is not int or BaseHead
    """
    if isinstance(head_config, int):
        return LinearHead(input_dim=input_dim, output_dim=head_config)
    elif isinstance(head_config, BaseHead):
        return head_config
    else:
        raise TypeError(
            f"head must be int or BaseHead, got {type(head_config)}"
        )


def _create_vit_model(
    backbone_cls: Type[ViTBackboneProtocol],
    model_name: str,
    head: Optional[Union[int, BaseHead]] = None,
    freeze_backbone: bool = False,
    freeze_layers: Optional[list] = None,
    load_pretrained: bool = True,
    backbone_dropout: float = 0.0,
    config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ViTModel:
    """Generic factory function to create ViT models.
    
    Args:
        backbone_cls: HuggingFace model class (e.g., ViTModel, DeiTModel)
        model_name: HuggingFace model identifier or path
        head: Head configuration (int or BaseHead). If int, creates LinearHead.
              If BaseHead, uses as-is (user must ensure input dimension matches).
        freeze_backbone: Freeze all backbone parameters
        freeze_layers: List of layer indices to freeze (0-indexed)
        load_pretrained: Whether to load pretrained weights
        backbone_dropout: Dropout probability for backbone
        config_kwargs: Extra config options passed to model config or from_pretrained().
                      Can include 'attn_implementation' to control attention mechanism
                      (e.g., 'eager' for attention weights, 'flash_attention_2', 'sdpa').
    
    Returns:
        Configured ViTModel instance
    """
    backbone = ViTBackbone(
        backbone_cls=backbone_cls,
        model_name=model_name,
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
        freeze_layers=freeze_layers,
    )


def build_model(
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    backbone_cls: Optional[Type[ViTBackboneProtocol]] = None,
    head: Optional[Union[int, BaseHead]] = None,
    freeze_backbone: bool = False,
    freeze_layers: Optional[list] = None,
    load_pretrained: bool = True,
    backbone_dropout: float = 0.0,
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> ViTModel:
    """Build a ViT model with flexible configuration.
    
    Can be used in three ways:
    
    1. **Registry shortcut** (recommended for common models):
       ```python
       model = build_model("vanilla_vit", head=10)
       ```
    
    2. **Override registry default** (use different variant):
       ```python
       model = build_model("vanilla_vit", model_name="google/vit-large-patch16-224", head=10)
       ```
    
    3. **Direct usage** (for any HuggingFace model):
       ```python
       from transformers import ViTModel
       model = build_model(
           model_name="google/vit-base-patch16-224",
           backbone_cls=ViTModel,
           head=10
       )
       ```
    
    Args:
        model_type: Registered model type (e.g., 'vanilla_vit', 'deit_vit', 'dino_v2_vit').
                   If provided, uses default backbone class and model name from registry.
                   When model_type is provided, backbone_cls is ignored and cannot be overridden.
        model_name: HuggingFace model identifier or path. If model_type is provided,
                   this overrides the default model name from registry. If model_type is not
                   provided, this is required along with backbone_cls.
        backbone_cls: HuggingFace model class. Required if model_type is not provided.
                     Ignored if model_type is provided (registry default is always used).
        head: 
            - int: Creates LinearHead with that output dimension
            - BaseHead: Uses provided head instance as-is. Users can subclass BaseHead
                       to create custom heads (e.g., MLP, UNET decoder, etc.)
            - None: No head (embedding extraction mode)
        freeze_backbone: Freeze all backbone parameters
        freeze_layers: List of layer indices to freeze (0-indexed)
        load_pretrained: Whether to load pretrained weights
        backbone_dropout: Dropout probability for backbone
        config_kwargs: Extra config options passed to model config or from_pretrained().
                      Can include 'attn_implementation' to control attention mechanism
                      (e.g., 'eager' for attention weights, 'flash_attention_2', 'sdpa').
    
    Returns:
        Configured ViTModel instance
    
    Examples:
        >>> # Simple classification with 10 classes using registry
        >>> model = build_model("vanilla_vit", head=10, freeze_backbone=True)
        
        >>> # Use different model variant
        >>> model = build_model("vanilla_vit", model_name="google/vit-large-patch16-224", head=10)
        
        >>> # Custom MLP head (create it yourself)
        >>> from vit_zoo import MLPHead
        >>> mlp_head = MLPHead(input_dim=768, hidden_dims=[512, 256], output_dim=100)
        >>> model = build_model("dino_v2_vit", head=mlp_head)
        
        >>> # Embedding extraction only
        >>> model = build_model("clip_vit", head=None)
        
        >>> # Direct usage with any HuggingFace model
        >>> from transformers import ViTModel
        >>> model = build_model(
        ...     model_name="google/vit-base-patch16-224",
        ...     backbone_cls=ViTModel,
        ...     head=10
        ... )
        
        >>> # Custom head instance
        >>> from vit_zoo import LinearHead
        >>> head = LinearHead(input_dim=768, output_dim=10)
        >>> model = build_model("vanilla_vit", head=head)
    """
    # (1) If model_type is provided, then user can only override model_name.
    if model_type is not None:
        if model_type not in MODEL_REGISTRY:
            available = list(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Available types: {available}"
            )
        
        # Get default backbone_cls and model_name from the registry
        default_backbone_cls, default_model_name = MODEL_REGISTRY[model_type]
        
        # Use defaults or overrides a model_name with provided.
        final_backbone_cls = default_backbone_cls
        final_model_name = model_name if model_name is not None else default_model_name
    
    # (2) If model_type is not provided, then both backbone_cls and model_name must be present.
    else:
        if backbone_cls is None or model_name is None:
            raise ValueError(
                "To use a non-default ViT backbone, you have to provide both `backbone_cls` and `model_name`."
            )
        final_backbone_cls = backbone_cls
        final_model_name = model_name
    
    # (3) Create an instance of ViT Model
    return _create_vit_model(
        backbone_cls=final_backbone_cls,
        model_name=final_model_name,
        head=head,
        freeze_backbone=freeze_backbone,
        freeze_layers=freeze_layers,
        load_pretrained=load_pretrained,
        backbone_dropout=backbone_dropout,
        config_kwargs=config_kwargs,
    )


def list_models() -> list:
    """List all registered model types.
    
    Returns:
        List of registered model type strings
    """
    return list(MODEL_REGISTRY.keys())
