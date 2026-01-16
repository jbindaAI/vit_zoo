"""Model registry for ViT models."""

from typing import Dict, Tuple, Type
from transformers import (
    ViTModel as HFViTModel,
    DeiTModel,
    Dinov2Model,
    Dinov2WithRegistersModel,
    DINOv3ViTModel,
    CLIPVisionModel,
)

from ..types.interfaces import ViTBackboneProtocol

# Registry stores (backbone_class, default_model_name) tuples
MODEL_REGISTRY: Dict[str, Tuple[Type[ViTBackboneProtocol], str]] = {}

# Register default models
MODEL_REGISTRY.update({
    "vanilla_vit": (HFViTModel, "google/vit-base-patch16-224"),
    "deit_vit": (DeiTModel, "facebook/deit-base-distilled-patch16-224"),
    "dino_vit": (HFViTModel, "facebook/dino-vitb16"),
    "dinov2_vit": (Dinov2Model, "facebook/dinov2-base"),
    "dinov2_reg_vit": (Dinov2WithRegistersModel, "facebook/dinov2-with-registers-base"),
    "clip_vit": (CLIPVisionModel, "openai/clip-vit-base-patch16"),
    "dinov3_vit": (DINOv3ViTModel, "facebook/dinov3-vitb16-pretrain-lvd1689m"),
})


def list_models() -> list:
    """List all registered model types.
    
    Returns:
        List of registered model type strings
    """
    return list(MODEL_REGISTRY.keys())
