from typing import Optional
from transformers import (
    ViTModel, ViTConfig,
    DeiTModel, DeiTConfig,
    Dinov2Model, Dinov2Config,
    Dinov2WithRegistersModel, Dinov2WithRegistersConfig,
    CLIPVisionModel, CLIPVisionConfig
)

from .model_registry import register_model, MODEL_REGISTRY
from .vision_base import VisionTransformer



@register_model("vanilla_vit")
def create_vanilla_vit(**kwargs) -> VisionTransformer:
    return VisionTransformer(
        backbone_cls=ViTModel,
        config_cls=ViTConfig,
        model_name="google/vit-base-patch16-224",
        **kwargs
    )


@register_model("deit_vit")
def create_deit(**kwargs) -> VisionTransformer:
    model_size = kwargs.pop("model_size", "base")
    assert model_size in ["tiny", "small", "base"]
    model_name = f"facebook/deit-{model_size}-distilled-patch16-224"
    return VisionTransformer(
        backbone_cls=DeiTModel,
        config_cls=DeiTConfig,
        model_name=model_name,
        **kwargs
    )


@register_model("dino_vit")
def create_dino(**kwargs) -> VisionTransformer:
    model_size = kwargs.pop("model_size", "small")
    patch_size = kwargs.pop("patch_size", 8)
    assert model_size in ["small", "base"]
    assert patch_size in [8, 16]
    model_name = f"facebook/dino-vit{model_size[0]}{patch_size}"
    return VisionTransformer(
        backbone_cls=ViTModel,
        config_cls=ViTConfig,
        model_name=model_name,
        **kwargs
    )


@register_model("dino_v2")
def create_dino_v2(**kwargs) -> VisionTransformer:
    model_size = kwargs.pop("model_size", "base")
    with_registers = kwargs.pop("with_registers", True)
    assert model_size in ["small", "base"]
    if with_registers:
        backbone_cls, config_cls = Dinov2WithRegistersModel, Dinov2WithRegistersConfig
        model_name = f"facebook/dinov2-with-registers-{model_size}"
    else:
        backbone_cls, config_cls = Dinov2Model, Dinov2Config
        model_name = f"facebook/dinov2-{model_size}"
    return VisionTransformer(
        backbone_cls=backbone_cls,
        config_cls=config_cls,
        model_name=model_name,
        **kwargs
    )


@register_model("clip")
def create_clip(**kwargs) -> VisionTransformer:
    return VisionTransformer(
        backbone_cls=CLIPVisionModel,
        config_cls=CLIPVisionConfig,
        model_name="openai/clip-vit-base-patch16",
        **kwargs
    )


def build_model(
    model_type: str,
    head_dim: int = 1,
    backbone_dropout: float = 0.0,
    load_pretrained_backbone: bool = False,
    freeze_backbone: bool = False,
    config_kwargs: Optional[dict] = None,
    **kwargs
    ) -> VisionTransformer:
    """
    Builds a vision model using a registered architecture.

    Args:
        model_type: One of registered model keys.
        head_dim: Output dimension of classification head.
        backbone_dropout: Dropout probability to apply in backbone.
        load_pretrained_backbone: Whether to load pretrained weights.
        freeze_backbone: If True, freezes all backbone parameters.
        config_kwargs: Extra config options passed to model configs.
        **kwargs: Extra model-specific arguments (e.g., model_size, patch_size).

    Returns:
        A VisionTransformer instance.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Available types: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type](
        head_dim=head_dim,
        backbone_dropout=backbone_dropout,
        load_pretrained_backbone=load_pretrained_backbone,
        freeze_backbone=freeze_backbone,
        config_kwargs=config_kwargs,
        **kwargs
    )