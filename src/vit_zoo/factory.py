import torch
from torch import nn
from typing import Optional, Type

from transformers import (
    ViTModel, ViTConfig,
    DeiTModel, DeiTConfig,
    Dinov2Model, Dinov2Config,
    Dinov2WithRegistersModel, Dinov2WithRegistersConfig,
    CLIPVisionModel, CLIPVisionConfig
)

from .utils import set_encoder_dropout_p
from .model_registry import register_model, MODEL_REGISTRY


# === Base Vision Transformer Wrapper ===
class VisionTransformer(nn.Module):
    """
    Wraps HuggingFace vision models with a custom classification head.
    """
    def __init__(
        self,
        backbone_cls: Type[nn.Module],
        config_cls: Type,
        model_name: str,
        head_dim: int,
        backbone_dropout: float = 0.0,
        freeze_backbone: bool = False,
        load_pretrained_backbone: bool = False,
        config_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        config_kwargs = config_kwargs or {}

        # Load model or config
        if load_pretrained_backbone:
            self.backbone = backbone_cls.from_pretrained(model_name, **config_kwargs)
        else:
            config = config_cls.from_pretrained(model_name, **config_kwargs)
            self.backbone = backbone_cls(config)

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Optionally modify dropout rate
        if backbone_dropout > 0.0:
            self.backbone.apply(lambda m: set_encoder_dropout_p(m, dropout_p=backbone_dropout))

        # Add classification head
        self.head = nn.Linear(self.backbone.config.hidden_size, head_dim)

    def forward(self, x: torch.Tensor, output_attentions: bool = False):
        out = self.backbone(x, output_attentions=output_attentions)
        x = self._get_embedding(out)
        x = self.head(x)
        if output_attentions and "attentions" in out:
            return x, out["attentions"]
        return x

    def _get_embedding(self, backbone_output):
        """Extracts pooled embedding (usually CLS token)."""
        return backbone_output["pooler_output"]


# === Register each model ===

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