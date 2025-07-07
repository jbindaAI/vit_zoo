import torch
from torch import nn
from typing import Optional, Type

from .utils import set_encoder_dropout_p


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