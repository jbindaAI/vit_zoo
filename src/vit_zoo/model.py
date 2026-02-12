"""Main ViT model class with flexible head and backbone."""

from typing import Optional, Union, Dict, Any, Type
import torch
from torch import nn

from .components.heads import BaseHead, IdentityHead
from .utils import (
    _load_backbone,
    _get_embedding_dim,
    _get_cls_token_embedding,
    _validate_head_for_backbone,
    _create_linear_head,
)


class ViTModel(nn.Module):
    """Main Vision Transformer model with flexible head and backbone.

    Single entry point: pass model_name (and optionally head, backbone_cls, etc.)
    to build from a HuggingFace identifier, or pass backbone= for a pre-built backbone.

    Args:
        model_name: HuggingFace model identifier (e.g. 'google/vit-base-patch16-224').
                   Required when backbone is not provided.
        head: int (LinearHead output dim), BaseHead instance, or None (IdentityHead).
        backbone: Optional pre-built backbone; if set, model_name and backbone args are ignored.
        backbone_cls: Optional HuggingFace model class (e.g. CLIPVisionModel).
        freeze_backbone: Whether to freeze backbone parameters.
        load_pretrained: Whether to load pretrained weights (when loading from model_name).
        backbone_dropout: Dropout probability in backbone.
        config_kwargs: Extra kwargs for config / from_pretrained (e.g. attn_implementation).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        head: Optional[Union[int, BaseHead]] = None,
        backbone: Optional[nn.Module] = None,
        backbone_cls: Optional[Type] = None,
        freeze_backbone: bool = False,
        load_pretrained: bool = True,
        backbone_dropout: float = 0.0,
        config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if backbone is not None:
            self.backbone = backbone
        else:
            if model_name is None:
                raise ValueError("Either model_name or backbone must be provided.")
            self.backbone = _load_backbone(
                model_name=model_name,
                backbone_cls=backbone_cls,
                load_pretrained=load_pretrained,
                config_kwargs=config_kwargs or {},
                backbone_dropout=backbone_dropout,
            )
        if head is None:
            self.head = IdentityHead(input_dim=self.embedding_dim)
        elif isinstance(head, int):
            self.head = _create_linear_head(head, self.embedding_dim)
        else:
            _validate_head_for_backbone(head, self.embedding_dim)
            self.head = head
        if freeze_backbone:
            self.freeze_backbone(freeze=True)

    @property
    def embedding_dim(self) -> int:
        """Returns the embedding dimension of the backbone."""
        return _get_embedding_dim(self.backbone)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze all backbone parameters.

        Args:
            freeze: If True, freeze parameters; if False, unfreeze them
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass through the model.

        Args:
            pixel_values: Input image tensor of shape (batch_size, channels, height, width)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return last_hidden_state from the backbone
            **kwargs: Additional arguments passed to the backbone

        Returns:
            A dict with at least the key 'predictions' (head output tensor). Optional keys:
            - 'attentions': when output_attentions=True
            - 'last_hidden_state': when output_hidden_states=True
        """
        backbone_outputs = self.backbone(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        cls_embedding = _get_cls_token_embedding(backbone_outputs)
        predictions = self.head(cls_embedding)
        results: Dict[str, Any] = {"predictions": predictions}
        if output_attentions:
            if "attentions" not in backbone_outputs:
                raise ValueError(
                    "Backbone did not return 'attentions' (output_attentions=True)."
                )
            results["attentions"] = backbone_outputs["attentions"]
        if output_hidden_states:
            if "last_hidden_state" not in backbone_outputs:
                raise ValueError(
                    "Backbone did not return 'last_hidden_state' (output_hidden_states=True)."
                )
            results["last_hidden_state"] = backbone_outputs["last_hidden_state"]
        return results
