"""Main ViT model class with flexible head and backbone."""

from typing import Optional, Union, Dict, Any, Type
import torch
from torch import nn

from .components.heads import BaseHead, IdentityHead, LinearHead
from .utils import load_backbone, get_embedding_dim, get_cls_token_embedding


def _create_head_from_config(
    head_config: Union[int, BaseHead],
    backbone_embedding_dim: int,
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
    head_input_dim = head_config.input_dim
    if head_input_dim != backbone_embedding_dim:
        raise ValueError(
            f"Head input dimension ({head_input_dim}) does not match "
            f"backbone embedding dimension ({backbone_embedding_dim}). "
            f"Please create a head with input_dim={backbone_embedding_dim}."
        )
    return head_config


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
            self.backbone = load_backbone(
                model_name=model_name,
                backbone_cls=backbone_cls,
                load_pretrained=load_pretrained,
                config_kwargs=config_kwargs or {},
                backbone_dropout=backbone_dropout,
            )
        dim = get_embedding_dim(self.backbone)
        if head is None:
            self.head = IdentityHead(input_dim=dim)
        else:
            self.head = _create_head_from_config(head, dim)
        if freeze_backbone:
            self.freeze_backbone(freeze=True)

    @property
    def embedding_dim(self) -> int:
        """Returns the embedding dimension of the backbone."""
        return get_embedding_dim(self.backbone)

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
        cls_embedding = get_cls_token_embedding(backbone_outputs)
        predictions = self.head(cls_embedding)
        results: Dict[str, Any] = {"predictions": predictions}
        if output_attentions:
            results["attentions"] = backbone_outputs["attentions"]
        if output_hidden_states:
            results["last_hidden_state"] = backbone_outputs["last_hidden_state"]
        return results
