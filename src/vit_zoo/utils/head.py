"""Helper functions for head validation and creation."""

from vit_zoo.components.heads import BaseHead, LinearHead


def _validate_head_for_backbone(head: BaseHead, backbone_embedding_dim: int) -> None:
    """Check that head's input_dim matches the backbone embedding dimension.

    Args:
        head: Head module with an input_dim property.
        backbone_embedding_dim: Backbone embedding dimension.

    Raises:
        ValueError: If head.input_dim != backbone_embedding_dim.
    """
    head_input_dim = head.input_dim
    if head_input_dim != backbone_embedding_dim:
        raise ValueError(
            f"Head input dimension ({head_input_dim}) does not match "
            f"backbone embedding dimension ({backbone_embedding_dim}). "
            f"Please create a head with input_dim={backbone_embedding_dim}."
        )


def _create_linear_head(output_dim: int, backbone_embedding_dim: int) -> BaseHead:
    """Create a LinearHead for the given backbone embedding and output dimensions.

    Args:
        output_dim: Head output dimension (e.g. number of classes).
        backbone_embedding_dim: Backbone embedding dimension.

    Returns:
        LinearHead instance.
    """
    return LinearHead(input_dim=backbone_embedding_dim, output_dim=output_dim)
