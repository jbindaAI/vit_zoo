"""Utility functions for backbone loading and head creation."""

from .backbone import (
    _load_backbone,
    get_embedding_dim,
    get_cls_token_embedding,
)

__all__ = [
    "_load_backbone",
    "get_embedding_dim",
    "get_cls_token_embedding",
]
