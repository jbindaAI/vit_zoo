"""Model registry for ViT models."""

from typing import Dict, Tuple, Type
from .interfaces import ViTBackboneProtocol

# Registry stores (backbone_class, default_model_name) tuples
MODEL_REGISTRY: Dict[str, Tuple[Type[ViTBackboneProtocol], str]] = {}
