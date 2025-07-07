from typing import Dict, Callable
from .factory import VisionTransformer

ModelFn = Callable[..., VisionTransformer]
MODEL_REGISTRY: Dict[str, ModelFn] = {}

def register_model(name: str):
    def decorator(fn: ModelFn):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator
