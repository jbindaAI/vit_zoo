# Vision Transformer Zoo

A clean, extensible factory for creating HuggingFace-based Vision Transformer models (ViT, DeiT, DINO, DINOv2, DINOv3, CLIP) with flexible heads and easy backbone freezing.

---

## Installation (PyPI)

```bash
pip3 install vit_zoo
```
## Installation (From source)

```bash
git clone git@github.com:jbindaAI/vit_zoo.git
cd vit_zoo
pip install -e .
```

For development: `pip install -e ".[dev]"`

---

## Quick Start

### Basic Usage

```python
from vit_zoo.factory import build_model

# Simple classification
model = build_model("vanilla_vit", head=10, freeze_backbone=True)
predictions = model(images)  # Shape: (batch_size, 10)
```

### Custom MLP Head

```python
from vit_zoo.factory import build_model
from vit_zoo.components import MLPHead

mlp_head = MLPHead(
    input_dim=768,
    hidden_dims=[512, 256],
    output_dim=100,
    dropout=0.1,
    activation="gelu"  # or 'relu', 'tanh', or nn.Module
)

model = build_model("dinov2_vit", head=mlp_head)
```

### Embedding Extraction

```python
model = build_model("clip_vit", head=None)
outputs = model(images, output_embeddings=True)
embeddings = outputs["embeddings"]  # Shape: (batch_size, embedding_dim)
```

### Attention Weights

```python
model = build_model(
    "vanilla_vit",
    head=10,
    config_kwargs={"attn_implementation": "eager"}
)
outputs = model(images, output_attentions=True)
attentions = outputs["attentions"]
```

### Custom Head

```python
from vit_zoo.components import BaseHead
import torch.nn as nn

class CustomHead(BaseHead):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self._input_dim = input_dim
        self.fc = nn.Linear(input_dim, num_classes)
    
    @property
    def input_dim(self) -> int:
        return self._input_dim
    
    def forward(self, embeddings):
        return self.fc(embeddings)

head = CustomHead(input_dim=768, num_classes=10)
model = build_model("vanilla_vit", head=head)
```

### Direct Usage (Any HuggingFace Model)

```python
from vit_zoo.factory import build_model
from transformers import ViTModel

model = build_model(
    model_name="google/vit-large-patch16-224",
    backbone_cls=ViTModel,
    head=10
)
```

---

## API Reference

### `build_model()`

```python
build_model(
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    backbone_cls: Optional[Type[ViTBackboneProtocol]] = None,
    head: Optional[Union[int, BaseHead]] = None,
    freeze_backbone: bool = False,
    load_pretrained: bool = True,
    backbone_dropout: float = 0.0,
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> ViTModel
```

**Parameters:**
- `model_type`: Registry key (`"vanilla_vit"`, `"deit_vit"`, `"dinov2_vit"`, etc.)
- `head`: `int` (creates LinearHead), `BaseHead` instance, or `None` (embedding extraction)
- `freeze_backbone`: Freeze all backbone parameters
- `config_kwargs`: Extra config options (e.g., `{"attn_implementation": "eager"}`)

**Usage:**
- Registry: `build_model("vanilla_vit", head=10)`
- Override: `build_model("vanilla_vit", model_name="google/vit-large-patch16-224", head=10)`
- Direct: `build_model(model_name="...", backbone_cls=ViTModel, head=10)`

### `ViTModel.forward()`

```python
forward(
    pixel_values: torch.Tensor,
    output_attentions: bool = False,
    output_embeddings: bool = False,
) -> Union[torch.Tensor, Dict[str, Any]]
```

Returns predictions tensor, or dict with `"predictions"`, `"attentions"`, `"embeddings"` keys.

### `ViTModel.freeze_backbone()`

```python
model.freeze_backbone(freeze: bool = True)  # Freeze/unfreeze backbone
```

### `list_models()`

```python
from vit_zoo.factory import list_models
available = list_models()  # Returns list of registered model types
```

---

## Supported Models

- `vanilla_vit`: Google ViT (`google/vit-base-patch16-224`)
- `deit_vit`: Facebook DeiT (`facebook/deit-base-distilled-patch16-224`)
- `dino_vit`: Facebook DINO (`facebook/dino-vitb16`)
- `dinov2_vit`: Facebook DINOv2 (`facebook/dinov2-base`)
- `dinov2_reg_vit`: DINOv2 with registers (`facebook/dinov2-with-registers-base`)
- `dinov3_vit`: Facebook DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`)
- `clip_vit`: OpenAI CLIP Vision (`openai/clip-vit-base-patch16`)

---

## Import Patterns

```python
from vit_zoo import ViTModel
from vit_zoo.factory import build_model, list_models
from vit_zoo.components import ViTBackbone, BaseHead, LinearHead, MLPHead, IdentityHead
```

---

## Available Heads

- `LinearHead`: Simple linear layer (auto-created when `head=int`)
- `MLPHead`: Multi-layer perceptron with configurable depth, activation, dropout
- `IdentityHead`: Returns embeddings unchanged

All heads must implement `input_dim` property. Custom heads by subclassing `BaseHead`.

---

## License

GPL-3.0
