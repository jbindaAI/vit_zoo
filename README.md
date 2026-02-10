<p align="center">
  <img src="https://raw.githubusercontent.com/jbindaAI/vit_zoo/main/assets/vit_zoo_logo_v2.png" alt="vit_zoo logo" width="220" />
</p>

<p align="center">
  <a href="https://pypi.org/project/vit-zoo/"><img alt="PyPI" src="https://img.shields.io/pypi/v/vit-zoo.svg" /></a>
  <a href="https://pypi.org/project/vit-zoo/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/vit-zoo.svg" /></a>
  <a href="https://github.com/jbindaAI/vit_zoo/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/jbindaAI/vit_zoo/actions/workflows/tests.yml/badge.svg" /></a>
  <a href="https://github.com/jbindaAI/vit_zoo"><img alt="Source" src="https://img.shields.io/badge/source-GitHub-0B1020" /></a>
</p>

A clean, extensible factory for creating HuggingFace-based Vision Transformer models (ViT, DeiT, DINO, DINOv2, DINOv3, CLIP) with flexible heads and easy backbone freezing.

## Installation

```bash
pip install vit_zoo
```

From source:

```bash
git clone https://github.com/jbindaAI/vit_zoo.git
cd vit_zoo
pip install -e .
```

For development: `pip install -e ".[dev]"`

## Quick start

```python
from vit_zoo.factory import build_model

model = build_model("facebook/dinov2-base", head=10, freeze_backbone=True)
outputs = model(images)
logits = outputs["predictions"]  # (batch_size, 10)
```

### Basic usage

```python
from vit_zoo.factory import build_model

# Simple classification - pass any HuggingFace model ID
model = build_model("google/vit-base-patch16-224", head=10, freeze_backbone=True)
outputs = model(images)
predictions = outputs["predictions"]  # Shape: (batch_size, 10)
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

model = build_model("facebook/dinov2-base", head=mlp_head)
```

### Embedding Extraction

```python
from transformers import CLIPVisionModel
model = build_model("openai/clip-vit-base-patch16", backbone_cls=CLIPVisionModel, head=None)
outputs = model(images, output_embeddings=True)
hidden_states = outputs["last_hidden_state"]  # (batch_size, seq_len, embedding_dim)
cls_embedding = hidden_states[:, 0, :]  # (batch_size, embedding_dim)
predictions = outputs["predictions"]  # same as cls_embedding when head=None (IdentityHead)
```

### Attention Weights

```python
model = build_model(
    "google/vit-base-patch16-224",
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
model = build_model("google/vit-base-patch16-224", head=head)
```

### Multi-modal Models (CLIP)

For CLIP and other multi-modal models, pass `backbone_cls` to load only the vision encoder (AutoModel would load the full model):

```python
from transformers import CLIPVisionModel
model = build_model("openai/clip-vit-base-patch16", backbone_cls=CLIPVisionModel, head=10)
```

### Any HuggingFace Model

`build_model` uses AutoModel to auto-detect the model type from the HuggingFace Hub. Any ViT-compatible model works:

```python
model = build_model("google/vit-large-patch16-224", head=10)
model = build_model("facebook/deit-base-distilled-patch16-224", head=10)
model = build_model("facebook/dinov2-with-registers-base", head=10)
```

## API Reference

### `build_model()`

```python
build_model(
    model_name: str,
    head: Optional[Union[int, BaseHead]] = None,
    backbone_cls: Optional[Type] = None,
    freeze_backbone: bool = False,
    load_pretrained: bool = True,
    backbone_dropout: float = 0.0,
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> ViTModel
```

**Parameters:**
- `model_name`: HuggingFace model identifier (e.g., `"google/vit-base-patch16-224"`, `"facebook/dinov2-base"`, `"openai/clip-vit-base-patch16"`). Uses AutoModel to auto-detect model type.
- `head`: `int` (creates LinearHead), `BaseHead` instance, or `None` (embedding extraction)
- `backbone_cls`: Optional HuggingFace model class (e.g., `CLIPVisionModel`). Use for multi-modal models where AutoModel loads the full model.
- `freeze_backbone`: Freeze all backbone parameters
- `config_kwargs`: Extra config options (e.g., `{"attn_implementation": "eager"}`)

**Usage:**
- `build_model("google/vit-base-patch16-224", head=10)`
- `build_model("facebook/dinov2-base", head=None)` (embedding extraction)

### `ViTModel.forward()`

```python
forward(
    pixel_values: torch.Tensor,
    output_attentions: bool = False,
    output_embeddings: bool = False,
) -> Dict[str, Any]
```

Always returns a dict. Keys:
- `"predictions"`: head output tensor (always present)
- `"attentions"`: optional, when `output_attentions=True`
- `"last_hidden_state"`: optional, when `output_embeddings=True`; shape `(batch_size, seq_len, embedding_dim)`

### Freezing the backbone

```python
model.backbone.freeze_backbone(freeze: bool = True)  # Freeze/unfreeze backbone
```

## Supported Models

Any ViT-compatible model on the HuggingFace Hub works. Examples:

- `google/vit-base-patch16-224`, `google/vit-large-patch16-224` (ViT)
- `facebook/deit-base-distilled-patch16-224` (DeiT)
- `facebook/dino-vitb16` (DINO)
- `facebook/dinov2-base`, `facebook/dinov2-with-registers-base` (DINOv2)
- `facebook/dinov3-vitb16-pretrain-lvd1689m` (DINOv3)
- `openai/clip-vit-base-patch16` (CLIP Vision; pass `backbone_cls=CLIPVisionModel`)

Browse the [HuggingFace Hub](https://huggingface.co/models?library=transformers&other=vision) for more models.

## Import Patterns

```python
from vit_zoo import ViTModel
from vit_zoo.factory import build_model
from vit_zoo.components import ViTBackbone, BaseHead, LinearHead, MLPHead, IdentityHead
```

## Available Heads

- `LinearHead`: Simple linear layer (auto-created when `head=int`)
- `MLPHead`: Multi-layer perceptron with configurable depth, activation, dropout
- `IdentityHead`: Returns embeddings unchanged

All heads must implement `input_dim` property. Custom heads by subclassing `BaseHead`.

## License

GPL-3.0
