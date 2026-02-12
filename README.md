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
from vit_zoo import ViTModel

model = ViTModel("facebook/dinov2-base", head=10, freeze_backbone=True)
outputs = model(images)
logits = outputs["predictions"]  # (batch_size, 10)
```

### Basic usage

```python
from vit_zoo import ViTModel

# Simple classification - pass any HuggingFace model ID
model = ViTModel("google/vit-base-patch16-224", head=10, freeze_backbone=True)
outputs = model(images)
predictions = outputs["predictions"]  # Shape: (batch_size, 10)
```

### Custom MLP Head

```python
from vit_zoo import ViTModel, MLPHead

mlp_head = MLPHead(
    input_dim=768,
    hidden_dims=[512, 256],
    output_dim=100,
    dropout=0.1,
    activation="gelu"  # or 'relu', 'tanh', or nn.Module
)

model = ViTModel("facebook/dinov2-base", head=mlp_head)
```

### Embedding Extraction

```python
from vit_zoo import ViTModel
from transformers import CLIPVisionModel

model = ViTModel("openai/clip-vit-base-patch16", backbone_cls=CLIPVisionModel, head=None)
outputs = model(images, output_hidden_states=True)
hidden_states = outputs["last_hidden_state"]  # (batch_size, seq_len, embedding_dim)
cls_embedding = hidden_states[:, 0, :]  # (batch_size, embedding_dim)
predictions = outputs["predictions"]  # same as cls_embedding when head=None (IdentityHead)
```

### Attention Weights

```python
from vit_zoo import ViTModel

model = ViTModel(
    "google/vit-base-patch16-224",
    head=10,
    config_kwargs={"attn_implementation": "eager"}
)
outputs = model(images, output_attentions=True)
attentions = outputs["attentions"]
```

### Custom Head

```python
from vit_zoo import ViTModel, BaseHead
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
model = ViTModel("google/vit-base-patch16-224", head=head)
```

### Multi-modal Models (CLIP)

For CLIP and other multi-modal models, pass `backbone_cls` to load only the vision encoder (AutoModel would load the full model):

```python
from vit_zoo import ViTModel
from transformers import CLIPVisionModel

model = ViTModel("openai/clip-vit-base-patch16", backbone_cls=CLIPVisionModel, head=10)
```

### Any HuggingFace Model

`ViTModel` uses AutoModel to auto-detect the model type from the HuggingFace Hub. Any ViT-compatible model works:

```python
from vit_zoo import ViTModel

model = ViTModel("google/vit-large-patch16-224", head=10)
model = ViTModel("facebook/deit-base-distilled-patch16-224", head=10)
model = ViTModel("facebook/dinov2-with-registers-base", head=10)
```

## API Reference

### `ViTModel`

Single entry point: construct from a HuggingFace model name or from a pre-built backbone.

```python
ViTModel(
    model_name: Optional[str] = None,
    head: Optional[Union[int, BaseHead]] = None,
    backbone: Optional[nn.Module] = None,
    backbone_cls: Optional[Type] = None,
    freeze_backbone: bool = False,
    load_pretrained: bool = True,
    backbone_dropout: float = 0.0,
    config_kwargs: Optional[Dict[str, Any]] = None,
)
```

**Parameters:**
- `model_name`: HuggingFace model identifier (e.g. `"google/vit-base-patch16-224"`). Required unless `backbone` is provided.
- `head`: `int` (creates LinearHead), `BaseHead` instance, or `None` (embedding extraction).
- `backbone`: Optional pre-built backbone; if set, `model_name` and backbone-loading args are ignored.
- `backbone_cls`: Optional HuggingFace model class (e.g. `CLIPVisionModel`). Use for multi-modal models.
- `freeze_backbone`: Freeze all backbone parameters.
- `load_pretrained`: Load pretrained weights when using `model_name`.
- `backbone_dropout`: Dropout probability in backbone.
- `config_kwargs`: Extra config options (e.g. `{"attn_implementation": "eager"}`).

**Usage:**
- `ViTModel("google/vit-base-patch16-224", head=10)`
- `ViTModel("facebook/dinov2-base", head=None)` (embedding extraction)
- Custom backbone: `backbone = vit_zoo.utils._load_backbone(...); ViTModel(backbone=backbone, head=10)`

### `ViTModel.forward()`

```python
forward(
    pixel_values: torch.Tensor,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
) -> Dict[str, Any]
```

Always returns a dict. Keys:
- `"predictions"`: head output tensor (always present)
- `"attentions"`: optional, when `output_attentions=True`
- `"last_hidden_state"`: optional, when `output_hidden_states=True`; shape `(batch_size, seq_len, embedding_dim)`

### Freezing the backbone

```python
model.freeze_backbone(freeze: bool = True)  # Freeze/unfreeze backbone
```

The backbone is the raw HuggingFace model (e.g., `model.backbone.encoder.layer.11` for ViT), so you can register hooks and access layers directly without an extra wrapper.

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

You can import the public API from the root package or from submodules:

```python
# One-line style (recommended)
from vit_zoo import ViTModel, BaseHead, LinearHead, MLPHead, IdentityHead

# Submodule style (explicit namespaces)
from vit_zoo import ViTModel
from vit_zoo.components import BaseHead, LinearHead, MLPHead, IdentityHead
from vit_zoo.utils import _load_backbone  # for custom backbone path (private)
```

## Architecture

- **Public API** (`vit_zoo.__all__`): `ViTModel`, `BaseHead`, `LinearHead`, `MLPHead`, `IdentityHead`, `get_embedding_dim`.
- **Layout:** `vit_zoo/model.py` (ViT model), `vit_zoo/utils/backbone.py` (`_load_backbone`, get_embedding_dim, `_get_cls_token_embedding`), `vit_zoo/components/` (heads).
- **Extending:** Add new heads in `components`; use `vit_zoo.utils._load_backbone` for custom backbones, then `ViTModel(backbone=..., head=...)`.

## Available Heads

- `LinearHead`: Simple linear layer (auto-created when `head=int`)
- `MLPHead`: Multi-layer perceptron with configurable depth, activation, dropout
- `IdentityHead`: Returns embeddings unchanged

All heads must implement `input_dim` property. Custom heads by subclassing `BaseHead`.

## License

GPL-3.0
