# Vision Transformer Zoo

A clean, extensible, and reusable factory for creating HuggingFace-based Vision Transformer models - including **ViT**, **DeiT**, **DINO**, **DINOv2**, **DINOv3**, and **CLIP Vision** — with flexible heads, easy backbone freezing, attention weight extraction, and seamless integration with PyTorch Lightning.

---

## Features

- **Easy model construction** via `build_model(...)` - create models in minutes
- **Flexible head support** - Linear, MLP, or custom heads
- **Common interface** for different ViT flavors from HuggingFace
- **Backbone freezing** - full or layer-wise freezing support
- **Attention weights** - easy extraction of attention weights
- **Embedding extraction** - get embeddings without classification head
- **PyTorch Lightning ready** - works seamlessly with Lightning modules
- **Model registry** - easy extensibility for new models

---

## Installation

### Local development install

```bash
git clone git@github.com:jbindaAI/vit_zoo.git
cd vit_zoo
pip install -e .
```

---

## Quick Start

### Example 1: Simple Classification (Lightning-ready)

```python
from vit_zoo import build_model

# Create a model with 10 classes, freeze backbone
model = build_model("vanilla_vit", head=10, freeze_backbone=True)

# Use in PyTorch Lightning
import pytorch_lightning as pl

class ViTLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = build_model("vanilla_vit", head=10, freeze_backbone=True)
    
    def forward(self, x):
        return self.model(x)
```

### Example 2: Custom MLP Head

```python
from vit_zoo import build_model, MLPHead
import torch.nn as nn

# Create MLP head with string activation (most common)
mlp_head = MLPHead(
    input_dim=768,  # Must match backbone embedding dimension
    hidden_dims=[512, 256],
    output_dim=100,
    dropout=0.1,
    activation="gelu"  # String literal: 'relu', 'gelu', or 'tanh'
)

# Or use custom nn.Module activation
mlp_head_custom = MLPHead(
    input_dim=768,
    hidden_dims=[512, 256],
    output_dim=100,
    activation=nn.SiLU()  # Any PyTorch activation module
)

model = build_model("dino_v2_vit", head=mlp_head)
```

### Example 3: Embedding Extraction Only

```python
from vit_zoo import build_model

# No head - just extract embeddings
model = build_model("clip_vit", head=None)

outputs = model(images, output_embeddings=True)
embeddings = outputs["embeddings"]  # Shape: (batch_size, embedding_dim)
```

### Example 4: Attention Weights

```python
from vit_zoo import build_model

# For attention weights, you may need to set attn_implementation='eager'
model = build_model(
    "vanilla_vit",
    head=10,
    config_kwargs={"attn_implementation": "eager"}
)
outputs = model(images, output_attentions=True)

predictions = outputs["predictions"]  # Shape: (batch_size, num_classes)
attentions = outputs["attentions"]     # Tuple of attention tensors (may be None if not supported)
```

### Example 5: Custom Head Class

You can subclass `BaseHead` to create any custom head architecture:

```python
from vit_zoo import build_model, BaseHead
import torch.nn as nn
import torch

class MyCustomHead(BaseHead):
    """Custom head - can be MLP, UNET decoder, attention-based, etc."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self._input_dim = input_dim  # Store for input_dim property
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    @property
    def input_dim(self) -> int:
        """Returns the input dimension of the head."""
        return self._input_dim
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(embeddings))
        return self.fc2(x)

# Use custom head - input_dim will be validated automatically
head = MyCustomHead(input_dim=768, num_classes=10)  # vanilla_vit has 768-dim embeddings
model = build_model("vanilla_vit", head=head)  # Validates input_dim matches
```

**Important:** All custom heads must implement the `input_dim` property. The factory will automatically validate that the head's `input_dim` matches the backbone's embedding dimension, helping catch dimension mismatches early.

### Example 6: Override Model Name

```python
from vit_zoo import build_model

# Use a different model variant from the registry default
model = build_model(
    "vanilla_vit",
    model_name="google/vit-large-patch16-224",  # Override default
    head=10
)
```

### Example 7: Direct Usage (Any HuggingFace Model)

```python
from vit_zoo import build_model
from transformers import ViTModel

# Use any HuggingFace model directly without registry
model = build_model(
    model_name="google/vit-base-patch16-224",
    backbone_cls=ViTModel,
    head=10
)
```

### Example 8: Layer-wise Freezing

```python
from vit_zoo import build_model

# Freeze first 6 layers, train rest
model = build_model(
    "vanilla_vit",
    head=10,
    freeze_layers=[0, 1, 2, 3, 4, 5]  # Freeze layers 0-5
)

# Or freeze/unfreeze programmatically
model.freeze_layers([6, 7, 8], freeze=True)   # Freeze layers 6-8
model.freeze_backbone(freeze=False)          # Unfreeze entire backbone
```

---

## API Reference

### `build_model()`

Main factory function to create ViT models.

```python
build_model(
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    backbone_cls: Optional[Type[ViTBackboneProtocol]] = None,
    head: Optional[Union[int, BaseHead]] = None,
    freeze_backbone: bool = False,
    freeze_layers: Optional[List[int]] = None,
    load_pretrained: bool = True,
    backbone_dropout: float = 0.0,
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> ViTModel
```

**Parameters:**
- `model_type`: Optional registered model type (`"vanilla_vit"`, `"deit_vit"`, `"dino_vit"`, `"dino_v2_vit"`, `"dinov2_reg_vit"`, `"dinov3_vit"`, `"clip_vit"`). If provided, uses default backbone class and model name from registry. When `model_type` is provided, `backbone_cls` is ignored and cannot be overridden.
- `model_name`: Optional HuggingFace model identifier. If `model_type` is provided, this overrides the default model name from registry. If `model_type` is not provided, this is required along with `backbone_cls`.
- `backbone_cls`: Optional HuggingFace model class. Required if `model_type` is not provided. Ignored if `model_type` is provided (registry default is always used).
- `head`: 
  - `int`: Creates `LinearHead` with that output dimension
  - `BaseHead`: Uses provided head instance. **Automatically validates** that `head.input_dim`
               matches the backbone's embedding dimension. Users can subclass `BaseHead` to
               create custom heads (e.g., MLP, UNET decoder, attention-based, etc.).
               **All custom heads must implement the `input_dim` property.**
  - `None`: No head (embedding extraction mode)
- `freeze_backbone`: Freeze all backbone parameters
- `freeze_layers`: List of layer indices to freeze (0-indexed)
- `load_pretrained`: Whether to load pretrained weights
- `backbone_dropout`: Dropout probability for backbone
- `config_kwargs`: Extra config options passed to model config or from_pretrained().
                  Can include 'attn_implementation' to control attention mechanism
                  (e.g., 'eager' for attention weights, 'flash_attention_2', 'sdpa').

**Returns:** `ViTModel` instance

**Usage patterns:**
1. **Registry shortcut** (recommended): `build_model("vanilla_vit", head=10)`
2. **Override default**: `build_model("vanilla_vit", model_name="google/vit-large-patch16-224", head=10)`
3. **Direct usage**: `build_model(model_name="custom/model", backbone_cls=CustomModel, head=10)`

### `ViTModel.forward()`

```python
forward(
    pixel_values: torch.Tensor,
    output_attentions: bool = False,
    output_embeddings: bool = False,
) -> Union[torch.Tensor, Dict[str, Any]]
```

**Returns:**
- If `output_attentions=False` and `output_embeddings=False`: predictions tensor
- If `output_attentions=True` or `output_embeddings=True`: dict with keys:
  - `"predictions"`: Model predictions
  - `"attentions"`: Optional tuple of attention tensors
  - `"embeddings"`: Optional embeddings tensor

---

## Supported Models

The following models are available in the registry with sensible defaults:

- **vanilla_vit**: Google ViT (default: `google/vit-base-patch16-224`)
- **deit_vit**: Facebook DeiT (default: `facebook/deit-base-distilled-patch16-224`)
- **dino_vit**: Facebook DINO (default: `facebook/dino-vitb16`)
- **dino_v2_vit**: Facebook DINOv2 without registers (default: `facebook/dinov2-base`)
- **dinov2_reg_vit**: Facebook DINOv2 with registers (default: `facebook/dinov2-with-registers-base`)
- **dinov3_vit**: Facebook DINOv3 (default: `facebook/dinov3-vitb16-pretrain-lvd1689m`)
- **clip_vit**: OpenAI CLIP Vision (default: `openai/clip-vit-base-patch16`)

You can override the default model name or use any HuggingFace model directly (see examples above).

---

## Advanced Usage

### Using Any HuggingFace Model

You can use any HuggingFace Vision Transformer model directly without registering it:

```python
from vit_zoo import build_model
from transformers import ViTModel, DeiTModel

# Use any ViT variant
model = build_model(
    model_name="google/vit-large-patch16-224",
    backbone_cls=ViTModel,
    head=10
)

# Use any DeiT variant
model = build_model(
    model_name="facebook/deit-small-distilled-patch16-224",
    backbone_cls=DeiTModel,
    head=10
)
```

### Adding Models to Registry

To add a model to the registry for convenience, you can modify the `MODEL_REGISTRY` in `src/vit_zoo/factory.py`:

```python
from transformers import YourCustomModel

MODEL_REGISTRY.update({
    "your_model": (YourCustomModel, "your-org/your-model-name"),
})
```

### Available Heads

The library provides several built-in head implementations:

- `LinearHead`: Simple linear transformation (created automatically when you pass an `int`)
- `MLPHead`: Multi-layer perceptron with configurable depth, activation, and dropout
- `IdentityHead`: Returns embeddings unchanged (for embedding extraction)

### Creating Custom Heads

You can create any custom head architecture by subclassing `BaseHead`. This is useful for:
- Complex MLP architectures
- UNET decoders
- Attention-based heads
- Multi-task heads
- Any other custom architecture

**Example:**
```python
from vit_zoo import BaseHead, build_model
import torch.nn as nn
import torch

class UNETDecoderHead(BaseHead):
    """Example: UNET-style decoder head."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self._input_dim = input_dim  # Store for input_dim property
        # Your custom architecture here
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    @property
    def input_dim(self) -> int:
        """Returns the input dimension of the head."""
        return self._input_dim
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.decoder(embeddings)

# Use it - input_dim will be automatically validated
head = UNETDecoderHead(input_dim=768, num_classes=10)
model = build_model("vanilla_vit", head=head)  # Validates input_dim matches
```

**Important:** All custom heads must implement the `input_dim` property. The factory automatically validates that the head's `input_dim` matches the backbone's embedding dimension, helping catch dimension mismatches early.

---

## License

GPL-3.0