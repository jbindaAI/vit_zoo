"""Tests for ViTModel initialization from model_name (replaces build_model tests)."""

import torch
import pytest
from transformers import CLIPVisionModel

from vit_zoo import ViTModel
from vit_zoo.components import LinearHead, MLPHead, IdentityHead, BaseHead


def test_vit_model_simple():
    """Test ViTModel with linear head from model_name."""
    model = ViTModel("google/vit-base-patch16-224", head=2, load_pretrained=False)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out["predictions"].shape == (1, 2)


def test_vit_model_no_head():
    """Test ViTModel without head (embedding extraction)."""
    model = ViTModel("google/vit-base-patch16-224", head=None, load_pretrained=False)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert len(out["predictions"].shape) == 2
    assert out["predictions"].shape[0] == 1


def test_vit_model_mlp_head():
    """Test ViTModel with MLP head."""
    import torch.nn as nn

    mlp_head = MLPHead(
        input_dim=768,
        hidden_dims=[512, 256],
        output_dim=10,
        activation="gelu",
        dropout=0.1,
    )
    model = ViTModel("google/vit-base-patch16-224", head=mlp_head, load_pretrained=False)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out["predictions"].shape == (1, 10)

    mlp_head_custom = MLPHead(
        input_dim=768,
        hidden_dims=[512],
        output_dim=10,
        activation=nn.SiLU(),
    )
    model2 = ViTModel("google/vit-base-patch16-224", head=mlp_head_custom, load_pretrained=False)
    out2 = model2(dummy)
    assert out2["predictions"].shape == (1, 10)


def test_vit_model_attention_weights():
    """Test extracting attention weights."""
    model = ViTModel(
        "google/vit-base-patch16-224",
        head=10,
        load_pretrained=False,
        config_kwargs={"attn_implementation": "eager"},
    )
    dummy = torch.rand(1, 3, 224, 224)
    outputs = model(dummy, output_attentions=True)
    assert isinstance(outputs, dict)
    assert "predictions" in outputs
    assert "attentions" in outputs
    assert outputs["predictions"].shape == (1, 10)
    assert outputs["attentions"] is None or isinstance(outputs["attentions"], tuple)


def test_vit_model_embeddings():
    """Test extracting embeddings."""
    model = ViTModel("google/vit-base-patch16-224", head=10, load_pretrained=False)
    dummy = torch.rand(1, 3, 224, 224)
    outputs = model(dummy, output_hidden_states=True)
    assert isinstance(outputs, dict)
    assert "predictions" in outputs
    assert "last_hidden_state" in outputs
    assert outputs["predictions"].shape == (1, 10)
    assert len(outputs["last_hidden_state"].shape) == 3
    assert outputs["last_hidden_state"].shape[0] == 1


def test_vit_model_freeze_backbone():
    """Test freezing backbone via init."""
    model = ViTModel(
        "google/vit-base-patch16-224",
        head=10,
        load_pretrained=False,
        freeze_backbone=True,
    )
    for param in model.backbone.parameters():
        assert not param.requires_grad
    for param in model.head.parameters():
        assert param.requires_grad


def test_vit_model_custom_head():
    """Test ViTModel with custom head instance."""
    head = LinearHead(input_dim=768, output_dim=5)
    model = ViTModel("google/vit-base-patch16-224", head=head, load_pretrained=False)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out["predictions"].shape == (1, 5)


def test_vit_model_deit():
    """Test DeiT model."""
    model = ViTModel("facebook/deit-base-distilled-patch16-224", head=10, load_pretrained=False)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out["predictions"].shape == (1, 10)


def test_vit_model_dinov2():
    """Test DINOv2 model."""
    model = ViTModel("facebook/dinov2-base", head=10, load_pretrained=False)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out["predictions"].shape == (1, 10)


def test_vit_model_large_variant():
    """Test ViT large variant."""
    model = ViTModel("google/vit-large-patch16-224", head=10, load_pretrained=False)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out["predictions"].shape == (1, 10)
    assert model.embedding_dim == 1024


def test_vit_model_dinov2_registers():
    """Test DINOv2 with registers."""
    model = ViTModel("facebook/dinov2-with-registers-base", head=10, load_pretrained=False)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out["predictions"].shape == (1, 10)


def test_vit_model_clip():
    """Test CLIP vision model."""
    model = ViTModel(
        "openai/clip-vit-base-patch16",
        backbone_cls=CLIPVisionModel,
        head=10,
        load_pretrained=False,
    )
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out["predictions"].shape == (1, 10)


def test_vit_model_custom_head_subclass():
    """Test custom head subclass."""
    import torch.nn as nn

    class SimpleCustomHead(BaseHead):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self._input_dim = input_dim
            self.fc = nn.Linear(input_dim, output_dim)

        @property
        def input_dim(self) -> int:
            return self._input_dim

        def forward(self, embeddings):
            return self.fc(embeddings)

    custom_head = SimpleCustomHead(input_dim=768, output_dim=5)
    model = ViTModel("google/vit-base-patch16-224", head=custom_head, load_pretrained=False)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out["predictions"].shape == (1, 5)


def test_vit_model_head_input_dim_validation():
    """Test that head input dimension is validated."""
    wrong_head = LinearHead(input_dim=512, output_dim=10)
    with pytest.raises(ValueError, match="Head input dimension.*does not match"):
        ViTModel("google/vit-base-patch16-224", head=wrong_head, load_pretrained=False)

    correct_head = LinearHead(input_dim=768, output_dim=10)
    model = ViTModel("google/vit-base-patch16-224", head=correct_head, load_pretrained=False)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out["predictions"].shape == (1, 10)
