"""Tests for the factory module."""

import torch
import pytest
from vit_zoo import ViTModel
from vit_zoo.factory import build_model
from vit_zoo.components import LinearHead, MLPHead, IdentityHead
from transformers import ViTModel as HFViTModel


def test_build_model_simple():
    """Test building a simple model with linear head."""
    model = build_model("vanilla_vit", head=2)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (1, 2)


def test_build_model_no_head():
    """Test building a model without head (embedding extraction)."""
    model = build_model("vanilla_vit", head=None)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    # Should return embeddings (batch_size, embedding_dim)
    assert len(out.shape) == 2
    assert out.shape[0] == 1


def test_build_model_mlp_head():
    """Test building a model with MLP head."""
    from vit_zoo.components import MLPHead
    import torch.nn as nn
    
    # Test with string activation
    mlp_head = MLPHead(
        input_dim=768,  # vanilla_vit embedding dim
        hidden_dims=[512, 256],
        output_dim=10,
        activation="gelu",
        dropout=0.1
    )
    model = build_model("vanilla_vit", head=mlp_head)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (1, 10)
    
    # Test with custom nn.Module activation
    mlp_head_custom = MLPHead(
        input_dim=768,
        hidden_dims=[512],
        output_dim=10,
        activation=nn.SiLU()  # Custom activation module
    )
    model2 = build_model("vanilla_vit", head=mlp_head_custom)
    out2 = model2(dummy)
    assert out2.shape == (1, 10)


def test_build_model_attention_weights():
    """Test extracting attention weights."""
    model = build_model("vanilla_vit", head=10, config_kwargs={"attn_implementation": "eager"})
    dummy = torch.rand(1, 3, 224, 224)
    outputs = model(dummy, output_attentions=True)
    
    assert isinstance(outputs, dict)
    assert "predictions" in outputs
    assert "attentions" in outputs
    assert outputs["predictions"].shape == (1, 10)
    
    assert outputs["attentions"] is not None
    assert isinstance(outputs["attentions"], tuple)
    assert len(outputs["attentions"]) > 0


def test_build_model_embeddings():
    """Test extracting embeddings."""
    model = build_model("vanilla_vit", head=10)
    dummy = torch.rand(1, 3, 224, 224)
    outputs = model(dummy, output_embeddings=True)
    
    assert isinstance(outputs, dict)
    assert "predictions" in outputs
    assert "embeddings" in outputs
    assert outputs["predictions"].shape == (1, 10)
    assert len(outputs["embeddings"].shape) == 3
    assert outputs["embeddings"].shape[0] == 1


def test_build_model_freeze_backbone():
    """Test freezing backbone."""
    model = build_model("vanilla_vit", head=10, freeze_backbone=True)
    
    # Check that backbone parameters are frozen
    for param in model.backbone.parameters():
        assert not param.requires_grad
    
    # Check that head parameters are not frozen
    for param in model.head.parameters():
        assert param.requires_grad


def test_build_model_custom_head():
    """Test building model with custom head instance."""
    head = LinearHead(input_dim=768, output_dim=5)
    model = build_model("vanilla_vit", head=head)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (1, 5)


def test_list_models():
    """Test listing available models."""
    from vit_zoo.factory import list_models
    models = list_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "vanilla_vit" in models


def test_invalid_model_type():
    """Test error handling for invalid model type."""
    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_model("invalid_model", head=10)


def test_deit_model():
    """Test DeiT model creation."""
    model = build_model("deit_vit", head=10)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (1, 10)


def test_dinov2_model():
    """Test DINOv2 model creation."""
    model = build_model("dinov2_vit", head=10)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (1, 10)


def test_build_model_override_model_name():
    """Test overriding default model name from registry."""
    # Use a different ViT variant
    model = build_model("vanilla_vit", model_name="google/vit-large-patch16-224", head=10)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (1, 10)
    # Large model has different embedding dim
    assert model.embedding_dim == 1024


def test_build_model_direct_usage():
    """Test building model directly without registry."""
    model = build_model(
        model_name="google/vit-base-patch16-224",
        backbone_cls=HFViTModel,
        head=10
    )
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (1, 10)


def test_build_model_direct_usage_missing_args():
    """Test error when using direct usage without required args."""
    with pytest.raises(ValueError, match="To use a non-default ViT backbone"):
        build_model(model_name="google/vit-base-patch16-224", head=10)
    
    with pytest.raises(ValueError, match="To use a non-default ViT backbone"):
        build_model(backbone_cls=HFViTModel, head=10)


def test_dinov2_reg_vit():
    """Test DINOv2 with registers."""
    model = build_model("dinov2_reg_vit", head=10)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (1, 10)


def test_custom_head_subclass():
    """Test using a custom head subclass."""
    from vit_zoo.components import BaseHead
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
    model = build_model("vanilla_vit", head=custom_head)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (1, 5)


def test_head_input_dim_validation():
    """Test that head input dimension is validated."""
    # Create head with wrong input dimension
    wrong_head = LinearHead(input_dim=512, output_dim=10)  # vanilla_vit has 768
    
    with pytest.raises(ValueError, match="Head input dimension.*does not match"):
        build_model("vanilla_vit", head=wrong_head)
    
    # Create head with correct input dimension
    correct_head = LinearHead(input_dim=768, output_dim=10)
    model = build_model("vanilla_vit", head=correct_head)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (1, 10)
