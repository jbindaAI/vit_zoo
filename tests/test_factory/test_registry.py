"""Tests for the registry module."""

import pytest
from vit_zoo.factory.registry import MODEL_REGISTRY, list_models
from transformers import (
    ViTModel as HFViTModel,
    DeiTModel,
    Dinov2Model,
    Dinov2WithRegistersModel,
    CLIPVisionModel,
)


class TestModelRegistry:
    """Tests for MODEL_REGISTRY."""
    
    def test_registry_is_dict(self):
        """Test that MODEL_REGISTRY is a dictionary."""
        assert isinstance(MODEL_REGISTRY, dict)
    
    def test_registry_contains_expected_models(self):
        """Test that registry contains expected model types."""
        expected_models = [
            "vanilla_vit",
            "deit_vit",
            "dino_vit",
            "dino_v2_vit",
            "dinov2_reg_vit",
            "clip_vit",
            "dinov3_vit",
        ]
        
        for model_type in expected_models:
            assert model_type in MODEL_REGISTRY, f"{model_type} not in registry"
    
    def test_registry_values_are_tuples(self):
        """Test that registry values are tuples of (backbone_class, model_name)."""
        for model_type, value in MODEL_REGISTRY.items():
            assert isinstance(value, tuple), f"{model_type} value is not a tuple"
            assert len(value) == 2, f"{model_type} tuple should have 2 elements"
            backbone_cls, model_name = value
            assert isinstance(model_name, str), f"{model_type} model_name should be string"
    
    def test_registry_vanilla_vit(self):
        """Test vanilla_vit registry entry."""
        backbone_cls, model_name = MODEL_REGISTRY["vanilla_vit"]
        assert backbone_cls == HFViTModel
        assert model_name == "google/vit-base-patch16-224"
    
    def test_registry_deit_vit(self):
        """Test deit_vit registry entry."""
        backbone_cls, model_name = MODEL_REGISTRY["deit_vit"]
        assert backbone_cls == DeiTModel
        assert model_name == "facebook/deit-base-distilled-patch16-224"
    
    def test_registry_dino_vit(self):
        """Test dino_vit registry entry."""
        backbone_cls, model_name = MODEL_REGISTRY["dino_vit"]
        assert backbone_cls == HFViTModel
        assert model_name == "facebook/dino-vitb16"
    
    def test_registry_dino_v2_vit(self):
        """Test dino_v2_vit registry entry."""
        backbone_cls, model_name = MODEL_REGISTRY["dino_v2_vit"]
        assert backbone_cls == Dinov2Model
        assert model_name == "facebook/dinov2-base"
    
    def test_registry_dinov2_reg_vit(self):
        """Test dinov2_reg_vit registry entry."""
        backbone_cls, model_name = MODEL_REGISTRY["dinov2_reg_vit"]
        assert backbone_cls == Dinov2WithRegistersModel
        assert model_name == "facebook/dinov2-with-registers-base"
    
    def test_registry_clip_vit(self):
        """Test clip_vit registry entry."""
        backbone_cls, model_name = MODEL_REGISTRY["clip_vit"]
        assert backbone_cls == CLIPVisionModel
        assert model_name == "openai/clip-vit-base-patch16"
    
    def test_registry_dinov3_vit(self):
        """Test dinov3_vit registry entry."""
        backbone_cls, model_name = MODEL_REGISTRY["dinov3_vit"]
        assert backbone_cls == HFViTModel
        assert model_name == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    
    def test_registry_backbone_classes_are_callable(self):
        """Test that backbone classes in registry are callable (classes)."""
        for model_type, (backbone_cls, _) in MODEL_REGISTRY.items():
            assert callable(backbone_cls), f"{model_type} backbone_cls should be callable"
            # Check it has from_pretrained method (HuggingFace models have this)
            assert hasattr(backbone_cls, 'from_pretrained'), \
                f"{model_type} backbone_cls should have from_pretrained method"
            assert hasattr(backbone_cls, 'config_class'), \
                f"{model_type} backbone_cls should have config_class attribute"


class TestListModels:
    """Tests for list_models function."""
    
    def test_list_models_returns_list(self):
        """Test that list_models returns a list."""
        models = list_models()
        assert isinstance(models, list)
    
    def test_list_models_contains_expected_models(self):
        """Test that list_models returns expected model types."""
        models = list_models()
        expected_models = [
            "vanilla_vit",
            "deit_vit",
            "dino_vit",
            "dino_v2_vit",
            "dinov2_reg_vit",
            "clip_vit",
            "dinov3_vit",
        ]
        
        for expected in expected_models:
            assert expected in models, f"{expected} not in list_models()"
    
    def test_list_models_matches_registry_keys(self):
        """Test that list_models returns the same keys as MODEL_REGISTRY."""
        models = list_models()
        registry_keys = set(MODEL_REGISTRY.keys())
        models_set = set(models)
        
        assert models_set == registry_keys, \
            "list_models() should return all keys from MODEL_REGISTRY"
    
    def test_list_models_all_strings(self):
        """Test that all items in list_models are strings."""
        models = list_models()
        for model in models:
            assert isinstance(model, str), f"Model type should be string, got {type(model)}"
    
    def test_list_models_no_duplicates(self):
        """Test that list_models has no duplicate entries."""
        models = list_models()
        assert len(models) == len(set(models)), "list_models() should have no duplicates"
    
    def test_list_models_length_matches_registry(self):
        """Test that list_models length matches registry size."""
        models = list_models()
        assert len(models) == len(MODEL_REGISTRY), \
            "list_models() length should match MODEL_REGISTRY size"
    
    def test_list_models_is_sorted(self):
        """Test that list_models returns a list (order may vary, but should be consistent)."""
        models1 = list_models()
        models2 = list_models()
        
        # Should return same order on multiple calls
        assert models1 == models2, "list_models() should return consistent order"
