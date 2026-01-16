"""Tests for the backbone module."""

import torch
import pytest
from vit_zoo.components import ViTBackbone
from transformers import ViTModel as HFViTModel, DeiTModel, CLIPVisionModel


class TestViTBackbone:
    """Tests for the ViTBackbone class."""
    
    def test_backbone_initialization_pretrained(self):
        """Test ViTBackbone initialization with pretrained weights."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=True
        )
        
        assert backbone.backbone is not None
        assert hasattr(backbone.backbone, 'config')
    
    def test_backbone_initialization_no_pretrained(self):
        """Test ViTBackbone initialization without pretrained weights."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        
        assert backbone.backbone is not None
        assert hasattr(backbone.backbone, 'config')
    
    def test_backbone_get_embedding_dim(self):
        """Test get_embedding_dim method."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        
        embedding_dim = backbone.get_embedding_dim()
        assert embedding_dim == 768
        assert isinstance(embedding_dim, int)
    
    def test_backbone_get_embedding_dim_different_model(self):
        """Test get_embedding_dim with different model size."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-large-patch16-224",
            load_pretrained=False
        )
        
        embedding_dim = backbone.get_embedding_dim()
        assert embedding_dim == 1024
    
    def test_backbone_get_cls_token_embedding_from_last_hidden_state(self):
        """Test get_cls_token_embedding with last_hidden_state format."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        
        # Simulate backbone output format
        outputs = {
            "last_hidden_state": torch.randn(2, 197, 768)  # (batch, seq_len, hidden_size)
        }
        
        embeddings = backbone.get_cls_token_embedding(outputs)
        
        assert embeddings.shape == (2, 768)
        # CLS token is first token
        assert torch.equal(embeddings, outputs["last_hidden_state"][:, 0, :])
    
    def test_backbone_get_cls_token_embedding_from_pooler_output(self):
        """Test get_cls_token_embedding with pooler_output format."""
        backbone = ViTBackbone(
            backbone_cls=CLIPVisionModel,
            model_name="openai/clip-vit-base-patch16",
            load_pretrained=False
        )
        
        # Simulate backbone output format with pooler_output
        outputs = {
            "pooler_output": torch.randn(3, 768)
        }
        
        embeddings = backbone.get_cls_token_embedding(outputs)
        
        assert embeddings.shape == (3, 768)
        assert torch.equal(embeddings, outputs["pooler_output"])
    
    def test_backbone_get_cls_token_embedding_prefers_pooler_output(self):
        """Test that pooler_output is preferred over last_hidden_state."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        
        # Both present, pooler_output should be used
        pooler = torch.randn(2, 768)
        outputs = {
            "pooler_output": pooler,
            "last_hidden_state": torch.randn(2, 197, 768)
        }
        
        embeddings = backbone.get_cls_token_embedding(outputs)
        
        assert torch.equal(embeddings, pooler)
    
    def test_backbone_get_cls_token_embedding_missing_outputs(self):
        """Test get_cls_token_embedding raises error when outputs are missing."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        
        outputs = {}
        
        with pytest.raises(ValueError, match="Backbone output must contain"):
            backbone.get_cls_token_embedding(outputs)
    
    def test_backbone_get_cls_token_embedding_none_pooler(self):
        """Test get_cls_token_embedding handles None pooler_output."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        
        outputs = {
            "pooler_output": None,
            "last_hidden_state": torch.randn(2, 197, 768)
        }
        
        embeddings = backbone.get_cls_token_embedding(outputs)
        
        # Should fall back to last_hidden_state
        assert embeddings.shape == (2, 768)
    
    def test_backbone_forward(self):
        """Test backbone forward pass."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        
        pixel_values = torch.randn(2, 3, 224, 224)
        outputs = backbone(pixel_values)
        
        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (2, 197, 768)
    
    def test_backbone_forward_with_attentions(self):
        """Test backbone forward pass with attention weights."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False,
            config_kwargs={"attn_implementation": "eager"}
        )
        
        pixel_values = torch.randn(1, 3, 224, 224)
        outputs = backbone(pixel_values, output_attentions=True)
        
        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        # Attentions may be None or a tuple
        assert "attentions" in outputs
    
    def test_backbone_forward_with_hidden_states(self):
        """Test backbone forward pass with hidden states."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        
        pixel_values = torch.randn(2, 3, 224, 224)
        outputs = backbone(pixel_values, output_hidden_states=True)
        
        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        # Hidden states may be None or a tuple
        assert "hidden_states" in outputs
    
    def test_backbone_forward_different_batch_sizes(self):
        """Test backbone forward pass with different batch sizes."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        
        for batch_size in [1, 2, 4]:
            pixel_values = torch.randn(batch_size, 3, 224, 224)
            outputs = backbone(pixel_values)
            
            assert outputs["last_hidden_state"].shape[0] == batch_size
    
    def test_backbone_with_dropout(self):
        """Test backbone with dropout applied."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False,
            backbone_dropout=0.5
        )
        
        # Check that dropout was applied (we can't easily verify the value,
        # but we can check the model was created successfully)
        assert backbone.backbone is not None
        
        pixel_values = torch.randn(1, 3, 224, 224)
        outputs = backbone(pixel_values)
        assert "last_hidden_state" in outputs
    
    def test_backbone_without_dropout(self):
        """Test backbone without dropout (default)."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False,
            backbone_dropout=0.0
        )
        
        assert backbone.backbone is not None
        
        pixel_values = torch.randn(1, 3, 224, 224)
        outputs = backbone(pixel_values)
        assert "last_hidden_state" in outputs
    
    def test_backbone_with_config_kwargs(self):
        """Test backbone with config_kwargs."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False,
            config_kwargs={"hidden_dropout_prob": 0.1}
        )
        
        assert backbone.backbone is not None
        assert backbone.backbone.config.hidden_dropout_prob == 0.1
    
    def test_backbone_different_model_types(self):
        """Test backbone with different HuggingFace model types."""
        # Test with DeiT
        deit_backbone = ViTBackbone(
            backbone_cls=DeiTModel,
            model_name="facebook/deit-base-distilled-patch16-224",
            load_pretrained=False
        )
        
        pixel_values = torch.randn(1, 3, 224, 224)
        outputs = deit_backbone(pixel_values)
        assert "last_hidden_state" in outputs
    
    def test_backbone_gradient_flow(self):
        """Test that gradients flow through backbone."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        
        pixel_values = torch.randn(2, 3, 224, 224, requires_grad=True)
        outputs = backbone(pixel_values)
        loss = outputs["last_hidden_state"].sum()
        loss.backward()
        
        assert pixel_values.grad is not None
        # Check that at least some parameters have gradients
        has_grad = any(p.grad is not None for p in backbone.parameters())
        assert has_grad
    
    def test_backbone_eval_mode(self):
        """Test backbone in eval mode."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        
        backbone.eval()
        pixel_values = torch.randn(1, 3, 224, 224)
        outputs = backbone(pixel_values)
        
        assert "last_hidden_state" in outputs
        assert not backbone.training
    
    def test_backbone_train_mode(self):
        """Test backbone in train mode."""
        backbone = ViTBackbone(
            backbone_cls=HFViTModel,
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        
        backbone.train()
        pixel_values = torch.randn(1, 3, 224, 224)
        outputs = backbone(pixel_values)
        
        assert "last_hidden_state" in outputs
        assert backbone.training
