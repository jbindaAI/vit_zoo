"""Tests for the model module."""

import torch
import pytest
from vit_zoo import ViTModel, get_embedding_dim
from vit_zoo.utils import _get_cls_token_embedding
from vit_zoo.utils import _load_backbone
from vit_zoo.components import LinearHead, MLPHead, IdentityHead


class TestViTModel:
    """Tests for the ViTModel class."""

    def test_vit_model_init_from_model_name(self):
        """Test ViTModel initialization from model_name (single entry point)."""
        model = ViTModel(
            model_name="google/vit-base-patch16-224",
            head=10,
            load_pretrained=False,
        )
        assert model.embedding_dim == 768
        pixel_values = torch.randn(2, 3, 224, 224)
        out = model(pixel_values)
        assert out["predictions"].shape == (2, 10)

    def test_vit_model_init_requires_model_name_or_backbone(self):
        """Test ViTModel raises when neither model_name nor backbone provided."""
        with pytest.raises(ValueError, match="Either model_name or backbone must be provided"):
            ViTModel(head=10)
    
    def test_vit_model_initialization_with_head(self):
        """Test ViTModel initialization with a head."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        head = LinearHead(input_dim=768, output_dim=10)
        model = ViTModel(backbone=backbone, head=head)
        
        assert model.backbone is backbone
        assert model.head == head
        assert model.embedding_dim == 768
    
    def test_vit_model_initialization_without_head(self):
        """Test ViTModel initialization without head (uses IdentityHead)."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        model = ViTModel(backbone=backbone, head=None)
        
        assert model.backbone is backbone
        assert isinstance(model.head, IdentityHead)
        assert model.head.input_dim == 768
    
    def test_vit_model_initialization_with_freeze_backbone(self):
        """Test ViTModel with frozen backbone (via model.freeze_backbone)."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        head = LinearHead(input_dim=768, output_dim=5)
        model = ViTModel(backbone=backbone, head=head)
        model.freeze_backbone(freeze=True)
        
        # Check that backbone parameters are frozen
        for param in model.backbone.parameters():
            assert not param.requires_grad
        
        # Check that head parameters are not frozen
        for param in model.head.parameters():
            assert param.requires_grad
    
    def test_vit_model_embedding_dim_property(self):
        """Test embedding_dim property."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        model = ViTModel(backbone=backbone, head=None)
        
        assert model.embedding_dim == 768
        assert model.embedding_dim == get_embedding_dim(backbone)
    
    def test_vit_model_forward_simple(self):
        """Test ViTModel forward pass with simple output."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        head = LinearHead(input_dim=768, output_dim=10)
        model = ViTModel(backbone=backbone, head=head)
        
        pixel_values = torch.randn(2, 3, 224, 224)
        output = model(pixel_values)
        
        assert isinstance(output, dict)
        assert output["predictions"].shape == (2, 10)
    
    def test_vit_model_forward_with_embeddings(self):
        """Test ViTModel forward pass with embeddings output."""
        # Model exposes token embeddings under 'last_hidden_state' for consistency with backbone.
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        head = LinearHead(input_dim=768, output_dim=5)
        model = ViTModel(backbone=backbone, head=head)
        
        model.eval()
        pixel_values = torch.randn(3, 3, 224, 224)
        with torch.no_grad():
            outputs = model(pixel_values, output_hidden_states=True)
        
        assert isinstance(outputs, dict)
        assert "predictions" in outputs
        assert "last_hidden_state" in outputs
        assert outputs["predictions"].shape == (3, 5)
        assert outputs["last_hidden_state"].shape == (3, 197, 768)
        
        # Predictions should be computed from the backbone's CLS embedding
        with torch.no_grad():
            backbone_outputs = model.backbone(pixel_values, output_attentions=False, output_hidden_states=False)
            cls_embedding = _get_cls_token_embedding(backbone_outputs)
            torch.testing.assert_close(outputs["predictions"], model.head(cls_embedding))
    
    def test_vit_model_forward_with_attentions(self):
        """Test ViTModel forward pass with attention weights."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False,
            config_kwargs={"attn_implementation": "eager"}
        )
        head = LinearHead(input_dim=768, output_dim=5)
        model = ViTModel(backbone=backbone, head=head)
        
        pixel_values = torch.randn(2, 3, 224, 224)
        outputs = model(pixel_values, output_attentions=True)
        
        assert isinstance(outputs, dict)
        assert "predictions" in outputs
        assert "attentions" in outputs
        assert outputs["predictions"].shape == (2, 5)
        # Attentions may be None if not supported, or a tuple if available
        assert outputs["attentions"] is None or isinstance(outputs["attentions"], tuple)
    
    def test_vit_model_forward_with_attentions_and_embeddings(self):
        """Test ViTModel forward pass with both attention weights and embeddings."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False,
            config_kwargs={"attn_implementation": "eager"}
        )
        head = LinearHead(input_dim=768, output_dim=10)
        model = ViTModel(backbone=backbone, head=head)
        
        model.eval()
        pixel_values = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            outputs = model(pixel_values, output_attentions=True, output_hidden_states=True)
        
        assert isinstance(outputs, dict)
        assert "predictions" in outputs
        assert "attentions" in outputs
        assert "last_hidden_state" in outputs
        assert outputs["predictions"].shape == (1, 10)
        assert outputs["last_hidden_state"].shape == (1, 197, 768)
        
        # Predictions should be computed from the backbone's CLS embedding
        with torch.no_grad():
            backbone_outputs = model.backbone(pixel_values, output_attentions=True, output_hidden_states=False)
            cls_embedding = _get_cls_token_embedding(backbone_outputs)
            torch.testing.assert_close(outputs["predictions"], model.head(cls_embedding))
    
    def test_vit_model_forward_different_batch_sizes(self):
        """Test ViTModel forward pass with different batch sizes."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        head = LinearHead(input_dim=768, output_dim=5)
        model = ViTModel(backbone=backbone, head=head)
        
        for batch_size in [1, 2, 4, 8]:
            pixel_values = torch.randn(batch_size, 3, 224, 224)
            output = model(pixel_values)
            assert output["predictions"].shape == (batch_size, 5)
    
    def test_vit_model_freeze_backbone(self):
        """Test model.freeze_backbone method."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        head = LinearHead(input_dim=768, output_dim=10)
        model = ViTModel(backbone=backbone, head=head)
        
        # Initially parameters should be trainable
        for param in model.backbone.parameters():
            assert param.requires_grad
        
        # Freeze backbone
        model.freeze_backbone(freeze=True)
        for param in model.backbone.parameters():
            assert not param.requires_grad
        
        # Unfreeze backbone
        model.freeze_backbone(freeze=False)
        for param in model.backbone.parameters():
            assert param.requires_grad
    
    def test_vit_model_freeze_backbone_default(self):
        """Test model.freeze_backbone with default parameter (freeze=True)."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        head = LinearHead(input_dim=768, output_dim=10)
        model = ViTModel(backbone=backbone, head=head)
        
        # Freeze with default
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad
    
    def test_vit_model_with_mlp_head(self):
        """Test ViTModel with MLP head."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        head = MLPHead(input_dim=768, hidden_dims=[256, 128], output_dim=20)
        model = ViTModel(backbone=backbone, head=head)
        
        pixel_values = torch.randn(2, 3, 224, 224)
        output = model(pixel_values)
        
        assert output["predictions"].shape == (2, 20)
    
    def test_vit_model_with_identity_head(self):
        """Test ViTModel with IdentityHead (embedding extraction)."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        head = IdentityHead(input_dim=768)
        model = ViTModel(backbone=backbone, head=head)
        
        pixel_values = torch.randn(3, 3, 224, 224)
        output = model(pixel_values)
        
        # Predictions are CLS embeddings (IdentityHead passes through)
        assert output["predictions"].shape == (3, 768)
    
    def test_vit_model_gradient_flow(self):
        """Test that gradients flow through ViTModel."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        head = LinearHead(input_dim=768, output_dim=5)
        model = ViTModel(backbone=backbone, head=head)
        
        pixel_values = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(pixel_values)
        loss = output["predictions"].sum()
        loss.backward()
        
        assert pixel_values.grad is not None
        # Check that at least some parameters have gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad
    
    def test_vit_model_eval_mode(self):
        """Test ViTModel in eval mode."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        head = LinearHead(input_dim=768, output_dim=10)
        model = ViTModel(backbone=backbone, head=head)
        
        model.eval()
        pixel_values = torch.randn(2, 3, 224, 224)
        output = model(pixel_values)
        
        assert output["predictions"].shape == (2, 10)
        assert not model.training
    
    def test_vit_model_train_mode(self):
        """Test ViTModel in train mode."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        head = LinearHead(input_dim=768, output_dim=10)
        model = ViTModel(backbone=backbone, head=head)
        
        model.train()
        pixel_values = torch.randn(2, 3, 224, 224)
        output = model(pixel_values)
        
        assert output["predictions"].shape == (2, 10)
        assert model.training
