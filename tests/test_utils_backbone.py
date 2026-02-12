"""Tests for the utils.backbone module."""

import torch
import pytest
from vit_zoo.utils import (
    _load_backbone,
    _get_embedding_dim,
    _get_cls_token_embedding,
)


class TestLoadBackbone:
    """Tests for _load_backbone function."""

    def test_backbone_initialization_pretrained(self):
        """Test _load_backbone with pretrained weights."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=True
        )
        assert backbone is not None
        assert hasattr(backbone, 'config')

    def test_backbone_initialization_no_pretrained(self):
        """Test _load_backbone without pretrained weights."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        assert backbone is not None
        assert hasattr(backbone, 'config')

    def test_backbone_forward(self):
        """Test backbone forward pass."""
        backbone = _load_backbone(
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
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False,
            config_kwargs={"attn_implementation": "eager"}
        )
        pixel_values = torch.randn(1, 3, 224, 224)
        outputs = backbone(pixel_values, output_attentions=True)
        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        assert "attentions" in outputs

    def test_backbone_forward_with_hidden_states(self):
        """Test backbone forward pass with hidden states."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        pixel_values = torch.randn(2, 3, 224, 224)
        outputs = backbone(pixel_values, output_hidden_states=True)
        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        assert "hidden_states" in outputs

    def test_backbone_forward_different_batch_sizes(self):
        """Test backbone forward pass with different batch sizes."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        for batch_size in [1, 2, 4]:
            pixel_values = torch.randn(batch_size, 3, 224, 224)
            outputs = backbone(pixel_values)
            assert outputs["last_hidden_state"].shape[0] == batch_size

    def test_backbone_with_dropout(self):
        """Test _load_backbone with dropout applied."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False,
            backbone_dropout=0.5
        )
        assert backbone is not None
        pixel_values = torch.randn(1, 3, 224, 224)
        outputs = backbone(pixel_values)
        assert "last_hidden_state" in outputs

    def test_backbone_with_config_kwargs(self):
        """Test _load_backbone with config_kwargs."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False,
            config_kwargs={"hidden_dropout_prob": 0.1}
        )
        assert backbone is not None
        assert backbone.config.hidden_dropout_prob == 0.1

    def test_backbone_different_model_types(self):
        """Test _load_backbone with different HuggingFace model types."""
        deit_backbone = _load_backbone(
            model_name="facebook/deit-base-distilled-patch16-224",
            load_pretrained=False
        )
        pixel_values = torch.randn(1, 3, 224, 224)
        outputs = deit_backbone(pixel_values)
        assert "last_hidden_state" in outputs

    def test_backbone_gradient_flow(self):
        """Test that gradients flow through backbone."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        pixel_values = torch.randn(2, 3, 224, 224, requires_grad=True)
        outputs = backbone(pixel_values)
        loss = outputs["last_hidden_state"].sum()
        loss.backward()
        assert pixel_values.grad is not None
        has_grad = any(p.grad is not None for p in backbone.parameters())
        assert has_grad

    def test_backbone_eval_mode(self):
        """Test backbone in eval mode."""
        backbone = _load_backbone(
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
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        backbone.train()
        pixel_values = torch.randn(1, 3, 224, 224)
        outputs = backbone(pixel_values)
        assert "last_hidden_state" in outputs
        assert backbone.training


class TestGetEmbeddingDim:
    """Tests for _get_embedding_dim function."""

    def test_get_embedding_dim_vit_base(self):
        """Test _get_embedding_dim with ViT base."""
        backbone = _load_backbone(
            model_name="google/vit-base-patch16-224",
            load_pretrained=False
        )
        embedding_dim = _get_embedding_dim(backbone)
        assert embedding_dim == 768
        assert isinstance(embedding_dim, int)

    def test_get_embedding_dim_vit_large(self):
        """Test _get_embedding_dim with different model size."""
        backbone = _load_backbone(
            model_name="google/vit-large-patch16-224",
            load_pretrained=False
        )
        embedding_dim = _get_embedding_dim(backbone)
        assert embedding_dim == 1024


class TestGetClsTokenEmbedding:
    """Tests for _get_cls_token_embedding function."""

    def test__get_cls_token_embedding_from_last_hidden_state(self):
        """Test _get_cls_token_embedding with last_hidden_state format."""
        outputs = {
            "last_hidden_state": torch.randn(2, 197, 768)
        }
        embeddings = _get_cls_token_embedding(outputs)
        assert embeddings.shape == (2, 768)
        assert torch.equal(embeddings, outputs["last_hidden_state"][:, 0, :])

    def test__get_cls_token_embedding_from_pooler_output(self):
        """Test _get_cls_token_embedding with pooler_output format."""
        outputs = {
            "pooler_output": torch.randn(3, 768)
        }
        embeddings = _get_cls_token_embedding(outputs)
        assert embeddings.shape == (3, 768)
        assert torch.equal(embeddings, outputs["pooler_output"])

    def test__get_cls_token_embedding_prefers_pooler_output(self):
        """Test that pooler_output is preferred over last_hidden_state."""
        pooler = torch.randn(2, 768)
        outputs = {
            "pooler_output": pooler,
            "last_hidden_state": torch.randn(2, 197, 768)
        }
        embeddings = _get_cls_token_embedding(outputs)
        assert torch.equal(embeddings, pooler)

    def test__get_cls_token_embedding_missing_outputs(self):
        """Test _get_cls_token_embedding raises error when outputs are missing."""
        outputs = {}
        with pytest.raises(ValueError, match="Backbone output must contain"):
            _get_cls_token_embedding(outputs)

    def test__get_cls_token_embedding_none_pooler(self):
        """Test _get_cls_token_embedding handles None pooler_output."""
        outputs = {
            "pooler_output": None,
            "last_hidden_state": torch.randn(2, 197, 768)
        }
        embeddings = _get_cls_token_embedding(outputs)
        assert embeddings.shape == (2, 768)
