"""Tests for the utils.head module."""

import pytest
from vit_zoo.utils import _validate_head_for_backbone, _create_linear_head
from vit_zoo.components import LinearHead, MLPHead


class TestValidateHeadForBackbone:
    """Tests for _validate_head_for_backbone function."""

    def test_validate_head_matching_dim(self):
        """Test _validate_head_for_backbone passes when dimensions match."""
        head = LinearHead(input_dim=768, output_dim=10)
        _validate_head_for_backbone(head, 768)  # no raise

    def test_validate_head_raises_when_mismatch(self):
        """Test _validate_head_for_backbone raises when dimensions mismatch."""
        head = LinearHead(input_dim=512, output_dim=10)
        with pytest.raises(ValueError, match="Head input dimension.*does not match"):
            _validate_head_for_backbone(head, 768)

    def test_validate_head_error_message_includes_dims(self):
        """Test error message includes head and backbone dimensions."""
        head = LinearHead(input_dim=256, output_dim=5)
        with pytest.raises(ValueError, match="256.*768") as exc_info:
            _validate_head_for_backbone(head, 768)
        assert "256" in str(exc_info.value)
        assert "768" in str(exc_info.value)

    def test_validate_head_with_mlp_head(self):
        """Test _validate_head_for_backbone with MLPHead."""
        head = MLPHead(input_dim=768, hidden_dims=[256], output_dim=10)
        _validate_head_for_backbone(head, 768)  # no raise

        with pytest.raises(ValueError, match="Head input dimension.*does not match"):
            _validate_head_for_backbone(head, 512)


class TestCreateLinearHead:
    """Tests for _create_linear_head function."""

    def test_create_linear_head_returns_linear_head(self):
        """Test _create_linear_head returns a LinearHead instance."""
        head = _create_linear_head(output_dim=10, backbone_embedding_dim=768)
        assert isinstance(head, LinearHead)

    def test_create_linear_head_input_output_dims(self):
        """Test _create_linear_head uses correct input and output dimensions."""
        head = _create_linear_head(output_dim=5, backbone_embedding_dim=768)
        assert head.input_dim == 768
        assert head.linear.in_features == 768
        assert head.linear.out_features == 5

    def test_create_linear_head_different_dims(self):
        """Test _create_linear_head with different dimensions."""
        head = _create_linear_head(output_dim=100, backbone_embedding_dim=1024)
        assert head.input_dim == 1024
        assert head.linear.out_features == 100
