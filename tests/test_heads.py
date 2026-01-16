"""Tests for the heads module."""

import torch
import pytest
import torch.nn as nn
from vit_zoo.heads import BaseHead, LinearHead, MLPHead, IdentityHead


class TestBaseHead:
    """Tests for the BaseHead abstract base class."""
    
    def test_base_head_is_abstract(self):
        """Test that BaseHead cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseHead()
    
    def test_base_head_requires_input_dim_property(self):
        """Test that subclasses must implement input_dim property."""
        class IncompleteHead(BaseHead):
            def forward(self, embeddings):
                return embeddings
        
        with pytest.raises(TypeError):
            IncompleteHead()
    
    def test_base_head_requires_forward_method(self):
        """Test that subclasses must implement forward method."""
        class IncompleteHead(BaseHead):
            @property
            def input_dim(self):
                return 768
        
        with pytest.raises(TypeError):
            IncompleteHead()
    
    def test_base_head_subclass_works(self):
        """Test that a proper subclass of BaseHead works."""
        class CustomHead(BaseHead):
            def __init__(self, input_dim: int, output_dim: int):
                super().__init__()
                self._input_dim = input_dim
                self.linear = nn.Linear(input_dim, output_dim)
            
            @property
            def input_dim(self) -> int:
                return self._input_dim
            
            def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
                return self.linear(embeddings)
        
        head = CustomHead(input_dim=768, output_dim=10)
        assert head.input_dim == 768
        embeddings = torch.randn(2, 768)
        output = head(embeddings)
        assert output.shape == (2, 10)


class TestLinearHead:
    """Tests for the LinearHead class."""
    
    def test_linear_head_initialization(self):
        """Test LinearHead initialization."""
        head = LinearHead(input_dim=768, output_dim=10)
        assert head.input_dim == 768
        assert head.linear.in_features == 768
        assert head.linear.out_features == 10
        assert head.linear.bias is not None
    
    def test_linear_head_initialization_without_bias(self):
        """Test LinearHead initialization without bias."""
        head = LinearHead(input_dim=768, output_dim=10, bias=False)
        assert head.input_dim == 768
        assert head.linear.bias is None
    
    def test_linear_head_forward(self):
        """Test LinearHead forward pass."""
        head = LinearHead(input_dim=512, output_dim=5)
        embeddings = torch.randn(3, 512)
        output = head(embeddings)
        
        assert output.shape == (3, 5)
        assert isinstance(output, torch.Tensor)
    
    def test_linear_head_forward_different_batch_sizes(self):
        """Test LinearHead with different batch sizes."""
        head = LinearHead(input_dim=256, output_dim=20)
        
        for batch_size in [1, 5, 32, 128]:
            embeddings = torch.randn(batch_size, 256)
            output = head(embeddings)
            assert output.shape == (batch_size, 20)
    
    def test_linear_head_input_dim_property(self):
        """Test that input_dim property returns correct value."""
        head = LinearHead(input_dim=1024, output_dim=100)
        assert head.input_dim == 1024
    
    def test_linear_head_gradient_flow(self):
        """Test that gradients flow through LinearHead."""
        head = LinearHead(input_dim=128, output_dim=10)
        embeddings = torch.randn(2, 128, requires_grad=True)
        output = head(embeddings)
        loss = output.sum()
        loss.backward()
        
        assert embeddings.grad is not None
        assert head.linear.weight.grad is not None
        if head.linear.bias is not None:
            assert head.linear.bias.grad is not None


class TestMLPHead:
    """Tests for the MLPHead class."""
    
    def test_mlp_head_initialization_basic(self):
        """Test basic MLPHead initialization."""
        head = MLPHead(
            input_dim=768,
            hidden_dims=[512, 256],
            output_dim=10
        )
        assert head.input_dim == 768
    
    def test_mlp_head_forward(self):
        """Test MLPHead forward pass."""
        head = MLPHead(
            input_dim=256,
            hidden_dims=[128, 64],
            output_dim=5
        )
        embeddings = torch.randn(4, 256)
        output = head(embeddings)
        
        assert output.shape == (4, 5)
        assert isinstance(output, torch.Tensor)
    
    def test_mlp_head_string_activation_relu(self):
        """Test MLPHead with ReLU activation (string)."""
        head = MLPHead(
            input_dim=512,
            hidden_dims=[256],
            output_dim=10,
            activation="relu"
        )
        assert isinstance(head.activation, nn.ReLU)
        
        embeddings = torch.randn(2, 512)
        output = head(embeddings)
        assert output.shape == (2, 10)
    
    def test_mlp_head_string_activation_gelu(self):
        """Test MLPHead with GELU activation (string)."""
        head = MLPHead(
            input_dim=512,
            hidden_dims=[256],
            output_dim=10,
            activation="gelu"
        )
        assert isinstance(head.activation, nn.GELU)
    
    def test_mlp_head_string_activation_tanh(self):
        """Test MLPHead with Tanh activation (string)."""
        head = MLPHead(
            input_dim=512,
            hidden_dims=[256],
            output_dim=10,
            activation="tanh"
        )
        assert isinstance(head.activation, nn.Tanh)
    
    def test_mlp_head_string_activation_case_insensitive(self):
        """Test MLPHead activation string is case insensitive."""
        head1 = MLPHead(
            input_dim=512,
            hidden_dims=[256],
            output_dim=10,
            activation="RELU"
        )
        head2 = MLPHead(
            input_dim=512,
            hidden_dims=[256],
            output_dim=10,
            activation="relu"
        )
        assert type(head1.activation) == type(head2.activation)
    
    def test_mlp_head_custom_activation_module(self):
        """Test MLPHead with custom nn.Module activation."""
        head = MLPHead(
            input_dim=512,
            hidden_dims=[256],
            output_dim=10,
            activation=nn.SiLU()
        )
        assert isinstance(head.activation, nn.SiLU)
        
        embeddings = torch.randn(2, 512)
        output = head(embeddings)
        assert output.shape == (2, 10)
    
    def test_mlp_head_invalid_activation_string(self):
        """Test MLPHead raises error for invalid activation string."""
        with pytest.raises(ValueError, match="Unknown activation"):
            MLPHead(
                input_dim=512,
                hidden_dims=[256],
                output_dim=10,
                activation="invalid_activation"
            )
    
    def test_mlp_head_empty_hidden_dims(self):
        """Test MLPHead raises error for empty hidden_dims."""
        with pytest.raises(ValueError, match="hidden_dims must be a non-empty list"):
            MLPHead(
                input_dim=512,
                hidden_dims=[],
                output_dim=10
            )
    
    def test_mlp_head_with_dropout(self):
        """Test MLPHead with dropout."""
        head = MLPHead(
            input_dim=256,
            hidden_dims=[128, 64],
            output_dim=10,
            dropout=0.5
        )
        
        # Count dropout layers (should be 2, one after each hidden layer)
        dropout_count = sum(1 for module in head.mlp if isinstance(module, nn.Dropout))
        assert dropout_count == 2
        
        embeddings = torch.randn(2, 256)
        output = head(embeddings)
        assert output.shape == (2, 10)
    
    def test_mlp_head_without_dropout(self):
        """Test MLPHead without dropout (default)."""
        head = MLPHead(
            input_dim=256,
            hidden_dims=[128, 64],
            output_dim=10,
            dropout=0.0
        )
        
        # Should have no dropout layers
        dropout_count = sum(1 for module in head.mlp if isinstance(module, nn.Dropout))
        assert dropout_count == 0
    
    def test_mlp_head_with_batch_norm(self):
        """Test MLPHead with batch normalization."""
        head = MLPHead(
            input_dim=256,
            hidden_dims=[128, 64],
            output_dim=10,
            use_batch_norm=True
        )
        
        # Count batch norm layers (should be 2, one after each hidden layer)
        bn_count = sum(1 for module in head.mlp if isinstance(module, nn.BatchNorm1d))
        assert bn_count == 2
        
        embeddings = torch.randn(2, 256)
        output = head(embeddings)
        assert output.shape == (2, 10)
    
    def test_mlp_head_without_batch_norm(self):
        """Test MLPHead without batch normalization (default)."""
        head = MLPHead(
            input_dim=256,
            hidden_dims=[128, 64],
            output_dim=10,
            use_batch_norm=False
        )
        
        # Should have no batch norm layers
        bn_count = sum(1 for module in head.mlp if isinstance(module, nn.BatchNorm1d))
        assert bn_count == 0
    
    def test_mlp_head_with_dropout_and_batch_norm(self):
        """Test MLPHead with both dropout and batch norm."""
        head = MLPHead(
            input_dim=256,
            hidden_dims=[128, 64],
            output_dim=10,
            dropout=0.3,
            use_batch_norm=True
        )
        
        embeddings = torch.randn(2, 256)
        output = head(embeddings)
        assert output.shape == (2, 10)
        
        # Verify both dropout and batch norm are present
        dropout_count = sum(1 for module in head.mlp if isinstance(module, nn.Dropout))
        bn_count = sum(1 for module in head.mlp if isinstance(module, nn.BatchNorm1d))
        assert dropout_count == 2
        assert bn_count == 2
    
    def test_mlp_head_single_hidden_layer(self):
        """Test MLPHead with single hidden layer."""
        head = MLPHead(
            input_dim=512,
            hidden_dims=[256],
            output_dim=10
        )
        
        embeddings = torch.randn(3, 512)
        output = head(embeddings)
        assert output.shape == (3, 10)
    
    def test_mlp_head_multiple_hidden_layers(self):
        """Test MLPHead with multiple hidden layers."""
        head = MLPHead(
            input_dim=1024,
            hidden_dims=[512, 256, 128, 64],
            output_dim=20
        )
        
        embeddings = torch.randn(2, 1024)
        output = head(embeddings)
        assert output.shape == (2, 20)
    
    def test_mlp_head_input_dim_property(self):
        """Test that input_dim property returns correct value."""
        head = MLPHead(
            input_dim=768,
            hidden_dims=[256],
            output_dim=10
        )
        assert head.input_dim == 768
    
    def test_mlp_head_gradient_flow(self):
        """Test that gradients flow through MLPHead."""
        head = MLPHead(
            input_dim=128,
            hidden_dims=[64],
            output_dim=10
        )
        embeddings = torch.randn(2, 128, requires_grad=True)
        output = head(embeddings)
        loss = output.sum()
        loss.backward()
        
        assert embeddings.grad is not None
        # Check that at least some parameters have gradients
        has_grad = any(p.grad is not None for p in head.parameters())
        assert has_grad
    
    def test_mlp_head_different_batch_sizes(self):
        """Test MLPHead with different batch sizes."""
        head = MLPHead(
            input_dim=256,
            hidden_dims=[128, 64],
            output_dim=5
        )
        
        for batch_size in [1, 5, 32, 128]:
            embeddings = torch.randn(batch_size, 256)
            output = head(embeddings)
            assert output.shape == (batch_size, 5)
    
    def test_mlp_head_layer_structure(self):
        """Test that MLPHead has correct layer structure."""
        head = MLPHead(
            input_dim=256,
            hidden_dims=[128, 64],
            output_dim=10,
            activation="relu",
            dropout=0.1,
            use_batch_norm=True
        )
        
        # Expected structure for each hidden layer:
        # Linear -> BatchNorm -> Activation -> Dropout
        # Then final: Linear
        
        layers = list(head.mlp)
        # First hidden layer: Linear(256, 128) -> BN -> ReLU -> Dropout
        assert isinstance(layers[0], nn.Linear)
        assert layers[0].in_features == 256
        assert layers[0].out_features == 128
        assert isinstance(layers[1], nn.BatchNorm1d)
        assert isinstance(layers[2], nn.ReLU)
        assert isinstance(layers[3], nn.Dropout)
        
        # Second hidden layer: Linear(128, 64) -> BN -> ReLU -> Dropout
        assert isinstance(layers[4], nn.Linear)
        assert layers[4].in_features == 128
        assert layers[4].out_features == 64
        
        # Output layer: Linear(64, 10)
        assert isinstance(layers[-1], nn.Linear)
        assert layers[-1].in_features == 64
        assert layers[-1].out_features == 10


class TestIdentityHead:
    """Tests for the IdentityHead class."""
    
    def test_identity_head_initialization(self):
        """Test IdentityHead initialization."""
        head = IdentityHead(input_dim=768)
        assert head.input_dim == 768
    
    def test_identity_head_forward(self):
        """Test IdentityHead forward pass returns input unchanged."""
        head = IdentityHead(input_dim=512)
        embeddings = torch.randn(3, 512)
        output = head(embeddings)
        
        assert output.shape == (3, 512)
        assert torch.equal(output, embeddings)
    
    def test_identity_head_forward_different_shapes(self):
        """Test IdentityHead with different input shapes."""
        head = IdentityHead(input_dim=256)
        
        for batch_size in [1, 5, 32, 128]:
            embeddings = torch.randn(batch_size, 256)
            output = head(embeddings)
            assert output.shape == (batch_size, 256)
            assert torch.equal(output, embeddings)
    
    def test_identity_head_input_dim_property(self):
        """Test that input_dim property returns correct value."""
        head = IdentityHead(input_dim=1024)
        assert head.input_dim == 1024
    
    def test_identity_head_gradient_flow(self):
        """Test that gradients flow through IdentityHead."""
        head = IdentityHead(input_dim=128)
        embeddings = torch.randn(2, 128, requires_grad=True)
        output = head(embeddings)
        loss = output.sum()
        loss.backward()
        
        assert embeddings.grad is not None
    
    def test_identity_head_preserves_tensor_properties(self):
        """Test that IdentityHead preserves tensor properties."""
        head = IdentityHead(input_dim=256)
        
        # Test with requires_grad
        embeddings = torch.randn(2, 256, requires_grad=True)
        output = head(embeddings)
        assert output.requires_grad == embeddings.requires_grad
        
        # Test with device (if CUDA available)
        if torch.cuda.is_available():
            embeddings_cuda = torch.randn(2, 256).cuda()
            head_cuda = head.cuda()
            output_cuda = head_cuda(embeddings_cuda)
            assert output_cuda.device == embeddings_cuda.device


class TestHeadsIntegration:
    """Integration tests for heads working together."""
    
    def test_all_heads_same_input_dim(self):
        """Test that all head types work with the same input dimension."""
        input_dim = 768
        batch_size = 4
        embeddings = torch.randn(batch_size, input_dim)
        
        linear_head = LinearHead(input_dim=input_dim, output_dim=10)
        mlp_head = MLPHead(input_dim=input_dim, hidden_dims=[256], output_dim=10)
        identity_head = IdentityHead(input_dim=input_dim)
        
        linear_out = linear_head(embeddings)
        mlp_out = mlp_head(embeddings)
        identity_out = identity_head(embeddings)
        
        assert linear_out.shape == (batch_size, 10)
        assert mlp_out.shape == (batch_size, 10)
        assert identity_out.shape == (batch_size, input_dim)
        assert torch.equal(identity_out, embeddings)
    
    def test_heads_with_different_input_dims(self):
        """Test heads with various input dimensions."""
        test_dims = [128, 256, 512, 768, 1024]
        
        for dim in test_dims:
            linear_head = LinearHead(input_dim=dim, output_dim=5)
            mlp_head = MLPHead(input_dim=dim, hidden_dims=[dim // 2], output_dim=5)
            identity_head = IdentityHead(input_dim=dim)
            
            embeddings = torch.randn(2, dim)
            
            assert linear_head(embeddings).shape == (2, 5)
            assert mlp_head(embeddings).shape == (2, 5)
            assert identity_head(embeddings).shape == (2, dim)
    
    def test_heads_eval_mode(self):
        """Test that heads work correctly in eval mode."""
        embeddings = torch.randn(2, 256)
        
        linear_head = LinearHead(input_dim=256, output_dim=10)
        mlp_head = MLPHead(input_dim=256, hidden_dims=[128], output_dim=10, dropout=0.5)
        identity_head = IdentityHead(input_dim=256)
        
        # Set to eval mode
        linear_head.eval()
        mlp_head.eval()
        identity_head.eval()
        
        # Should work without errors
        assert linear_head(embeddings).shape == (2, 10)
        assert mlp_head(embeddings).shape == (2, 10)
        assert identity_head(embeddings).shape == (2, 256)
    
    def test_heads_train_mode(self):
        """Test that heads work correctly in train mode."""
        embeddings = torch.randn(2, 256)
        
        linear_head = LinearHead(input_dim=256, output_dim=10)
        mlp_head = MLPHead(input_dim=256, hidden_dims=[128], output_dim=10, dropout=0.5)
        identity_head = IdentityHead(input_dim=256)
        
        # Set to train mode
        linear_head.train()
        mlp_head.train()
        identity_head.train()
        
        # Should work without errors
        assert linear_head(embeddings).shape == (2, 10)
        assert mlp_head(embeddings).shape == (2, 10)
        assert identity_head(embeddings).shape == (2, 256)
