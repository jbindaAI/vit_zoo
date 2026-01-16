"""Head implementations for ViT models."""

from abc import ABC, abstractmethod
from typing import List, Union, Literal
import torch
from torch import nn


class BaseHead(nn.Module, ABC):
    """Abstract base class for all classification/regression heads.
    
    All heads must implement:
    1. The `input_dim` property - returns the expected input embedding dimension
    2. The `forward` method - takes embeddings and returns predictions
    
    Users can subclass this to create custom heads (e.g., MLP, UNET decoder,
    attention-based heads, etc.). The factory will automatically validate that
    the head's `input_dim` matches the backbone's embedding dimension.
    
    The head should be designed to accept embeddings of shape
    (batch_size, embedding_dim) where embedding_dim matches the backbone's
    embedding dimension.
    """
    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Property returning expected input dimension.
        
        Returns:
            Expected input dimension of the head. This will be validated
            against the backbone's embedding dimension when creating a model.
        """
        pass
    
    @abstractmethod
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Process embeddings and return predictions.
        
        Args:
            embeddings: Input embeddings tensor of shape (batch_size, embedding_dim)
        
        Returns:
            Predictions tensor of shape (batch_size, output_dim)
        """
        pass


class LinearHead(BaseHead):
    """Simple linear classification/regression head.
    
    Args:
        input_dim: Input embedding dimension
        output_dim: Output dimension (number of classes for classification)
        bias: Whether to use bias in the linear layer
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
    
    @property
    def input_dim(self) -> int:
        """Returns the input dimension of the head."""
        return self._input_dim
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation to embeddings."""
        return self.linear(embeddings)


class MLPHead(BaseHead):
    """Multi-layer perceptron head with configurable depth and activation.
    
    Args:
        input_dim: Input embedding dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (number of classes)
        activation: Activation function. Can be:
            - String literal: 'relu', 'gelu', or 'tanh' (most common)
            - nn.Module: Any PyTorch activation module (e.g., nn.SiLU(), nn.LeakyReLU())
        dropout: Dropout probability (applied after each hidden layer)
        use_batch_norm: Whether to use batch normalization after each hidden layer
    """
    # Mapping from string names to activation classes
    _ACTIVATION_MAP = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: Union[Literal["relu", "gelu", "tanh"], nn.Module] = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False
    ) -> None:
        super().__init__()
        
        if not hidden_dims:
            raise ValueError("hidden_dims must be a non-empty list")
        
        # Parse activation function
        if isinstance(activation, str):
            activation_lower = activation.lower()
            if activation_lower not in self._ACTIVATION_MAP:
                raise ValueError(
                    f"Unknown activation '{activation}'. "
                    f"Supported: {list(self._ACTIVATION_MAP.keys())}"
                )
            self.activation = self._ACTIVATION_MAP[activation_lower]()
        else:
            self.activation = activation
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(self.activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self._input_dim = input_dim
    
    @property
    def input_dim(self) -> int:
        """Returns the input dimension of the head."""
        return self._input_dim
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation to embeddings."""
        return self.mlp(embeddings)


class IdentityHead(BaseHead):
    """Identity head for embedding extraction only.
    
    This head simply returns the input embeddings unchanged,
    useful when you only want to extract embeddings without
    any classification/regression head.
    
    Args:
        input_dim: Input embedding dimension (for validation purposes)
    """
    
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim
    
    @property
    def input_dim(self) -> int:
        """Returns the input dimension of the head."""
        return self._input_dim
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return embeddings unchanged."""
        return embeddings
