"""
Tests for the factory module.
"""

import torch
from vit_zoo import build_model

def test_build_model():
    model = build_model("vanilla_vit", head_dim=2)
    dummy = torch.rand(1, 3, 224, 224)
    out = model(dummy)
    assert out.shape[-1] == 2
