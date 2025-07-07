import torch.nn as nn

def set_encoder_dropout_p(module, dropout_p: float):
    """Recursively updates dropout rate for all nn.Dropout modules."""
    if isinstance(module, nn.Dropout):
        module.p = dropout_p
