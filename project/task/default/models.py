"""Define our models, and training and eval functions."""
from typing import Dict

import torch.nn as nn


class Model(nn.Module):
    """A PyTorch model."""

    # TODO: define your model here
    pass


def get_model(config: Dict) -> nn.Module:
    """Return a model instance.

    Args:
        config: A dictionary with the model configuration.

    Returns
    -------
    nn.Module
        A PyTorch model.
    """
    return Model()
